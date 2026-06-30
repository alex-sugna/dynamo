// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TrtllmService gRPC entrypoint.
//!
//! Mounts `crate::grpc::service::trtllm::TrtllmServiceImpl` on a tonic server,
//! sharing the model registry with the etcd-backed `ModelWatcher` so workers
//! that join via discovery become routable through this server. Mirrors the
//! KServe entrypoint at `lib/llm/src/entrypoint/input/grpc.rs`.

use std::sync::Arc;

use dynamo_runtime::DistributedRuntime;
use tonic::transport::Server;

use crate::{
    discovery::{ModelManager, ModelWatcher},
    entrypoint::{EngineConfig, RouterConfig, input::common},
    engines::StreamingEngineAdapter,
    grpc::service::{
        kserve::{self, GrpcTuningConfig},
        trtllm::{TrtllmServiceImpl, TrtllmServiceServer},
    },
    http::service::metrics::Metrics,
    namespace::NamespaceFilter,
    types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};

/// Build and run the TrtllmService gRPC server.
pub async fn run(
    distributed_runtime: DistributedRuntime,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    // The TrtllmService gRPC binds to the same port the KServe path uses
    // (`--http-port`). They're mutually exclusive at the Python frontend, so
    // reusing the port keeps the operator interface symmetric.
    let grpc_port = engine_config.local_model().http_port();

    // Reuse KserveService as a state-builder shortcut: it gives us a State
    // (ModelManager + Metrics) and an HTTP service for /metrics, both of which
    // we want here too. The actual KServe gRPC trait is NOT mounted; we mount
    // TrtllmService instead.
    let mut grpc_service_builder = kserve::KserveService::builder()
        .port(grpc_port)
        .http_cancel_token(Some(distributed_runtime.primary_token()))
        .with_request_template(engine_config.local_model().request_template());

    if let Some(http_metrics_port) = engine_config.local_model().http_metrics_port() {
        grpc_service_builder = grpc_service_builder.http_metrics_port(http_metrics_port);
    }

    let kserve_service = match engine_config {
        EngineConfig::Dynamic { ref model, .. } => {
            let svc = grpc_service_builder.build()?;
            let router_config = model.router_config();
            let migration_limit = model.migration_limit();
            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                model.namespace(),
                model.namespace_prefix(),
            );
            run_watcher(
                distributed_runtime.clone(),
                svc.state().manager_clone(),
                router_config.clone(),
                migration_limit,
                namespace_filter,
            )
            .await?;
            svc
        }
        EngineConfig::InProcessText { engine, model, .. } => {
            let svc = grpc_service_builder.build()?;
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = svc.model_manager();
            let checksum = model.card().mdcsum();
            manager.add_completions_model(model.service_name(), checksum, engine.clone())?;
            manager.add_chat_completions_model(model.service_name(), checksum, engine)?;
            svc
        }
        EngineConfig::InProcessTokens {
            engine: inner_engine,
            model,
            ..
        } => {
            let svc = grpc_service_builder.build()?;
            let manager = svc.model_manager();
            let checksum = model.card().mdcsum();

            let tokenizer = model.card().tokenizer()?;
            let chat_pipeline = common::build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine.clone(), tokenizer.clone())
            .await?;
            manager.add_chat_completions_model(model.service_name(), checksum, chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<
                NvCreateCompletionRequest,
                NvCreateCompletionResponse,
            >(model.card(), inner_engine, tokenizer)
            .await?;
            manager.add_completions_model(model.service_name(), checksum, cmpl_pipeline)?;
            svc
        }
    };

    // Bind the TrtllmService server. We deliberately do NOT mount the KServe
    // trait — frontend_args.py validates that --kserve-grpc-server and
    // --trtllm-grpc-server are mutually exclusive.
    let state = kserve_service.state_clone();
    let trtllm_impl = TrtllmServiceImpl::new(state);
    let address = format!("0.0.0.0:{}", grpc_port);

    tracing::info!(%address, "Starting TrtllmService gRPC server");

    let tuning = GrpcTuningConfig::from_env();
    let mut builder = Server::builder();
    if let Some(size) = tuning.initial_connection_window_size {
        builder = builder.initial_connection_window_size(size);
    }
    if let Some(size) = tuning.initial_stream_window_size {
        builder = builder.initial_stream_window_size(size);
    }

    let shutdown_token = distributed_runtime.primary_token();
    let trtllm_server_fut = builder
        .add_service(TrtllmServiceServer::new(trtllm_impl))
        .serve_with_shutdown(address.parse()?, {
            let token = shutdown_token.clone();
            async move { token.cancelled().await }
        });

    // Run the gRPC server alongside the HTTP metrics server, mirroring kserve.
    let http_service = kserve_service.http_service().clone();
    let http_shutdown = shutdown_token.clone();

    tokio::try_join!(
        async move {
            trtllm_server_fut
                .await
                .map_err(|e| anyhow::anyhow!("TrtllmService gRPC server error: {e}"))
        },
        async move { http_service.run(http_shutdown).await },
    )?;

    distributed_runtime.shutdown();
    Ok(())
}

/// Spawns a task that watches for new models in store and registers them with
/// the ModelManager. Identical in shape to `grpc::run_watcher`.
async fn run_watcher(
    runtime: DistributedRuntime,
    model_manager: Arc<ModelManager>,
    router_config: RouterConfig,
    migration_limit: u32,
    namespace_filter: NamespaceFilter,
) -> anyhow::Result<()> {
    let metrics = Arc::new(Metrics::new());
    let mut watch_obj = ModelWatcher::new(
        runtime.clone(),
        model_manager,
        router_config,
        migration_limit,
        None,
        metrics,
    );
    // SMG owns chat templating and serves chat via the completions plane, so
    // Dynamo's own chat engine is optional here: an unparseable chat_template
    // (e.g. GLM-5.2) must not fail registration in this mode.
    watch_obj.set_chat_engine_optional(true);
    tracing::debug!("Waiting for remote model");
    let discovery = runtime.discovery();
    let discovery_stream = discovery
        .list_and_watch(
            dynamo_runtime::discovery::DiscoveryQuery::AllModels,
            Some(runtime.primary_token()),
        )
        .await?;
    let _watcher_task = tokio::spawn(async move {
        watch_obj.watch(discovery_stream, namespace_filter).await;
    });
    Ok(())
}
