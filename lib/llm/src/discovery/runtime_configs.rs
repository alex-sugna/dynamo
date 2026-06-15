// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use tokio::sync::watch;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::prelude::DistributedRuntimeProvider;

use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;

/// Type alias for the runtime config watch receiver.
pub type RuntimeConfigWatch = watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>;

/// Join instance availability and config discovery into a single watch.
///
/// Only includes workers that have BOTH an instance registration AND a runtime config.
/// Spawns a background task that recomputes the joined state whenever either source changes.
/// The returned `watch::Receiver` always contains the latest joined snapshot.
pub async fn runtime_config_watch(endpoint: &Endpoint) -> anyhow::Result<RuntimeConfigWatch> {
    let component = endpoint.component();
    let cancel_token = component.drt().primary_token();

    // Source 1: instance availability (watches DiscoveryQuery::Endpoint)
    let client = endpoint.client().await?;
    let mut instance_ids_rx = client.instance_avail_watcher();

    // Source 2: runtime configs from discovery (watches DiscoveryQuery::EndpointModels)
    let discovery = component.drt().discovery();
    let eid = endpoint.id();
    let stream = discovery
        .list_and_watch(
            DiscoveryQuery::EndpointModels {
                namespace: eid.namespace.clone(),
                component: eid.component.clone(),
                endpoint: eid.name.clone(),
            },
            Some(cancel_token.clone()),
        )
        .await?;
    let mut configs_rx =
        watch_and_extract_field(stream, |card: ModelDeploymentCard| card.runtime_config);

    let (tx, rx) = watch::channel(HashMap::new());
    let eid_for_log = eid.clone();

    tokio::spawn(async move {
        let mut last_count: Option<usize> = None;
        loop {
            let wake_source = tokio::select! {
                _ = cancel_token.cancelled() => break,
                result = instance_ids_rx.changed() => {
                    if result.is_err() { break; }
                    "instance_avail"
                }
                result = configs_rx.changed() => {
                    if result.is_err() { break; }
                    "configs"
                }
            };

            let instances: HashSet<WorkerId> = instance_ids_rx
                .borrow_and_update()
                .iter()
                .copied()
                .collect();
            let configs = configs_rx.borrow_and_update().clone();

            let ready: HashMap<WorkerId, ModelRuntimeConfig> = instances
                .into_iter()
                .filter_map(|id| configs.get(&id).map(|cfg| (id, cfg.clone())))
                .collect();

            // Only send if the joined result actually changed, to avoid waking
            // downstream consumers (wait_for, changed) on no-op recomputations.
            if *tx.borrow() == ready {
                continue;
            }

            // Log the recomputation so we can verify at runtime that the join
            // wakes on report_instance_down (it didn't, pre-shared-avail-fix).
            tracing::info!(
                endpoint = ?eid_for_log,
                wake_source,
                count = ready.len(),
                prev_count = ?last_count,
                "[runtime_config_watch] joined map changed"
            );
            last_count = Some(ready.len());

            // Break if all receivers dropped (e.g., TOCTOU in model_manager discards a duplicate).
            if tx.send(ready).is_err() {
                break;
            }
        }
    });

    Ok(rx)
}
