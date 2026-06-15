// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: (DEP-635) this file should be renamed to system_http_server.rs
//  it is being used not just for status, health, but others like loras management.

use crate::config::HealthStatus;
use crate::config::environment_names::logging as env_logging;
use crate::config::environment_names::runtime::canary as env_canary;
use crate::config::environment_names::runtime::system as env_system;
use crate::instances::list_all_instances;
use crate::logging::make_request_span;
use crate::metrics::MetricsHierarchy;
use crate::traits::DistributedRuntimeProvider;
use axum::{
    Router,
    body::Bytes,
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{any, delete, get, post},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::{net::TcpListener, task::JoinHandle};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;

/// Environment variable for the frontend URL (e.g., "http://localhost:8000")
const FRONTEND_URL_ENV: &str = "DYN_FRONTEND_URL";
/// Environment variable for the model name to use in health checks
const HEALTH_CHECK_MODEL_ENV: &str = "DYN_HEALTH_CHECK_MODEL";
/// Environment variable for e2e health check timeout (seconds)
const E2E_HEALTH_CHECK_TIMEOUT_ENV: &str = "DYN_E2E_HEALTH_CHECK_TIMEOUT";
/// Environment variable for how long to cache a healthy status before re-checking (seconds)
const E2E_LAST_HEALTHY_TIMEOUT_ENV: &str = "DYN_E2E_LAST_HEALTHY_TIMEOUT";
/// Environment variable for the prefill component name (default: "prefill")
const PREFILL_COMPONENT_NAME_ENV: &str = "DYN_PREFILL_COMPONENT_NAME";
/// Default prefill component name
const DEFAULT_PREFILL_COMPONENT_NAME: &str = "prefill";
/// Default e2e health check timeout (30 seconds)
const DEFAULT_E2E_HEALTH_CHECK_TIMEOUT_SECS: u64 = 30;
/// Default last healthy cache timeout (10 seconds)
const DEFAULT_E2E_LAST_HEALTHY_TIMEOUT_SECS: u64 = 10;

/// Get the e2e health check timeout from environment or default
fn get_e2e_health_check_timeout() -> Duration {
    std::env::var(E2E_HEALTH_CHECK_TIMEOUT_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_E2E_HEALTH_CHECK_TIMEOUT_SECS))
}

/// Get the last healthy cache timeout from environment or default
/// Validate an HTTP-200 e2e probe response body. Returns Ok(()) only if
/// the assembled completion shows the engine actually generated tokens —
/// shape-agnostic, so it handles content / reasoning_content (thinking
/// models like Kimi K2.6) / tool_calls / multimodal uniformly. An
/// HTTP-200-with-empty/error body is the case the worker-side
/// push_handler stamp gate doesn't cover: trtllm yields error chunks as
/// data-level FinishReason::Error and the upstream HTTP layer doesn't
/// always promote those to 5xx, so the e2e check itself must inspect
/// the body before declaring "passed" and stamping `last_healthy`
/// (which would feed Bypass A for 30s).
///
/// Gate is `usage.completion_tokens > 0`: definitive proof the engine
/// emitted at least one output token. Earlier versions checked
/// `choices[0].message.content` only, which falsely rejected thinking
/// models that put their first token in `reasoning_content` with
/// `max_completion_tokens: 2`.
async fn validate_e2e_response_body(
    response: reqwest::Response,
) -> std::result::Result<(), String> {
    let body = response
        .text()
        .await
        .map_err(|e| format!("read body: {e}"))?;
    let parsed: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return Err(format!(
                "parse JSON: {e}; body[..500]={}",
                body.chars().take(500).collect::<String>()
            ));
        }
    };
    if let Some(err) = parsed.get("error") {
        return Err(format!(
            "top-level error: {err}; body[..500]={}",
            body.chars().take(500).collect::<String>()
        ));
    }
    let completion_tokens = parsed
        .pointer("/usage/completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if completion_tokens == 0 {
        let finish = parsed
            .pointer("/choices/0/finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("none");
        return Err(format!(
            "completion_tokens=0 (finish_reason={finish}); body[..500]={}",
            body.chars().take(500).collect::<String>()
        ));
    }
    Ok(())
}

fn get_e2e_last_healthy_timeout() -> Duration {
    std::env::var(E2E_LAST_HEALTHY_TIMEOUT_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_E2E_LAST_HEALTHY_TIMEOUT_SECS))
}

/// State for worker e2e health checks
pub struct WorkerE2EHealthState {
    /// Last successful e2e health check timestamp
    last_healthy: RwLock<Option<Instant>>,
    /// Lock to prevent concurrent health checks
    health_lock: Mutex<()>,
    /// HTTP client for making requests to frontend
    http_client: reqwest::Client,
}

impl WorkerE2EHealthState {
    pub fn new() -> Self {
        Self {
            last_healthy: RwLock::new(None),
            health_lock: Mutex::new(()),
            http_client: reqwest::Client::builder()
                .timeout(get_e2e_health_check_timeout())
                .build()
                .expect("Failed to create HTTP client"),
        }
    }
}

impl Default for WorkerE2EHealthState {
    fn default() -> Self {
        Self::new()
    }
}

/// System status server information containing socket address and handle
#[derive(Debug)]
pub struct SystemStatusServerInfo {
    pub socket_addr: std::net::SocketAddr,
    pub handle: Option<Arc<JoinHandle<()>>>,
}

impl SystemStatusServerInfo {
    pub fn new(socket_addr: std::net::SocketAddr, handle: Option<JoinHandle<()>>) -> Self {
        Self {
            socket_addr,
            handle: handle.map(Arc::new),
        }
    }

    pub fn address(&self) -> String {
        self.socket_addr.to_string()
    }

    pub fn hostname(&self) -> String {
        self.socket_addr.ip().to_string()
    }

    pub fn port(&self) -> u16 {
        self.socket_addr.port()
    }
}

impl Clone for SystemStatusServerInfo {
    fn clone(&self) -> Self {
        Self {
            socket_addr: self.socket_addr,
            handle: self.handle.clone(),
        }
    }
}

/// System status server state containing the distributed runtime reference
pub struct SystemStatusState {
    // global drt registry is for printing out the entire Prometheus format output
    root_drt: Arc<crate::DistributedRuntime>,
    // Discovery metadata (only for Kubernetes backend)
    discovery_metadata: Option<Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>>,
}

impl SystemStatusState {
    /// Create new system status server state with the provided distributed runtime
    pub fn new(
        drt: Arc<crate::DistributedRuntime>,
        discovery_metadata: Option<Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            root_drt: drt,
            discovery_metadata,
        })
    }

    /// Get a reference to the distributed runtime
    pub fn drt(&self) -> &crate::DistributedRuntime {
        &self.root_drt
    }

    /// Get a reference to the discovery metadata if available
    pub fn discovery_metadata(
        &self,
    ) -> Option<&Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>> {
        self.discovery_metadata.as_ref()
    }
}

/// Request body for POST /v1/loras
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoadLoraRequest {
    pub lora_name: String,
    pub source: LoraSource,
}

/// Source information for loading a LoRA
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoraSource {
    pub uri: String,
}

/// Response body for LoRA operations
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoraResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loras: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
}

/// Start system status server with metrics support
pub async fn spawn_system_status_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    drt: Arc<crate::DistributedRuntime>,
    discovery_metadata: Option<Arc<tokio::sync::RwLock<crate::discovery::DiscoveryMetadata>>>,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    // Create system status server state with the provided distributed runtime
    let server_state = Arc::new(SystemStatusState::new(drt, discovery_metadata)?);
    let health_path = server_state
        .drt()
        .system_health()
        .lock()
        .health_path()
        .to_string();
    let live_path = server_state
        .drt()
        .system_health()
        .lock()
        .live_path()
        .to_string();

    // Check if LoRA feature is enabled
    let lora_enabled = std::env::var(crate::config::environment_names::llm::DYN_LORA_ENABLED)
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(false);

    // Create e2e health check state for worker-level health checks
    let e2e_health_state = Arc::new(WorkerE2EHealthState::new());

    let mut app = Router::new()
        .route(
            &health_path,
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state)
            }),
        )
        .route(
            &live_path,
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state)
            }),
        )
        .route(
            "/metrics",
            get({
                let state = Arc::clone(&server_state);
                move || metrics_handler(state)
            }),
        )
        .route(
            "/metadata",
            get({
                let state = Arc::clone(&server_state);
                move || metadata_handler(state)
            }),
        )
        .route(
            "/health/prefill",
            get({
                let state = Arc::clone(&server_state);
                let e2e_state = Arc::clone(&e2e_health_state);
                move || prefill_e2e_health_handler(state, e2e_state)
            }),
        )
        .route(
            "/health/decode",
            get({
                let state = Arc::clone(&server_state);
                let e2e_state = Arc::clone(&e2e_health_state);
                move || decode_e2e_health_handler(state, e2e_state)
            }),
        )
        .route(
            "/engine/{*path}",
            any({
                let state = Arc::clone(&server_state);
                move |path, body| engine_route_handler(state, path, body)
            }),
        );

    // Add LoRA routes only if DYN_LORA_ENABLED is set to true
    if lora_enabled {
        app = app
            .route(
                "/v1/loras",
                get({
                    let state = Arc::clone(&server_state);
                    move || list_loras_handler(State(state))
                })
                .post({
                    let state = Arc::clone(&server_state);
                    move |body| load_lora_handler(State(state), body)
                }),
            )
            .route(
                "/v1/loras/{*lora_name}",
                delete({
                    let state = Arc::clone(&server_state);
                    move |path| unload_lora_handler(State(state), path)
                }),
            );
    }

    let app = app
        .fallback(|| async {
            tracing::info!("[fallback handler] called");
            (StatusCode::NOT_FOUND, "Route not found").into_response()
        })
        .layer(TraceLayer::new_for_http().make_span_with(make_request_span));

    let address = format!("{}:{}", host, port);
    tracing::info!("[spawn_system_status_server] binding to: {}", address);

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => {
            // get the actual address and port, print in debug level
            let actual_address = listener.local_addr()?;
            tracing::info!(
                "[spawn_system_status_server] system status server bound to: {}",
                actual_address
            );
            (listener, actual_address)
        }
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", address, e);
            return Err(anyhow::anyhow!("Failed to bind to address: {}", e));
        }
    };
    let (listener, actual_address) = listener;

    let observer = cancel_token.child_token();
    // Spawn the server in the background and return the handle
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
        {
            tracing::error!("System status server error: {}", e);
        }
    });

    Ok((actual_address, handle))
}

/// Health handler with optional active health checking
#[tracing::instrument(skip_all, level = "trace")]
async fn health_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    // Get basic health status
    let system_health = state.drt().system_health();
    let system_health_lock = system_health.lock();
    let (healthy, endpoints) = system_health_lock.get_health_status();
    let uptime = Some(system_health_lock.uptime());
    drop(system_health_lock);

    let healthy_string = if healthy { "ready" } else { "notready" };
    let status_code = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = json!({
        "status": healthy_string,
        "uptime": uptime,
        "endpoints": endpoints,
    });

    tracing::trace!("Response {}", response.to_string());

    (status_code, response.to_string())
}

/// Metrics handler with DistributedRuntime uptime
#[tracing::instrument(skip_all, level = "trace")]
async fn metrics_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    // Get all metrics from the DistributedRuntime.
    // The uptime gauge is updated automatically via a PrometheusUpdateCallback
    // registered in DistributedRuntime::new(), so it is always fresh before scrape.
    //
    // NOTE: We use a multi-registry model (e.g. one registry per endpoint) and merge at scrape time,
    // so /metrics traverses registered child registries and produces a single combined output.
    let response = match state.drt().metrics().prometheus_expfmt() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to get metrics from registry: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to get metrics".to_string(),
            );
        }
    };

    (StatusCode::OK, response)
}

/// Metadata handler
#[tracing::instrument(skip_all, level = "trace")]
async fn metadata_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    // Check if discovery metadata is available
    let metadata = match state.discovery_metadata() {
        Some(metadata) => metadata,
        None => {
            tracing::debug!("Metadata endpoint called but no discovery metadata available");
            return (
                StatusCode::NOT_FOUND,
                "Discovery metadata not available".to_string(),
            )
                .into_response();
        }
    };

    // Read the metadata
    let metadata_guard = metadata.read().await;

    // Serialize to JSON
    match serde_json::to_string(&*metadata_guard) {
        Ok(json) => {
            tracing::trace!("Returning metadata: {} bytes", json.len());
            (StatusCode::OK, json).into_response()
        }
        Err(e) => {
            tracing::error!("Failed to serialize metadata: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to serialize metadata".to_string(),
            )
                .into_response()
        }
    }
}

/// Handler for POST /v1/loras - Load a LoRA adapter
#[tracing::instrument(skip_all, level = "debug")]
async fn load_lora_handler(
    State(state): State<Arc<SystemStatusState>>,
    Json(request): Json<LoadLoraRequest>,
) -> impl IntoResponse {
    tracing::info!("Loading LoRA: {}", request.lora_name);

    // Call the load_lora endpoint for each available backend
    match call_lora_endpoint(
        state.drt(),
        "load_lora",
        json!({
            "lora_name": request.lora_name,
            "source": {
                "uri": request.source.uri
            },
        }),
    )
    .await
    {
        Ok(response) => {
            if response.status == "error" {
                tracing::error!(
                    "Failed to load LoRA {}: {}",
                    request.lora_name,
                    response.message.as_deref().unwrap_or("Unknown error")
                );
                (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
            } else {
                tracing::info!("LoRA loaded successfully: {}", request.lora_name);
                (StatusCode::OK, Json(response))
            }
        }
        Err(e) => {
            tracing::error!("Failed to load LoRA {}: {}", request.lora_name, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoraResponse {
                    status: "error".to_string(),
                    message: Some(e.to_string()),
                    lora_name: Some(request.lora_name),
                    lora_id: None,
                    loras: None,
                    count: None,
                }),
            )
        }
    }
}

/// Handler for DELETE /v1/loras/*lora_name - Unload a LoRA adapter
#[tracing::instrument(skip_all, level = "debug")]
async fn unload_lora_handler(
    State(state): State<Arc<SystemStatusState>>,
    Path(lora_name): Path<String>,
) -> impl IntoResponse {
    // Strip the leading slash from the wildcard capture
    let lora_name = lora_name
        .strip_prefix('/')
        .unwrap_or(&lora_name)
        .to_string();
    tracing::info!("Unloading LoRA: {}", lora_name);

    // Call the unload_lora endpoint for each available backend
    match call_lora_endpoint(
        state.drt(),
        "unload_lora",
        json!({
            "lora_name": lora_name.clone(),
        }),
    )
    .await
    {
        Ok(response) => {
            if response.status == "error" {
                tracing::error!(
                    "Failed to unload LoRA {}: {}",
                    lora_name,
                    response.message.as_deref().unwrap_or("Unknown error")
                );
                (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
            } else {
                tracing::info!("LoRA unloaded successfully: {}", lora_name);
                (StatusCode::OK, Json(response))
            }
        }
        Err(e) => {
            tracing::error!("Failed to unload LoRA {}: {}", lora_name, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoraResponse {
                    status: "error".to_string(),
                    message: Some(e.to_string()),
                    lora_name: Some(lora_name),
                    lora_id: None,
                    loras: None,
                    count: None,
                }),
            )
        }
    }
}

/// Handler for GET /v1/loras - List all LoRA adapters
#[tracing::instrument(skip_all, level = "debug")]
async fn list_loras_handler(State(state): State<Arc<SystemStatusState>>) -> impl IntoResponse {
    tracing::info!("Listing all LoRAs");

    // Call the list_loras endpoint for each available backend
    match call_lora_endpoint(state.drt(), "list_loras", json!({})).await {
        Ok(response) => {
            tracing::info!("Successfully retrieved LoRA list");
            (StatusCode::OK, Json(response))
        }
        Err(e) => {
            tracing::error!("Failed to list LoRAs: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoraResponse {
                    status: "error".to_string(),
                    message: Some(e.to_string()),
                    lora_name: None,
                    lora_id: None,
                    loras: None,
                    count: None,
                }),
            )
        }
    }
}

/// Helper function to call a LoRA management endpoint locally via in-process registry
///
/// This function ONLY uses the local endpoint registry for direct in-process calls.
/// It does NOT fall back to network discovery if the endpoint is not found.
async fn call_lora_endpoint(
    drt: &crate::DistributedRuntime,
    endpoint_name: &str,
    request_body: serde_json::Value,
) -> anyhow::Result<LoraResponse> {
    use crate::engine::AsyncEngine;

    tracing::debug!("Calling local endpoint: '{}'", endpoint_name);

    // Get the endpoint from the local registry (in-process call only)
    let local_registry = drt.local_endpoint_registry();
    let engine = local_registry
        .get(endpoint_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Endpoint '{}' not found in local registry. Make sure it's registered with .register_local_engine()",
                endpoint_name
            )
        })?;

    tracing::debug!(
        "Found endpoint '{}' in local registry, calling directly",
        endpoint_name
    );

    // Call the engine directly without going through the network stack
    let request = crate::pipeline::SingleIn::new(request_body);
    let mut stream = engine.generate(request).await?;

    // Get the first response
    if let Some(response) = stream.next().await {
        let response_data = response.data.unwrap_or_default();

        // Try structured deserialization first, fall back to manual field extraction
        let lora_response = serde_json::from_value::<LoraResponse>(response_data.clone())
            .unwrap_or_else(|_| parse_lora_response(&response_data));

        return Ok(lora_response);
    }

    anyhow::bail!("No response received from endpoint '{}'", endpoint_name)
}

/// Helper to parse response data into LoraResponse
fn parse_lora_response(response_data: &serde_json::Value) -> LoraResponse {
    LoraResponse {
        status: response_data
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("success")
            .to_string(),
        message: response_data
            .get("message")
            .and_then(|m| m.as_str())
            .map(|s| s.to_string()),
        lora_name: response_data
            .get("lora_name")
            .and_then(|n| n.as_str())
            .map(|s| s.to_string()),
        lora_id: response_data.get("lora_id").and_then(|id| id.as_u64()),
        loras: response_data.get("loras").cloned(),
        count: response_data
            .get("count")
            .and_then(|c| c.as_u64())
            .map(|c| c as usize),
    }
}

/// Engine route handler for /engine/* routes
///
/// This handler looks up registered callbacks in the engine routes registry
/// and invokes them with the request body, returning the response as JSON.
#[tracing::instrument(skip_all, level = "trace", fields(path = %path))]
async fn engine_route_handler(
    state: Arc<SystemStatusState>,
    Path(path): Path<String>,
    body: Bytes,
) -> impl IntoResponse {
    tracing::trace!("Engine route request to /engine/{}", path);

    // Parse body as JSON (empty object for GET/empty body)
    let body_json: serde_json::Value = if body.is_empty() {
        serde_json::json!({})
    } else {
        match serde_json::from_slice(&body) {
            Ok(json) => json,
            Err(e) => {
                tracing::warn!("Invalid JSON in request body: {}", e);
                return (
                    StatusCode::BAD_REQUEST,
                    json!({
                        "error": "Invalid JSON",
                        "message": format!("{}", e)
                    })
                    .to_string(),
                )
                    .into_response();
            }
        }
    };

    // Look up callback
    let callback = match state.drt().engine_routes().get(&path) {
        Some(cb) => cb,
        None => {
            tracing::debug!("Route /engine/{} not found", path);
            return (
                StatusCode::NOT_FOUND,
                json!({
                    "error": "Route not found",
                    "message": format!("Route /engine/{} not found", path)
                })
                .to_string(),
            )
                .into_response();
        }
    };

    // Call callback (it's async, so await it)
    match callback(body_json).await {
        Ok(response) => {
            tracing::trace!("Engine route handler succeeded for /engine/{}", path);
            (StatusCode::OK, response.to_string()).into_response()
        }
        Err(e) => {
            tracing::error!("Engine route handler error for /engine/{}: {}", path, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({
                    "error": "Handler error",
                    "message": format!("{}", e)
                })
                .to_string(),
            )
                .into_response()
        }
    }
}

/// E2E health check handler for prefill workers
/// Makes an HTTP request to the frontend to trigger a full e2e generation
#[tracing::instrument(skip_all, level = "debug")]
async fn prefill_e2e_health_handler(
    state: Arc<SystemStatusState>,
    e2e_state: Arc<WorkerE2EHealthState>,
) -> impl IntoResponse {
    worker_e2e_health_check(state, e2e_state, "prefill").await
}

/// E2E health check handler for decode workers
/// Tries multiple prefill workers until one succeeds
#[tracing::instrument(skip_all, level = "debug")]
async fn decode_e2e_health_handler(
    state: Arc<SystemStatusState>,
    e2e_state: Arc<WorkerE2EHealthState>,
) -> impl IntoResponse {
    decode_e2e_health_check_with_prefill_fallback(state, e2e_state).await
}

/// Decode worker e2e health check that tries multiple prefill workers
/// Returns healthy if ANY prefill worker succeeds (since decode is working)
/// Returns unhealthy only if ALL prefill workers fail
async fn decode_e2e_health_check_with_prefill_fallback(
    state: Arc<SystemStatusState>,
    e2e_state: Arc<WorkerE2EHealthState>,
) -> impl IntoResponse {
    let last_healthy_timeout = get_e2e_last_healthy_timeout();

    // Get this decode worker's instance_id for logging
    let decode_instance_id = state.drt().connection_id();
    tracing::info!(
        "[decode e2e health] Starting health check (decode_instance_id={}, last_healthy_timeout={:?})",
        decode_instance_id,
        last_healthy_timeout
    );

    // Acquire lock to prevent concurrent health checks
    let _guard = e2e_state.health_lock.lock().await;
    tracing::info!("[decode e2e health] Acquired health check lock");

    // Check if recent health check is still valid (cached)
    {
        let last_healthy = e2e_state.last_healthy.read().await;
        if let Some(last) = *last_healthy {
            let age = last.elapsed();
            tracing::info!(
                "[decode e2e health] Last healthy check was {:?} ago (timeout: {:?})",
                age,
                last_healthy_timeout
            );
            if age < last_healthy_timeout {
                tracing::info!(
                    "[decode e2e health] Returning cached healthy response (age: {:?})",
                    age
                );
                return (
                    StatusCode::OK,
                    json!({
                        "status": "healthy",
                        "worker_type": "decode",
                        "decode_instance_id": decode_instance_id,
                        "cached": true,
                        "last_check_age_secs": age.as_secs_f64()
                    })
                    .to_string(),
                )
                    .into_response();
            }
        } else {
            tracing::info!("[decode e2e health] No previous healthy check recorded");
        }
    }

    // Check if a recent user request completed successfully (active traffic = healthy)
    {
        let last_req_age = state.drt().system_health().lock().last_successful_request_age();
        if let Some(age) = last_req_age {
            if age < last_healthy_timeout {
                tracing::info!(
                    "[decode e2e health] Active traffic detected: last successful request was {:?} ago, returning healthy",
                    age
                );
                return (
                    StatusCode::OK,
                    json!({
                        "status": "healthy",
                        "worker_type": "decode",
                        "decode_instance_id": decode_instance_id,
                        "cached": true,
                        "source": "active_traffic",
                        "last_request_age_secs": age.as_secs_f64()
                    })
                    .to_string(),
                )
                    .into_response();
            }
        }
    }

    // Get frontend URL from environment
    let frontend_url = match std::env::var(FRONTEND_URL_ENV) {
        Ok(url) => {
            tracing::info!("[decode e2e health] Frontend URL: {}", url);
            url
        }
        Err(_) => {
            tracing::warn!(
                "[decode e2e health] Skipped: {} not set",
                FRONTEND_URL_ENV
            );
            return (
                StatusCode::OK,
                json!({
                    "status": "healthy",
                    "worker_type": "decode",
                    "decode_instance_id": decode_instance_id,
                    "e2e_check": "skipped",
                    "reason": format!("{} not configured", FRONTEND_URL_ENV)
                })
                .to_string(),
            )
                .into_response();
        }
    };

    // Get model name from environment
    let model = match std::env::var(HEALTH_CHECK_MODEL_ENV) {
        Ok(m) => {
            tracing::info!("[decode e2e health] Model: {}", m);
            m
        }
        Err(_) => {
            tracing::warn!(
                "[decode e2e health] Skipped: {} not set",
                HEALTH_CHECK_MODEL_ENV
            );
            return (
                StatusCode::OK,
                json!({
                    "status": "healthy",
                    "worker_type": "decode",
                    "decode_instance_id": decode_instance_id,
                    "e2e_check": "skipped",
                    "reason": format!("{} not configured", HEALTH_CHECK_MODEL_ENV)
                })
                .to_string(),
            )
                .into_response();
        }
    };

    // Get prefill worker instance IDs - either from environment or auto-discovery
    // Format for env var: comma-separated list of instance IDs, e.g., "123,456,789"
    let prefill_instance_ids: Vec<u64> = match std::env::var("DYN_PREFILL_INSTANCE_IDS") {
        Ok(ids_str) => {
            let ids: Vec<u64> = ids_str
                .split(',')
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .collect();
            tracing::info!(
                "[decode e2e health] DYN_PREFILL_INSTANCE_IDS configured with {} prefill workers: {:?}",
                ids.len(),
                ids
            );
            ids
        }
        Err(_) => {
            // Auto-discover prefill workers from the discovery service
            tracing::info!(
                "[decode e2e health] DYN_PREFILL_INSTANCE_IDS not set, auto-discovering prefill workers..."
            );

            let prefill_component_name = std::env::var(PREFILL_COMPONENT_NAME_ENV)
                .unwrap_or_else(|_| DEFAULT_PREFILL_COMPONENT_NAME.to_string());

            tracing::info!(
                "[decode e2e health] Looking for prefill workers with component name: {}",
                prefill_component_name
            );

            // Scope to this DGD's dynamo namespace so a sibling deployment sharing the
            // etcd cluster doesn't bleed in (operator sets DYN_NAMESPACE = {k8sNs}-{dgdName}).
            // Unset env ⇒ empty string ⇒ no instance matches ⇒ "no prefills found" path fires.
            let target_ns = std::env::var("DYN_NAMESPACE").unwrap_or_default();
            match list_all_instances(state.drt().discovery()).await {
                Ok(instances) => {
                    let foreign = instances.iter().filter(|i| i.namespace != target_ns).count();
                    if foreign > 0 {
                        tracing::error!(
                            local_namespace = %target_ns, foreign_count = foreign,
                            "[decode e2e health] cross-namespace discovery sightings dropped"
                        );
                    }
                    let prefill_ids: Vec<u64> = instances
                        .iter()
                        .filter(|instance| instance.namespace == target_ns)
                        .filter(|instance| instance.component == prefill_component_name)
                        .map(|instance| instance.instance_id)
                        .collect();

                    if prefill_ids.is_empty() {
                        tracing::warn!(
                            "[decode e2e health] No prefill workers found with component name '{}'. Found {} total instances.",
                            prefill_component_name,
                            instances.len()
                        );
                        // Log what components we did find for debugging
                        let components: std::collections::HashSet<_> = instances.iter().map(|i| i.component.as_str()).collect();
                        tracing::info!(
                            "[decode e2e health] Available components in discovery: {:?}",
                            components
                        );
                    } else {
                        tracing::info!(
                            "[decode e2e health] Auto-discovered {} prefill workers: {:?}",
                            prefill_ids.len(),
                            prefill_ids
                        );
                    }
                    prefill_ids
                }
                Err(e) => {
                    tracing::warn!(
                        "[decode e2e health] Failed to query discovery service: {}. Will send single request without prefill targeting.",
                        e
                    );
                    vec![]
                }
            }
        }
    };

    let chat_url = format!("{}/v1/chat/completions", frontend_url.trim_end_matches('/'));
    let start_time = Instant::now();

    // If we have specific prefill instance IDs, try each one
    if !prefill_instance_ids.is_empty() {
        tracing::info!(
            "[decode e2e health] Starting prefill worker fallback: will try {} workers sequentially until one succeeds",
            prefill_instance_ids.len()
        );

        let mut last_error = String::new();
        let mut tried_count = 0;

        for prefill_id in &prefill_instance_ids {
            tried_count += 1;

            let health_request = json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 2,
                "stream": false,
                "temperature": 0.0,
                "nvext": {
                    "backend_instance_id": prefill_id,
                    "decode_instance_id": decode_instance_id
                }
            });

            tracing::info!(
                "[decode e2e health] Attempt {}/{}: Trying prefill worker {} | URL: {} | backend_instance_id: {} | decode_instance_id: {}",
                tried_count,
                prefill_instance_ids.len(),
                prefill_id,
                chat_url,
                prefill_id,
                decode_instance_id
            );

            let result = e2e_state
                .http_client
                .post(&chat_url)
                .header("Content-Type", "application/json")
                .header("X-Health-Check", "true")
                .json(&health_request)
                .send()
                .await;

            match result {
                Ok(response) if response.status().is_success() => {
                    match validate_e2e_response_body(response).await {
                        Ok(()) => {
                            let elapsed = start_time.elapsed();
                            *e2e_state.last_healthy.write().await = Some(Instant::now());

                            tracing::info!(
                                "[decode e2e health] PASSED via prefill worker {} in {:?} (decode_instance_id={}, attempts={}/{})",
                                prefill_id,
                                elapsed,
                                decode_instance_id,
                                tried_count,
                                prefill_instance_ids.len()
                            );

                            return (
                                StatusCode::OK,
                                json!({
                                    "status": "healthy",
                                    "worker_type": "decode",
                                    "decode_instance_id": decode_instance_id,
                                    "e2e_check": "passed",
                                    "prefill_worker_used": prefill_id,
                                    "prefill_workers_tried": tried_count,
                                    "prefill_workers_total": prefill_instance_ids.len(),
                                    "latency_ms": elapsed.as_millis()
                                })
                                .to_string(),
                            )
                                .into_response();
                        }
                        Err(body_err) => {
                            last_error = format!("HTTP 200 but body invalid: {body_err}");
                            tracing::warn!(
                                "[decode e2e health] Attempt {}/{}: Prefill worker {} returned HTTP 200 with bad body: {}",
                                tried_count,
                                prefill_instance_ids.len(),
                                prefill_id,
                                body_err
                            );
                        }
                    }
                }
                Ok(response) => {
                    let status = response.status();
                    let error_body = response.text().await.unwrap_or_default();
                    last_error = format!("HTTP {} - {}", status, error_body);
                    tracing::warn!(
                        "[decode e2e health] Attempt {}/{}: Prefill worker {} FAILED: {}",
                        tried_count,
                        prefill_instance_ids.len(),
                        prefill_id,
                        last_error
                    );
                }
                Err(e) => {
                    last_error = if e.is_timeout() {
                        "Request timed out".to_string()
                    } else if e.is_connect() {
                        format!("Connection failed: {}", e)
                    } else {
                        e.to_string()
                    };
                    tracing::warn!(
                        "[decode e2e health] Attempt {}/{}: Prefill worker {} FAILED: {}",
                        tried_count,
                        prefill_instance_ids.len(),
                        prefill_id,
                        last_error
                    );
                }
            }
        }

        // All prefill workers failed
        let elapsed = start_time.elapsed();
        tracing::error!(
            "[decode e2e health] FAILED: All {} prefill workers failed (decode_instance_id={}, total_time={:?}). Last error: {}",
            prefill_instance_ids.len(),
            decode_instance_id,
            elapsed,
            last_error
        );

        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({
                "status": "unhealthy",
                "worker_type": "decode",
                "decode_instance_id": decode_instance_id,
                "e2e_check": "failed",
                "error": format!("All {} prefill workers failed", prefill_instance_ids.len()),
                "last_error": last_error,
                "prefill_workers_tried": tried_count,
                "prefill_instance_ids": prefill_instance_ids,
                "latency_ms": elapsed.as_millis()
            })
            .to_string(),
        )
            .into_response();
    }

    // No specific prefill IDs configured - send single request without prefill targeting
    // but still pin the decode worker via decode_instance_id
    let health_request = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 2,
        "stream": false,
        "temperature": 0.0,
        "nvext": {
            "decode_instance_id": decode_instance_id
        }
    });

    tracing::info!(
        "[decode e2e health] Sending single request (no prefill targeting, decode_instance_id={}) to {} | Request payload: {}",
        decode_instance_id,
        chat_url,
        serde_json::to_string(&health_request).unwrap_or_else(|_| "failed to serialize".to_string())
    );

    let result = e2e_state
        .http_client
        .post(&chat_url)
        .header("Content-Type", "application/json")
        .header("X-Health-Check", "true")
        .json(&health_request)
        .send()
        .await;

    let elapsed = start_time.elapsed();

    match result {
        Ok(response) => {
            let status = response.status();
            tracing::info!(
                "[decode e2e health] Received response: HTTP {} in {:?}",
                status,
                elapsed
            );

            if status.is_success() {
                match validate_e2e_response_body(response).await {
                    Ok(()) => {
                        *e2e_state.last_healthy.write().await = Some(Instant::now());

                        tracing::info!(
                            "[decode e2e health] PASSED in {:?} (decode_instance_id={}, no prefill targeting)",
                            elapsed,
                            decode_instance_id
                        );

                        return (
                            StatusCode::OK,
                            json!({
                                "status": "healthy",
                                "worker_type": "decode",
                                "decode_instance_id": decode_instance_id,
                                "e2e_check": "passed",
                                "prefill_targeting": "none",
                                "latency_ms": elapsed.as_millis()
                            })
                            .to_string(),
                        )
                            .into_response();
                    }
                    Err(body_err) => {
                        tracing::warn!(
                            "[decode e2e health] FAILED: HTTP 200 but body invalid: {} (decode_instance_id={})",
                            body_err,
                            decode_instance_id
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            json!({
                                "status": "unhealthy",
                                "worker_type": "decode",
                                "decode_instance_id": decode_instance_id,
                                "e2e_check": "failed",
                                "error": format!("HTTP 200 with empty/error body: {body_err}"),
                                "latency_ms": elapsed.as_millis()
                            })
                            .to_string(),
                        )
                            .into_response();
                    }
                }
            } else {
                let error_body = response.text().await.unwrap_or_default();
                tracing::warn!(
                    "[decode e2e health] FAILED: HTTP {} - {} (decode_instance_id={})",
                    status,
                    error_body,
                    decode_instance_id
                );

                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({
                        "status": "unhealthy",
                        "worker_type": "decode",
                        "decode_instance_id": decode_instance_id,
                        "e2e_check": "failed",
                        "error": format!("HTTP {}: {}", status, error_body),
                        "latency_ms": elapsed.as_millis()
                    })
                    .to_string(),
                )
                    .into_response()
            }
        }
        Err(e) => {
            let error_msg = if e.is_timeout() {
                "Request timed out".to_string()
            } else if e.is_connect() {
                format!("Failed to connect to frontend: {}", e)
            } else {
                e.to_string()
            };

            tracing::warn!(
                "[decode e2e health] FAILED: {} (decode_instance_id={})",
                error_msg,
                decode_instance_id
            );

            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({
                    "status": "unhealthy",
                    "worker_type": "decode",
                    "decode_instance_id": decode_instance_id,
                    "e2e_check": "failed",
                    "error": error_msg,
                    "latency_ms": elapsed.as_millis()
                })
                .to_string(),
            )
                .into_response()
        }
    }
}

/// E2E health check logic for prefill workers
/// Routes the health check request to THIS specific prefill worker using backend_instance_id
async fn worker_e2e_health_check(
    _state: Arc<SystemStatusState>,
    e2e_state: Arc<WorkerE2EHealthState>,
    worker_type: &str,
) -> impl IntoResponse {
    let last_healthy_timeout = get_e2e_last_healthy_timeout();

    tracing::info!(
        "[{} e2e health] Starting health check (last_healthy_timeout: {:?})",
        worker_type,
        last_healthy_timeout
    );

    // Acquire lock to prevent concurrent health checks
    let _guard = e2e_state.health_lock.lock().await;
    tracing::info!("[{} e2e health] Acquired health check lock", worker_type);

    // Check if recent health check is still valid (cached)
    {
        let last_healthy = e2e_state.last_healthy.read().await;
        if let Some(last) = *last_healthy {
            let age = last.elapsed();
            tracing::info!(
                "[{} e2e health] Last healthy check was {:?} ago (timeout: {:?})",
                worker_type,
                age,
                last_healthy_timeout
            );
            if age < last_healthy_timeout {
                tracing::info!(
                    "[{} e2e health] Returning cached healthy response (age: {:?})",
                    worker_type,
                    age
                );
                return (
                    StatusCode::OK,
                    json!({
                        "status": "healthy",
                        "worker_type": worker_type,
                        "cached": true,
                        "last_check_age_secs": age.as_secs_f64()
                    })
                    .to_string(),
                )
                    .into_response();
            }
        } else {
            tracing::info!(
                "[{} e2e health] No previous healthy check recorded",
                worker_type
            );
        }
    }

    // Check if a recent user request completed successfully (active traffic = healthy)
    {
        let last_req_age = _state.drt().system_health().lock().last_successful_request_age();
        if let Some(age) = last_req_age {
            if age < last_healthy_timeout {
                tracing::info!(
                    "[{} e2e health] Active traffic detected: last successful request was {:?} ago, returning healthy",
                    worker_type,
                    age
                );
                return (
                    StatusCode::OK,
                    json!({
                        "status": "healthy",
                        "worker_type": worker_type,
                        "cached": true,
                        "source": "active_traffic",
                        "last_request_age_secs": age.as_secs_f64()
                    })
                    .to_string(),
                )
                    .into_response();
            }
        }
    }

    // Get frontend URL from environment
    let frontend_url = match std::env::var(FRONTEND_URL_ENV) {
        Ok(url) => {
            tracing::info!("[{} e2e health] Frontend URL: {}", worker_type, url);
            url
        }
        Err(_) => {
            tracing::warn!(
                "[{} e2e health] Skipped: {} not set. Set this to enable e2e health checks.",
                worker_type,
                FRONTEND_URL_ENV
            );
            return (
                StatusCode::OK,
                json!({
                    "status": "healthy",
                    "worker_type": worker_type,
                    "e2e_check": "skipped",
                    "reason": format!("{} not configured", FRONTEND_URL_ENV)
                })
                .to_string(),
            )
                .into_response();
        }
    };

    // Get model name from environment
    let model = match std::env::var(HEALTH_CHECK_MODEL_ENV) {
        Ok(m) => {
            tracing::info!("[{} e2e health] Model: {}", worker_type, m);
            m
        }
        Err(_) => {
            tracing::warn!(
                "[{} e2e health] Skipped: {} not set",
                worker_type,
                HEALTH_CHECK_MODEL_ENV
            );
            return (
                StatusCode::OK,
                json!({
                    "status": "healthy",
                    "worker_type": worker_type,
                    "e2e_check": "skipped",
                    "reason": format!("{} not configured", HEALTH_CHECK_MODEL_ENV)
                })
                .to_string(),
            )
                .into_response();
        }
    };

    // Get this worker's instance_id for routing (prefill workers only)
    let instance_id = _state.drt().connection_id();
    tracing::info!(
        "[{} e2e health] This worker's instance_id (connection_id): {}",
        worker_type,
        instance_id
    );

    // Create minimal chat completion request
    // For prefill workers, include backend_instance_id to route to THIS specific worker
    let health_request = if worker_type == "prefill" {
        tracing::info!(
            "[prefill e2e health] Creating request with backend_instance_id={} to route to THIS worker",
            instance_id
        );
        json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 2,
            "stream": false,
            "temperature": 0.0,
            "nvext": {
                "backend_instance_id": instance_id
            }
        })
    } else {
        // For decode workers, don't specify backend_instance_id initially
        // (we'll handle this differently below)
        tracing::info!(
            "[decode e2e health] Creating request without backend_instance_id (will use load balancer)"
        );
        json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 2,
            "stream": false,
            "temperature": 0.0
        })
    };

    let chat_url = format!("{}/v1/chat/completions", frontend_url.trim_end_matches('/'));
    let start_time = Instant::now();

    tracing::info!(
        "[{} e2e health] Sending request to {} | Request payload: {}",
        worker_type,
        chat_url,
        serde_json::to_string(&health_request).unwrap_or_else(|_| "failed to serialize".to_string())
    );

    // Send request to frontend
    let result = e2e_state
        .http_client
        .post(&chat_url)
        .header("Content-Type", "application/json")
        .header("X-Health-Check", "true")
        .json(&health_request)
        .send()
        .await;

    let elapsed = start_time.elapsed();

    match result {
        Ok(response) => {
            let status = response.status();
            tracing::info!(
                "[{} e2e health] Received response: HTTP {} in {:?}",
                worker_type,
                status,
                elapsed
            );

            if status.is_success() {
                match validate_e2e_response_body(response).await {
                    Ok(()) => {
                        *e2e_state.last_healthy.write().await = Some(Instant::now());

                        tracing::info!(
                            "[{} e2e health] PASSED in {:?} (instance_id={})",
                            worker_type,
                            elapsed,
                            instance_id
                        );

                        return (
                            StatusCode::OK,
                            json!({
                                "status": "healthy",
                                "worker_type": worker_type,
                                "instance_id": instance_id,
                                "e2e_check": "passed",
                                "latency_ms": elapsed.as_millis()
                            })
                            .to_string(),
                        )
                            .into_response();
                    }
                    Err(body_err) => {
                        tracing::warn!(
                            "[{} e2e health] FAILED: HTTP 200 but body invalid: {} (instance_id={})",
                            worker_type,
                            body_err,
                            instance_id
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            json!({
                                "status": "unhealthy",
                                "worker_type": worker_type,
                                "instance_id": instance_id,
                                "e2e_check": "failed",
                                "error": format!("HTTP 200 with empty/error body: {body_err}"),
                                "latency_ms": elapsed.as_millis()
                            })
                            .to_string(),
                        )
                            .into_response();
                    }
                }
            } else {
                let error_body = response.text().await.unwrap_or_default();
                tracing::warn!(
                    "[{} e2e health] FAILED: HTTP {} - {} (instance_id={})",
                    worker_type,
                    status,
                    error_body,
                    instance_id
                );

                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({
                        "status": "unhealthy",
                        "worker_type": worker_type,
                        "instance_id": instance_id,
                        "e2e_check": "failed",
                        "error": format!("HTTP {}: {}", status, error_body),
                        "latency_ms": elapsed.as_millis()
                    })
                    .to_string(),
                )
                    .into_response()
            }
        }
        Err(e) => {
            let error_msg = if e.is_timeout() {
                "Request timed out".to_string()
            } else if e.is_connect() {
                format!("Failed to connect to frontend: {}", e)
            } else {
                e.to_string()
            };

            tracing::warn!(
                "[{} e2e health] FAILED: {} (instance_id={})",
                worker_type,
                error_msg,
                instance_id
            );

            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({
                    "status": "unhealthy",
                    "worker_type": worker_type,
                    "instance_id": instance_id,
                    "e2e_check": "failed",
                    "error": error_msg,
                    "latency_ms": elapsed.as_millis()
                })
                .to_string(),
            )
                .into_response()
        }
    }
}

// Regular tests: cargo test system_status_server --lib
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

    // This is a basic test to verify the HTTP server is working before testing other more complicated tests
    #[tokio::test]
    async fn test_http_server_lifecycle() {
        let cancel_token = CancellationToken::new();
        let cancel_token_for_server = cancel_token.clone();

        // Test basic HTTP server lifecycle without DistributedRuntime
        let app = Router::new().route("/test", get(|| async { (StatusCode::OK, "test") }));

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(cancel_token_for_server.cancelled_owned())
                .await;
        });

        // server starts immediately, no need to wait

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }
}

// Integration tests: cargo test system_status_server --lib --features integration
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::config::environment_names::logging as env_logging;
    use crate::config::environment_names::runtime::canary as env_canary;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::metrics::MetricsHierarchy;
    use anyhow::Result;
    use rstest::rstest;
    use std::sync::Arc;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_uptime_from_system_health() {
        // Test that uptime is available from SystemHealth
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            // Get uptime from SystemHealth
            let uptime = drt.system_health().lock().uptime();
            // Uptime should exist (even if close to zero)
            assert!(uptime.as_nanos() > 0 || uptime.is_zero());

            // Sleep briefly and check uptime increases
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let uptime_after = drt.system_health().lock().uptime();
            assert!(uptime_after > uptime);
        })
        .await;
    }

    #[tokio::test]
    async fn test_runtime_metrics_initialization_and_namespace() {
        // Test that metrics have correct namespace
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;
            // SystemStatusState is already created in distributed.rs
            // so we don't need to create it again here

            // The uptime_seconds metric should already be registered and available
            let response = drt.metrics().prometheus_expfmt().unwrap();
            println!("Full metrics response:\n{}", response);

            // Check that uptime_seconds metric is present with correct namespace
            assert!(
                response.contains("# HELP dynamo_component_uptime_seconds"),
                "Should contain uptime_seconds help text"
            );
            assert!(
                response.contains("# TYPE dynamo_component_uptime_seconds gauge"),
                "Should contain uptime_seconds type"
            );
            assert!(
                response.contains("dynamo_component_uptime_seconds"),
                "Should contain uptime_seconds metric with correct namespace"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_uptime_gauge_updates() {
        // Test that the uptime gauge is properly updated and increases over time
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            // Get initial uptime
            let initial_uptime = drt.system_health().lock().uptime();

            // Update the gauge with initial value
            drt.system_health().lock().update_uptime_gauge();

            // Sleep for 100ms
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            // Get uptime after sleep
            let uptime_after_sleep = drt.system_health().lock().uptime();

            // Update the gauge again
            drt.system_health().lock().update_uptime_gauge();

            // Verify uptime increased by at least 100ms
            let elapsed = uptime_after_sleep - initial_uptime;
            assert!(
                elapsed >= std::time::Duration::from_millis(100),
                "Uptime should have increased by at least 100ms after sleep, but only increased by {:?}",
                elapsed
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_http_requests_fail_when_system_disabled() {
        // Test that system status server is not running when disabled
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            let drt = create_test_drt_async().await;

            // Verify that system status server info is None when disabled
            let system_info = drt.system_status_server_info();
            assert!(
                system_info.is_none(),
                "System status server should not be running when disabled"
            );

            println!("✓ System status server correctly disabled when not enabled");
        })
        .await;
    }

    /// This test verifies the health and liveness endpoints of the system status server.
    /// It checks that the endpoints respond with the correct HTTP status codes and bodies
    /// based on the initial health status and any custom endpoint paths provided via environment variables.
    /// The test is parameterized using multiple #[case] attributes to cover various scenarios,
    /// including different initial health states ("ready" and "notready"), default and custom endpoint paths,
    /// and expected response codes and bodies.
    #[rstest]
    #[case("ready", 200, "ready", None, None, 3)]
    #[case("notready", 503, "notready", None, None, 3)]
    #[case("ready", 200, "ready", Some("/custom/health"), Some("/custom/live"), 5)]
    #[case(
        "notready",
        503,
        "notready",
        Some("/custom/health"),
        Some("/custom/live"),
        5
    )]
    #[tokio::test]
    #[cfg(feature = "integration")]
    async fn test_health_endpoints(
        #[case] starting_health_status: &'static str,
        #[case] expected_status: u16,
        #[case] expected_body: &'static str,
        #[case] custom_health_path: Option<&'static str>,
        #[case] custom_live_path: Option<&'static str>,
        #[case] expected_num_tests: usize,
    ) {
        use std::sync::Arc;
        // use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // use reqwest for HTTP requests

        // Closure call is needed here to satisfy async_with_vars

        crate::logging::init();

        #[allow(clippy::redundant_closure_call)]
        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (
                    env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS,
                    Some(starting_health_status),
                ),
                (env_system::DYN_SYSTEM_HEALTH_PATH, custom_health_path),
                (env_system::DYN_SYSTEM_LIVE_PATH, custom_live_path),
            ],
            (async || {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;

                let client = reqwest::Client::new();

                // Prepare test cases
                let mut test_cases = vec![];
                match custom_health_path {
                    None => {
                        // When using default paths, test the default paths
                        test_cases.push(("/health", expected_status, expected_body));
                    }
                    Some(chp) => {
                        // When using custom paths, default paths should not exist
                        test_cases.push(("/health", 404, "Route not found"));
                        test_cases.push((chp, expected_status, expected_body));
                    }
                }
                match custom_live_path {
                    None => {
                        // When using default paths, test the default paths
                        test_cases.push(("/live", expected_status, expected_body));
                    }
                    Some(clp) => {
                        // When using custom paths, default paths should not exist
                        test_cases.push(("/live", 404, "Route not found"));
                        test_cases.push((clp, expected_status, expected_body));
                    }
                }
                test_cases.push(("/someRandomPathNotFoundHere", 404, "Route not found"));
                assert_eq!(test_cases.len(), expected_num_tests);

                for (path, expect_status, expect_body) in test_cases {
                    println!("[test] Sending request to {}", path);
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    println!(
                        "[test] Response for {}: status={}, body={:?}",
                        path, status, body
                    );
                    assert_eq!(
                        status, expect_status,
                        "Response: status={}, body={:?}",
                        status, body
                    );
                    assert!(
                        body.contains(expect_body),
                        "Response: status={}, body={:?}",
                        status,
                        body
                    );
                }
            })(),
        )
        .await;
    }

    #[tokio::test]
    async fn test_health_endpoint_tracing() -> Result<()> {
        use std::sync::Arc;

        // Closure call is needed here to satisfy async_with_vars

        #[allow(clippy::redundant_closure_call)]
        let _ = temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS, Some("ready")),
                (env_logging::DYN_LOGGING_JSONL, Some("1")),
                (env_logging::DYN_LOG, Some("trace")),
            ],
            (async || {
                // TODO Add proper testing for
                // trace id and parent id

                crate::logging::init();

                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;
                let client = reqwest::Client::new();
                for path in [("/health"), ("/live"), ("/someRandomPathNotFoundHere")] {
                    let traceparent_value =
                        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
                    let tracestate_value = "vendor1=opaqueValue1,vendor2=opaqueValue2";
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::HeaderName::from_static("traceparent"),
                        reqwest::header::HeaderValue::from_str(traceparent_value)?,
                    );
                    headers.insert(
                        reqwest::header::HeaderName::from_static("tracestate"),
                        reqwest::header::HeaderValue::from_str(tracestate_value)?,
                    );
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).headers(headers).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    tracing::info!(body = body, status = status.to_string());
                }

                Ok::<(), anyhow::Error>(())
            })(),
        )
        .await;
        Ok(())
    }

    #[tokio::test]
    async fn test_health_endpoint_with_changing_health_status() {
        // Test health endpoint starts in not ready status, then becomes ready
        // when endpoints are created (DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=generate)
        const ENDPOINT_NAME: &str = "generate";
        const ENDPOINT_HEALTH_CONFIG: &str = "[\"generate\"]";
        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS, Some("notready")),
                (env_system::DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS, Some(ENDPOINT_HEALTH_CONFIG)),
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Check if system status server was started
                let system_info_opt = drt.system_status_server_info();

                // Ensure system status server was spawned by DRT
                assert!(
                    system_info_opt.is_some(),
                    "System status server was not spawned by DRT. Expected DRT to spawn server when DYN_SYSTEM_PORT is set to a positive value, but system_status_server_info() returned None. Environment: DYN_SYSTEM_PORT={:?}",
                    std::env::var(env_system::DYN_SYSTEM_PORT)
                );

                // Get the system status server info from DRT - this should never fail now due to above check
                let system_info = system_info_opt.unwrap();
                let addr = system_info.socket_addr;

                // Initially check health - should be not ready
                let client = reqwest::Client::new();
                let health_url = format!("http://{}/health", addr);

                let response = client.get(&health_url).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();

                // Health should be not ready (503) initially
                assert_eq!(status, 503, "Health should be 503 (not ready) initially, got: {}", status);
                assert!(body.contains("\"status\":\"notready\""), "Health should contain status notready");

                // Now create a namespace, component, and endpoint to make the system healthy
                let namespace = drt.namespace("ns1234").unwrap();
                let component = namespace.component("comp1234").unwrap();

                // Create a simple test handler
                use crate::pipeline::{async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, SingleIn};
                use crate::protocols::annotated::Annotated;

                struct TestHandler;

                #[async_trait]
                impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error> for TestHandler {
                    async fn generate(&self, input: SingleIn<String>) -> anyhow::Result<ManyOut<Annotated<String>>> {
                        let (data, ctx) = input.into_parts();
                        let response = Annotated::from_data(format!("You responded: {}", data));
                        Ok(crate::pipeline::ResponseStream::new(
                            Box::pin(crate::stream::iter(vec![response])),
                            ctx.context()
                        ))
                    }
                }

                // Create the ingress and start the endpoint service
                let ingress = Ingress::for_engine(std::sync::Arc::new(TestHandler)).unwrap();

                // Start the service and endpoint with a health check payload
                // This will automatically register the endpoint for health monitoring
                tokio::spawn(async move {
                    let _ = component.endpoint(ENDPOINT_NAME)
                        .endpoint_builder()
                        .handler(ingress)
                        .health_check_payload(serde_json::json!({
                            "test": "health_check"
                        }))
                        .start()
                        .await;
                });

                // Hit health endpoint 200 times to verify consistency
                let mut success_count = 0;
                let mut failures = Vec::new();

                for i in 1..=200 {
                    let response = client.get(&health_url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();

                    if status == 200 && body.contains("\"status\":\"ready\"") {
                        success_count += 1;
                    } else {
                        failures.push((i, status.as_u16(), body.clone()));
                        if failures.len() <= 5 {  // Only log first 5 failures
                            tracing::warn!("Request {}: status={}, body={}", i, status, body);
                        }
                    }
                }

                tracing::info!("Health endpoint test results: {}/200 requests succeeded", success_count);
                if !failures.is_empty() {
                    tracing::warn!("Failed requests: {}", failures.len());
                }

                // Expect at least 150 out of 200 requests to be successful
                assert!(success_count >= 150, "Expected at least 150 out of 200 requests to succeed, but only {} succeeded", success_count);
            },
        )
        .await;
    }

    #[tokio::test]
    async fn test_spawn_system_status_server_endpoints() {
        // use reqwest for HTTP requests
        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS, Some("ready")),
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;
                let client = reqwest::Client::new();
                for (path, expect_200, expect_body) in [
                    ("/health", true, "ready"),
                    ("/live", true, "ready"),
                    ("/someRandomPathNotFoundHere", false, "Route not found"),
                ] {
                    println!("[test] Sending request to {}", path);
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    println!(
                        "[test] Response for {}: status={}, body={:?}",
                        path, status, body
                    );
                    if expect_200 {
                        assert_eq!(status, 200, "Response: status={}, body={:?}", status, body);
                    } else {
                        assert_eq!(status, 404, "Response: status={}, body={:?}", status, body);
                    }
                    assert!(
                        body.contains(expect_body),
                        "Response: status={}, body={:?}",
                        status,
                        body
                    );
                }
                // DRT handles server cleanup automatically
            },
        )
        .await;
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn test_health_check_with_payload_and_timeout() {
        // Test the complete health check flow with the new canary-based system:
        crate::logging::init();

        temp_env::async_with_vars(
            [
                (env_system::DYN_SYSTEM_PORT, Some("0")),
                (
                    env_system::DYN_SYSTEM_STARTING_HEALTH_STATUS,
                    Some("notready"),
                ),
                (
                    env_system::DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS,
                    Some("[\"test.endpoint\"]"),
                ),
                // Enable health check with short intervals for testing
                ("DYN_HEALTH_CHECK_ENABLED", Some("true")),
                (env_canary::DYN_CANARY_WAIT_TIME, Some("1")), // Send canary after 1 second of inactivity
                ("DYN_HEALTH_CHECK_REQUEST_TIMEOUT", Some("1")), // Immediately timeout to mimic unresponsiveness
                ("RUST_LOG", Some("info")),                      // Enable logging for test
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started");
                let addr = system_info.socket_addr;

                let client = reqwest::Client::new();
                let health_url = format!("http://{}/health", addr);

                // Register an endpoint with health check payload
                let endpoint = "test.endpoint";
                let health_check_payload = serde_json::json!({
                    "prompt": "health check test",
                    "_health_check": true
                });

                // Register the endpoint and its health check payload
                {
                    let system_health = drt.system_health();
                    let system_health_lock = system_health.lock();
                    system_health_lock.register_health_check_target(
                        endpoint,
                        crate::component::Instance {
                            component: "test_component".to_string(),
                            endpoint: "health".to_string(),
                            namespace: "test_namespace".to_string(),
                            instance_id: 1,
                            transport: crate::component::TransportType::Nats(endpoint.to_string()),
                        },
                        health_check_payload.clone(),
                    );
                }

                // Check initial health - should be ready (default state)
                let response = client.get(&health_url).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();
                assert_eq!(status, 503, "Should be unhealthy initially (default state)");
                assert!(
                    body.contains("\"status\":\"notready\""),
                    "Should show notready status initially"
                );

                // Set endpoint to healthy state
                drt.system_health()
                    .lock()
                    .set_endpoint_health_status(endpoint, HealthStatus::Ready);

                // Check health again - should now be healthy
                let response = client.get(&health_url).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();

                assert_eq!(status, 200, "Should be healthy due to recent response");
                assert!(
                    body.contains("\"status\":\"ready\""),
                    "Should show ready status after response"
                );

                // Verify the endpoint status in SystemHealth directly
                let endpoint_status = drt
                    .system_health()
                    .lock()
                    .get_endpoint_health_status(endpoint);
                assert_eq!(
                    endpoint_status,
                    Some(HealthStatus::Ready),
                    "SystemHealth should show endpoint as Ready after response"
                );
            },
        )
        .await;
    }
}
