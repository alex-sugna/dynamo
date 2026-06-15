// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use axum::{Json, Router, http::Method, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_runtime::instances::list_all_instances;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

/// Environment variables for frontend health check
const FRONTEND_URL_ENV: &str = "DYN_FRONTEND_URL";
const HEALTH_CHECK_MODEL_ENV: &str = "DYN_HEALTH_CHECK_MODEL";
const PREFILL_COMPONENT_NAME_ENV: &str = "DYN_PREFILL_COMPONENT_NAME";
const E2E_HEALTH_CHECK_TIMEOUT_ENV: &str = "DYN_E2E_HEALTH_CHECK_TIMEOUT";
const E2E_LAST_HEALTHY_TIMEOUT_ENV: &str = "DYN_E2E_LAST_HEALTHY_TIMEOUT";

const DEFAULT_PREFILL_COMPONENT_NAME: &str = "prefill";
const DEFAULT_E2E_HEALTH_CHECK_TIMEOUT_SECS: u64 = 30;
const DEFAULT_E2E_LAST_HEALTHY_TIMEOUT_SECS: u64 = 10;

/// Validate an HTTP-200 e2e probe response body. Mirror of the helper in
/// `system_status_server.rs`; see that file for full rationale.
/// Gate is `usage.completion_tokens > 0` — shape-agnostic so it handles
/// content / reasoning_content (thinking models) / tool_calls uniformly.
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

fn get_e2e_health_check_timeout() -> Duration {
    std::env::var(E2E_HEALTH_CHECK_TIMEOUT_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_E2E_HEALTH_CHECK_TIMEOUT_SECS))
}

fn get_e2e_last_healthy_timeout() -> Duration {
    std::env::var(E2E_LAST_HEALTHY_TIMEOUT_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_E2E_LAST_HEALTHY_TIMEOUT_SECS))
}

/// State for frontend e2e health checks
pub struct FrontendHealthState {
    last_healthy: RwLock<Option<Instant>>,
    health_lock: Mutex<()>,
    http_client: reqwest::Client,
}

impl FrontendHealthState {
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

impl Default for FrontendHealthState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let health_path = path.unwrap_or_else(|| "/health".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &health_path)];

    let router = Router::new()
        .route(&health_path, get(health_handler))
        .with_state(state);

    (docs, router)
}

pub fn live_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let live_path = path.unwrap_or_else(|| "/live".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &live_path)];

    let router = Router::new()
        .route(&live_path, get(live_handler))
        .with_state(state);

    (docs, router)
}

async fn live_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    // Check if the http service is being cancelled/shutdown
    if state.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "shutting_down",
                "message": "Service is shutting down"
            })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "live",
            "message": "Service is live"
        })),
    )
}

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let instances = match list_all_instances(state.discovery()).await {
        Ok(instances) => instances,
        Err(err) => {
            tracing::warn!(%err, "Failed to fetch instances from discovery");
            vec![]
        }
    };
    let mut endpoints: Vec<String> = instances
        .iter()
        .map(|instance| instance.endpoint_id().as_url())
        .collect();
    endpoints.sort();
    endpoints.dedup();
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "endpoints": endpoints,
            "instances": instances
        })),
    )
}

/// Router for /health/frontend endpoint (e2e health check)
pub fn frontend_health_check_router(
    state: Arc<service_v2::State>,
    health_state: Arc<FrontendHealthState>,
) -> (Vec<RouteDoc>, Router) {
    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, "/health/frontend")];

    let router = Router::new()
        .route("/health/frontend", get(frontend_health_handler))
        .layer(axum::Extension(health_state))
        .with_state(state);

    (docs, router)
}

/// Frontend e2e health check handler
/// Loops through prefill workers until one succeeds, letting decode be selected naturally
async fn frontend_health_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    axum::Extension(health_state): axum::Extension<Arc<FrontendHealthState>>,
) -> impl IntoResponse {
    let last_healthy_timeout = get_e2e_last_healthy_timeout();

    tracing::info!("[frontend health] Starting health check");

    // Acquire lock to prevent concurrent health checks
    let _guard = health_state.health_lock.lock().await;

    // Check if recent health check is still valid (cached)
    {
        let last_healthy = health_state.last_healthy.read().await;
        if let Some(last) = *last_healthy {
            let age = last.elapsed();
            if age < last_healthy_timeout {
                tracing::info!("[frontend health] Returning cached healthy response (age: {:?})", age);
                return (
                    StatusCode::OK,
                    Json(json!({
                        "status": "healthy",
                        "cached": true,
                        "last_check_age_secs": age.as_secs_f64()
                    })),
                );
            }
        }
    }

    // Check if a recent user request completed successfully (active traffic = healthy)
    if let Some(age) = state.last_successful_request_age() {
        if age < last_healthy_timeout {
            tracing::info!(
                "[frontend health] Active traffic detected: last successful request was {:?} ago, returning healthy",
                age
            );
            return (
                StatusCode::OK,
                Json(json!({
                    "status": "healthy",
                    "cached": true,
                    "source": "active_traffic",
                    "last_request_age_secs": age.as_secs_f64()
                })),
            );
        }
    }

    // Get frontend URL from environment
    let frontend_url = match std::env::var(FRONTEND_URL_ENV) {
        Ok(url) => url,
        Err(_) => {
            tracing::warn!("[frontend health] Skipped: {} not set", FRONTEND_URL_ENV);
            return (
                StatusCode::OK,
                Json(json!({
                    "status": "healthy",
                    "e2e_check": "skipped",
                    "reason": format!("{} not configured", FRONTEND_URL_ENV)
                })),
            );
        }
    };

    // Get model name from environment
    let model = match std::env::var(HEALTH_CHECK_MODEL_ENV) {
        Ok(m) => m,
        Err(_) => {
            tracing::warn!("[frontend health] Skipped: {} not set", HEALTH_CHECK_MODEL_ENV);
            return (
                StatusCode::OK,
                Json(json!({
                    "status": "healthy",
                    "e2e_check": "skipped",
                    "reason": format!("{} not configured", HEALTH_CHECK_MODEL_ENV)
                })),
            );
        }
    };

    // Get prefill component name
    let prefill_component_name = std::env::var(PREFILL_COMPONENT_NAME_ENV)
        .unwrap_or_else(|_| DEFAULT_PREFILL_COMPONENT_NAME.to_string());

    // Discover prefill workers
    let target_namespace = std::env::var("DYN_NAMESPACE").ok();

    let prefill_instance_ids: Vec<u64> = match list_all_instances(state.discovery()).await {
        Ok(instances) => {
            instances
                .iter()
                .filter(|i| target_namespace.as_ref().map_or(false, |ns| &i.namespace == ns))
                .filter(|i| i.component == prefill_component_name)
                .map(|i| i.instance_id)
                .collect()
        }
        Err(e) => {
            tracing::warn!("[frontend health] Failed to query discovery: {}", e);
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "unhealthy",
                    "error": format!("Failed to query discovery: {}", e)
                })),
            );
        }
    };

    if prefill_instance_ids.is_empty() {
        tracing::warn!("[frontend health] No prefill workers found");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "error": "No prefill workers found"
            })),
        );
    }

    tracing::info!(
        "[frontend health] Found {} prefill workers: {:?}",
        prefill_instance_ids.len(),
        prefill_instance_ids
    );

    let chat_url = format!("{}/v1/chat/completions", frontend_url.trim_end_matches('/'));
    let start_time = Instant::now();
    let mut last_error = String::new();
    let mut tried_count = 0;

    // Loop through prefill workers until one succeeds
    for prefill_id in &prefill_instance_ids {
        tried_count += 1;

        let health_request = json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 2,
            "stream": false,
            "temperature": 0.0,
            "nvext": {
                "backend_instance_id": prefill_id
            }
        });

        tracing::info!(
            "[frontend health] Attempt {}/{}: Trying prefill worker {}",
            tried_count,
            prefill_instance_ids.len(),
            prefill_id
        );

        let result = health_state
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
                        *health_state.last_healthy.write().await = Some(Instant::now());

                        tracing::info!(
                            "[frontend health] PASSED via prefill worker {} in {:?} (attempts={}/{})",
                            prefill_id,
                            elapsed,
                            tried_count,
                            prefill_instance_ids.len()
                        );

                        return (
                            StatusCode::OK,
                            Json(json!({
                                "status": "healthy",
                                "e2e_check": "passed",
                                "prefill_worker_used": prefill_id,
                                "prefill_workers_tried": tried_count,
                                "prefill_workers_total": prefill_instance_ids.len(),
                                "latency_ms": elapsed.as_millis()
                            })),
                        );
                    }
                    Err(body_err) => {
                        last_error = format!("HTTP 200 but body invalid: {body_err}");
                        tracing::warn!(
                            "[frontend health] Attempt {}/{}: Prefill worker {} returned HTTP 200 with bad body: {}",
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
                    "[frontend health] Attempt {}/{}: Prefill worker {} FAILED: {}",
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
                    "[frontend health] Attempt {}/{}: Prefill worker {} FAILED: {}",
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
        "[frontend health] FAILED: All {} prefill workers failed. Last error: {}",
        prefill_instance_ids.len(),
        last_error
    );

    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "status": "unhealthy",
            "e2e_check": "failed",
            "error": format!("All {} prefill workers failed", prefill_instance_ids.len()),
            "last_error": last_error,
            "prefill_workers_tried": tried_count,
            "prefill_instance_ids": prefill_instance_ids,
            "latency_ms": elapsed.as_millis()
        })),
    )
}
