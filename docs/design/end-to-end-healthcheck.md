# End-to-End Health Check Design Document

## Problem Statement

The current health check in dynamo disaggregated serving only verifies that endpoints are registered in the discovery service. It returns instance information but does not perform an actual generation request to verify the entire pipeline is functional.

**Current behavior:**
```bash
curl -X GET http://localhost:19971/health
```
Returns:
```json
{
  "status": "healthy",
  "endpoints": ["dyn://dynamo.prefill.generate", "dyn://dynamo.tensorrt_llm.generate"],
  "instances": [...]
}
```

This is insufficient because:
1. Workers may be registered but stalled/deadlocked
2. GPU memory issues may prevent actual generation
3. Pipeline connectivity issues won't be detected
4. Model loading failures after registration won't be caught

## Proposed Solution

Implement an end-to-end health check that performs a minimal generation request through the entire pipeline.

### Key Components

#### 1. Configuration (Environment Variables)

```python
LAST_HEALTHY_TIMEOUT = float(os.getenv("LAST_HEALTHY_TIMEOUT", "10.0"))  # seconds
CHECK_HEALTHY_TIMEOUT = float(os.getenv("CHECK_HEALTHY_TIMEOUT", "10.0"))  # seconds
```

- `LAST_HEALTHY_TIMEOUT`: How long to cache a healthy status before re-checking
- `CHECK_HEALTHY_TIMEOUT`: Timeout for the actual generation check

#### 2. State Tracking

Track health state at the frontend level:
- `last_healthy_timestamp: Option<DateTime>` - When the last successful generation completed
- `health_lock: AsyncMutex` - Prevent concurrent health checks
- `healthy_event: Event` - Signal when generation completes successfully

#### 3. Health Check Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     GET /health                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Acquire health  │
                    │     lock        │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Check last      │
                    │ healthy time    │
                    └─────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ < TIMEOUT ago   │             │ >= TIMEOUT ago  │
    │ Return 200 OK   │             │ or never        │
    └─────────────────┘             └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Create minimal  │
                                    │ chat request    │
                                    └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Send through    │
                                    │ pipeline        │
                                    │ (with timeout)  │
                                    └─────────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                              ▼                               ▼
                    ┌─────────────────┐             ┌─────────────────┐
                    │ Success         │             │ Timeout/Error   │
                    │ Update timestamp│             │ Return 500      │
                    │ Return 200      │             └─────────────────┘
                    └─────────────────┘
```

### Implementation Options

#### Option A: Rust-side Implementation (Recommended)

Implement in `lib/llm/src/http/service/health.rs`:

**Pros:**
- Consistent with existing HTTP service architecture
- Can share state with the chat completions engine
- Better integration with tracing/logging

**Cons:**
- More complex Rust async handling
- Need to create internal chat completion request

**Implementation outline:**
```rust
struct HealthCheckState {
    last_healthy: RwLock<Option<Instant>>,
    health_lock: Mutex<()>,
}

async fn health_handler_e2e(
    state: Arc<service_v2::State>,
    health_state: Arc<HealthCheckState>,
) -> impl IntoResponse {
    let _guard = health_state.health_lock.lock().await;

    // Check if recent health check is still valid
    if let Some(last) = *health_state.last_healthy.read().await {
        if last.elapsed() < Duration::from_secs_f64(LAST_HEALTHY_TIMEOUT) {
            return (StatusCode::OK, Json(json!({"status": "healthy", "cached": true})));
        }
    }

    // Perform actual generation check
    match timeout(Duration::from_secs_f64(CHECK_HEALTHY_TIMEOUT),
                  health_generate(&state)).await {
        Ok(Ok(_)) => {
            *health_state.last_healthy.write().await = Some(Instant::now());
            (StatusCode::OK, Json(json!({"status": "healthy", "generation_check": "passed"})))
        }
        Ok(Err(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"status": "unhealthy", "error": e.to_string()})))
        }
        Err(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"status": "unhealthy", "error": "timeout"})))
        }
    }
}
```

#### Option B: Python-side Implementation (Frontend)

Add health check logic to `components/src/dynamo/frontend/main.py`:

**Pros:**
- Easier to implement quickly
- Can reuse existing Python HTTP client code

**Cons:**
- Requires separate HTTP endpoint or middleware
- May not integrate as cleanly with existing Rust HTTP service

#### Option C: Hybrid Approach

- Keep existing Rust `/health` endpoint for basic checks
- Add new `/health/deep` or `/health/e2e` endpoint for generation check
- Can be implemented in either Rust or Python

### Minimal Health Check Request

```python
health_request = {
    "model": "<discovered_model>",
    "messages": [{"role": "user", "content": "hi"}],
    "max_completion_tokens": 1,
    "stream": False,
    "temperature": 0.0
}
```

Key characteristics:
- Minimal input: "hi" (typically 1-2 tokens)
- Minimal output: 1 token
- Deterministic: temperature=0.0
- Non-streaming: easier to handle response

### Disaggregated Serving Considerations

In disaggregated mode, the health check must verify:

1. **Frontend** can receive and process requests
2. **Prefill Worker** can tokenize and compute prefill
3. **Decode Worker** can generate tokens
4. **Pipeline connectivity** between all components

The generation request naturally tests all of these.

### Error Handling

| Scenario | Response Code | Response Body |
|----------|---------------|---------------|
| Recent healthy check | 200 | `{"status": "healthy", "cached": true}` |
| Generation succeeds | 200 | `{"status": "healthy", "generation_check": "passed"}` |
| Generation timeout | 500 | `{"status": "unhealthy", "error": "Health check timed out"}` |
| Generation error | 500 | `{"status": "unhealthy", "error": "<error_message>"}` |
| No workers available | 503 | `{"status": "unhealthy", "error": "No workers available"}` |

### Kubernetes Integration

The endpoint should be compatible with Kubernetes probes:

```yaml
livenessProbe:
  httpGet:
    path: /live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health  # or /health/e2e
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
  timeoutSeconds: 15
```

### Metrics

Consider adding Prometheus metrics:
- `health_check_total{result="success|failure|timeout"}` - Counter
- `health_check_duration_seconds` - Histogram
- `last_healthy_timestamp` - Gauge

## Recommendation

**Implement Option A (Rust-side)** with a new `/health/e2e` endpoint while keeping the existing `/health` endpoint for backward compatibility.

This provides:
1. Backward compatibility with existing health checks
2. Opt-in deep health checking via `/health/e2e`
3. Proper integration with the existing HTTP service architecture
4. Consistent tracing and logging

## Next Steps

1. Add `HealthCheckState` to `service_v2::State`
2. Implement `health_generate()` function that creates internal chat request
3. Add `/health/e2e` route in `health.rs`
4. Add configuration for timeouts via environment variables
5. Add metrics for health check monitoring
6. Update documentation and Kubernetes deployment examples
