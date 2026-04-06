# Multi-Worker End-to-End Health Check Design Document

## Problem Statement

The current e2e health check implementation sends a single request through the pipeline, which only verifies that *one* worker is functional. In disaggregated serving with multiple prefill workers, this is insufficient because:

1. A single healthy worker may mask unhealthy workers
2. Load balancing issues won't be detected until traffic hits unhealthy workers
3. Kubernetes probes need to know if the system has sufficient healthy workers

## Current Architecture Understanding

### Worker Discovery

Workers register via the discovery service (etcd-backed) with:
- **Instance ID**: Unique 64-bit identifier per worker instance
- **Endpoint ID**: `{namespace}/{component}/{endpoint}` tuple
- **Model Deployment Card**: Metadata about the model/worker capabilities

```rust
// From lib/runtime/src/component.rs
pub struct Instance {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub instance_id: u64,
    pub transport: TransportType,
}
```

### Routing Capabilities

The `PushRouter` supports multiple routing modes:

```rust
// lib/runtime/src/pipeline/network/egress/push_router.rs
pub async fn round_robin(&self, request: SingleIn<T>) -> Result<ManyOut<U>>
pub async fn random(&self, request: SingleIn<T>) -> Result<ManyOut<U>>
pub async fn direct(&self, request: SingleIn<T>, instance_id: u64) -> Result<ManyOut<U>>
```

The `direct()` method allows routing to a specific worker by instance_id.

### Existing Health Infrastructure

- `SystemHealth` tracks per-endpoint health status
- `KvWorkerMonitor` tracks worker load states (active/total KV blocks)
- `Client` maintains `instance_avail` and `instance_free` lists

---

## Proposed Approaches

### Option A: Direct Instance Routing via Prefill Router

**Description**: Use the discovery service to list all prefill worker instances, then use the `direct()` routing method to send a health check request to each specific worker.

**Implementation**:

```rust
async fn e2e_health_handler_multi_worker(...) -> impl IntoResponse {
    // 1. Get the prefill router's client
    let client = prefill_router.client();

    // 2. List all available instances
    let instances = client.instances();

    // 3. Send health check to each instance using direct routing
    let mut results = HashMap::new();
    for instance in instances {
        let result = tokio::time::timeout(
            check_timeout,
            prefill_router.direct(health_request.clone(), instance.instance_id)
        ).await;

        results.insert(instance.instance_id, match result {
            Ok(Ok(_)) => WorkerHealthStatus::Healthy,
            Ok(Err(e)) => WorkerHealthStatus::Error(e.to_string()),
            Err(_) => WorkerHealthStatus::Timeout,
        });
    }

    // 4. Aggregate results
    Json(MultiWorkerHealthResponse { workers: results, ... })
}
```

**Pros**:
- Guarantees each worker is individually tested
- Reuses existing routing infrastructure
- Clear per-worker health status
- Works with existing request/response flow

**Cons**:
- Requires access to the prefill router from the health handler
- Sequential checks may be slow (can parallelize with `join_all`)
- More complex state sharing between HTTP service and routing layer
- Health check load scales with worker count

**Complexity**: Medium-High

---

### Option B: Discovery-Based Instance Enumeration with Parallel Checks

**Description**: Query the discovery service directly for all prefill model instances, then send parallel health check requests through the normal pipeline, tracking which workers respond.

**Implementation**:

```rust
async fn e2e_health_handler_multi_worker(...) -> impl IntoResponse {
    // 1. Query discovery for all prefill instances
    let discovery = state.discovery();
    let instances = discovery.list(DiscoveryQuery::AllModels).await?;
    let prefill_instances: Vec<_> = instances
        .iter()
        .filter(|i| i.is_prefill_model())
        .collect();

    // 2. Send N requests in parallel (N = worker count)
    let num_workers = prefill_instances.len();
    let requests = (0..num_workers).map(|_| {
        let request = create_health_request();
        tokio::spawn(async move {
            engine.generate(request).await
        })
    });

    // 3. Collect results and track responding workers
    let results = futures::future::join_all(requests).await;

    // 4. Report status
    Json(HealthResponse {
        requests_sent: num_workers,
        successful: results.iter().filter(|r| r.is_ok()).count(),
        ...
    })
}
```

**Pros**:
- Simpler implementation (doesn't need router access)
- Parallel execution is natural
- Uses existing discovery infrastructure
- Less coupling between components

**Cons**:
- Cannot guarantee all workers are tested (load balancer may route multiple requests to same worker)
- May need to send more requests than workers to ensure coverage
- No definitive per-worker status

**Complexity**: Low-Medium

---

### Option C: Worker-Level Health Endpoints (gRPC/NATS Direct)

**Description**: Each worker exposes its own health endpoint. The frontend health check queries each worker directly via the transport layer (NATS or gRPC).

**Implementation**:

```rust
// Worker side (in Python/Rust worker code)
async fn handle_health_check(request: HealthCheckRequest) -> HealthCheckResponse {
    // Perform local health check (GPU memory, model loaded, etc.)
    HealthCheckResponse {
        status: "healthy",
        gpu_memory_used: get_gpu_memory(),
        requests_in_flight: get_inflight_count(),
    }
}

// Frontend side
async fn e2e_health_handler_multi_worker(...) -> impl IntoResponse {
    // 1. Get all worker instances from discovery
    let instances = discovery.list(DiscoveryQuery::ComponentEndpoints {
        namespace: "dynamo",
        component: "prefill",
    }).await?;

    // 2. Send direct health check RPC to each worker
    let checks = instances.iter().map(|instance| {
        let client = create_direct_client(instance);
        tokio::spawn(async move {
            client.health_check().await
        })
    });

    // 3. Aggregate results
    let results = futures::future::join_all(checks).await;
    // ...
}
```

**Pros**:
- Most accurate per-worker health status
- Can include worker-specific metrics (GPU memory, queue depth)
- No routing layer involvement
- Workers can report detailed local status

**Cons**:
- Requires new health check endpoint on each worker
- More infrastructure changes (both Rust and Python workers)
- Direct communication bypasses normal request flow
- May not catch pipeline integration issues

**Complexity**: High

---

### Option D: Hybrid - Request Tracking with Instance ID Headers

**Description**: Send requests through the normal pipeline but include/return the instance_id that handled the request. Track which workers respond over multiple requests.

**Implementation**:

```rust
// Modify LLMEngineOutput to include worker_instance_id
pub struct LLMEngineOutput {
    pub token_ids: Vec<u32>,
    pub finish_reason: Option<FinishReason>,
    pub worker_instance_id: Option<u64>,  // NEW: Which worker handled this
}

// Health check handler
async fn e2e_health_handler_multi_worker(...) -> impl IntoResponse {
    // 1. Get expected worker count
    let expected_workers = discovery.list_prefill_count().await?;

    // 2. Send requests until all workers are seen (with limit)
    let mut seen_workers = HashSet::new();
    let max_requests = expected_workers * 3;  // Safety limit

    for _ in 0..max_requests {
        if seen_workers.len() >= expected_workers {
            break;
        }

        let response = engine.generate(health_request.clone()).await?;
        if let Some(worker_id) = response.worker_instance_id {
            seen_workers.insert(worker_id);
        }
    }

    // 3. Report coverage
    Json(HealthResponse {
        workers_seen: seen_workers.len(),
        workers_expected: expected_workers,
        all_healthy: seen_workers.len() >= expected_workers,
    })
}
```

**Pros**:
- Uses normal request flow (catches integration issues)
- Minimal infrastructure changes
- Probabilistically covers all workers
- Can be combined with existing health check

**Cons**:
- Requires protocol change to return worker_instance_id
- Non-deterministic (may need many requests for full coverage)
- Slower for large worker counts
- Load balancing may cause uneven coverage

**Complexity**: Medium

---

## Comparison Matrix

| Criteria | Option A | Option B | Option C | Option D |
|----------|----------|----------|----------|----------|
| **Per-worker guarantee** | Yes | No | Yes | Probabilistic |
| **Implementation complexity** | Medium-High | Low-Medium | High | Medium |
| **Infrastructure changes** | Medium | Low | High | Medium |
| **Tests full pipeline** | Yes | Yes | No | Yes |
| **Parallel execution** | Yes | Yes | Yes | Sequential |
| **Detailed worker metrics** | No | No | Yes | No |
| **Latency (10 workers)** | ~1-2s | ~1s | ~0.5s | Variable |

---

## Recommendation

**Primary Recommendation: Option A (Direct Instance Routing)**

This option provides the best balance of:
1. **Guaranteed coverage**: Each worker is definitely tested
2. **Full pipeline validation**: Requests go through the actual generation flow
3. **Reuse of existing infrastructure**: Uses the `direct()` routing already available
4. **Clear reporting**: Unambiguous per-worker status

**Implementation Priority**:
1. First, implement Option A for accurate multi-worker health checks
2. Later, consider Option C additions if detailed worker metrics are needed

**Fallback: Option B** if Option A proves too complex to wire up the router access.

---

## API Design

### Request
```
GET /health/e2e?check_all_workers=true
```

Or a dedicated endpoint:
```
GET /health/e2e/workers
```

### Response (Success - All Healthy)
```json
{
  "status": "healthy",
  "check_type": "multi_worker",
  "workers": {
    "12345678": {
      "status": "healthy",
      "latency_ms": 45,
      "endpoint": "dyn://dynamo.prefill.generate"
    },
    "87654321": {
      "status": "healthy",
      "latency_ms": 52,
      "endpoint": "dyn://dynamo.prefill.generate"
    }
  },
  "summary": {
    "total_workers": 2,
    "healthy_workers": 2,
    "unhealthy_workers": 0
  },
  "cached": false
}
```

### Response (Partial Failure)
```json
{
  "status": "degraded",
  "check_type": "multi_worker",
  "workers": {
    "12345678": {
      "status": "healthy",
      "latency_ms": 45
    },
    "87654321": {
      "status": "unhealthy",
      "error": "timeout"
    }
  },
  "summary": {
    "total_workers": 2,
    "healthy_workers": 1,
    "unhealthy_workers": 1
  }
}
```

HTTP Status Codes:
- `200 OK`: All workers healthy
- `206 Partial Content`: Some workers unhealthy (degraded)
- `500 Internal Server Error`: Majority of workers unhealthy
- `503 Service Unavailable`: No workers available

---

## Next Steps

1. Decide on approach (recommend Option A)
2. Determine if this should be:
   - A new endpoint (`/health/e2e/workers`)
   - A query parameter on existing endpoint (`/health/e2e?all_workers=true`)
   - Replace the existing single-worker check
3. Implement the chosen approach
4. Add configuration for:
   - Minimum healthy worker threshold
   - Per-worker timeout
   - Parallel vs sequential checks
