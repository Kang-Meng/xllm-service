# Failover First-Token Metrics Design

## Goal

Add validation instrumentation for request failover so `xllm-service` can measure:

- how long a rehandled request takes to receive its first token after failover
- how many failover rehandle attempts happened
- which request and routing transition produced each sampled latency

The implementation must also include unit tests that mock downstream behavior instead of calling real lower services.

## Scope

In scope:

- add request runtime state for failover timing
- add dedicated failover metrics
- add request-level logs and request trace entries
- add unit tests for the metric/log trigger path with mocked downstream dispatch

Out of scope:

- changing existing load-balance policy semantics
- making failover transparent at protocol level beyond current behavior
- integration tests against real downstream instances

## Design

### Runtime state

Extend `xllm_service::Request` with runtime-only failover fields:

- `int32_t failover_attempt`
- `bool awaiting_failover_first_token`
- `absl::Time latest_failover_start_time`
- `std::string last_failover_from_prefill`
- `std::string last_failover_from_decode`

These fields are not persisted externally. They only exist to track the latest rehandle attempt and its sampling window.

### Sampling start

When `XllmHttpServiceImpl::rehandle_impl()` starts a new rehandle attempt, it will:

1. capture the current prefill/decode routing as the previous route
2. increment `request->failover_attempt`
3. set `request->awaiting_failover_first_token = true`
4. set `request->latest_failover_start_time = absl::Now()`
5. emit a start log and request trace line

The timing starts before `schedule()` so the sampled latency includes the reschedule cost, which is part of recovery latency.

### Sampling end

`Scheduler::update_token_latency_metrics()` already identifies the first token by checking `finished_on_prefill_instance`.

Keep the existing generic TTFT metric unchanged. Additionally, when all of the following are true:

- `finished_on_prefill_instance == true`
- `request->awaiting_failover_first_token == true`

then:

1. compute `failover_ttft_ms = absl::ToInt64Milliseconds(absl::Now() - request->latest_failover_start_time)`
2. observe `failover_ttft_ms` into a dedicated histogram
3. emit a structured info log with request id, attempt, old route, new route, and latency
4. emit a request trace line if tracing is enabled
5. clear `request->awaiting_failover_first_token`

If a request fails again before receiving the first token, the histogram is not sampled for that attempt. A later rehandle attempt may re-arm the flag and produce a sample.

## Metrics

Add:

- `failover_rehandle_total`
- `failover_time_to_first_token_latency_milliseconds`

Semantics:

- `failover_rehandle_total`: increments once per rehandle attempt
- `failover_time_to_first_token_latency_milliseconds`: records one sample per rehandle attempt that successfully reaches the first token

This keeps failover samples separate from the existing `time_to_first_token_latency_milliseconds`.

## Logging and Trace

Add two log points:

1. rehandle start
2. failover first token observed

Both must include:

- `service_request_id`
- `failover_attempt`
- previous prefill/decode route
- current prefill/decode route when available

If `request->trace_callback` exists, emit matching compact trace strings for both events.

## Testing

Add focused unit tests with mocked downstream behavior. Do not talk to real downstream RPC services.

### Tests to add

1. rehandle start arms failover timing state
   Verify `rehandle_impl()` increments attempt state, stores previous route, sets the waiting flag, and increments `failover_rehandle_total`.

2. first token after failover records dedicated histogram and clears the waiting flag
   Create a request with armed failover state, drive `Scheduler::update_token_latency_metrics()` through the first-token path, and verify the flag is cleared and the metric count changes.

3. normal first token does not touch failover metric
   Verify a non-failover request still updates the generic TTFT path only.

4. repeated failover before first token only records on the successful attempt
   Re-arm the failover state twice and verify only the attempt that reaches first token contributes a histogram sample.

### Mocking strategy

- mock `Scheduler::schedule()` dependencies by using a fake or mocked load-balance path instead of real instance discovery
- mock downstream RPC dispatch for `handle/rehandle` tests so the code path stops after request submission setup
- prefer gtest/gmock-based unit tests in the existing test style used by the repo

## Files expected to change

- `xllm_service/request/request.h`
- `xllm_service/common/metrics.h`
- `xllm_service/common/metrics.cpp`
- `xllm_service/http_service/service.cpp`
- `xllm_service/scheduler/scheduler.cpp`
- related unit test files under `xllm_service/...`

## Risks

- current request timing uses mutable request state without a dedicated lock; tests should verify the flag lifecycle remains simple and one-shot
- existing failover flow may re-run tokenization and scheduling; this design only instruments that flow and does not change its semantics
- bvar histogram validation in unit tests should avoid asserting exact percentile values; assert sample presence or counter-like state instead
