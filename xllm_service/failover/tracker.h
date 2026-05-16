/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>

#include "request/request.h"

namespace xllm_service {

struct FailoverRecoveryConfig {
  int64_t restore_block_cost_ms = 1;
  int64_t decode_recompute_token_cost_ms = 1;
  int64_t prefill_recompute_token_cost_ms = 1;
};

struct FailoverFirstTokenSample {
  int32_t attempt = 0;
  int64_t latency_ms = 0;
  std::string from_prefill;
  std::string from_decode;
  std::string to_prefill;
  std::string to_decode;
};

inline const char* FailoverTypeName(FailoverType failover_type) {
  switch (failover_type) {
    case FailoverType::PREFILL_CRASH:
      return "PREFILL_CRASH";
    case FailoverType::DECODE_CRASH:
      return "DECODE_CRASH";
    case FailoverType::NONE:
      return "NONE";
  }
  return "UNKNOWN";
}

inline std::optional<FailoverType> MatchFailedInstanceForFailover(
    const Request& request,
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  const auto incarnation_matches = [](const std::string& expected,
                                      const std::string& actual) {
    return expected.empty() || expected == actual;
  };

  const bool can_match_prefill =
      type == InstanceType::DEFAULT || type == InstanceType::PREFILL;
  if (can_match_prefill && request.routing.prefill_name == instance_name &&
      incarnation_matches(incarnation_id, request.prefill_incarnation_id) &&
      !request.prefill_stage_finished && request.num_generated_tokens == 0) {
    return FailoverType::PREFILL_CRASH;
  }

  if (type == InstanceType::DECODE &&
      request.routing.decode_name == instance_name &&
      incarnation_matches(incarnation_id, request.decode_incarnation_id)) {
    return FailoverType::DECODE_CRASH;
  }

  return std::nullopt;
}

inline int64_t DivideRoundUp(int64_t value, int64_t divisor) {
  if (value <= 0 || divisor <= 0) {
    return 0;
  }
  return (value + divisor - 1) / divisor;
}

inline void ApplyFailoverSlo(Request* request) {
  request->failover.slo.effective_ttft_ms = request->failover.slo.original_ttft_ms;
  request->failover.slo.effective_tpot_ms = request->failover.slo.original_tpot_ms;
  request->failover.slo.effective_ttlt_ms = request->failover.slo.original_ttlt_ms;

  switch (request->failover.runtime.type) {
    case FailoverType::PREFILL_CRASH:
      request->failover.slo.effective_ttft_ms =
          request->failover.slo.prefill_recovery_ttft_ms != kUnsetSloMs
              ? request->failover.slo.prefill_recovery_ttft_ms
              : request->failover.slo.original_ttft_ms;
      break;
    case FailoverType::DECODE_CRASH:
      request->failover.slo.effective_ttft_ms =
          request->failover.slo.decode_recovery_ttft_ms != kUnsetSloMs
              ? request->failover.slo.decode_recovery_ttft_ms
              : request->failover.slo.original_ttft_ms;
      break;
    case FailoverType::NONE:
      break;
  }
}

inline void MarkRequestForFailover(Request* request,
                                   FailoverType failover_type,
                                   absl::Time detected_time,
                                   int32_t block_size,
                                   const FailoverRecoveryConfig& config) {
  request->failover.runtime.type = failover_type;
  request->failover.runtime.detected_time = detected_time;
  request->failover.runtime.redispatched_time = absl::InfinitePast();
  request->failover.runtime.rpc_redispatched_time = absl::InfinitePast();
  request->failover.runtime.estimated_restore_cost_ms = 0;
  request->failover.runtime.estimated_recompute_cost_ms = 0;
  request->failover.runtime.estimated_total_recovery_cost_ms = 0;
  request->failover.runtime.estimated_restore_blocks = 0;
  request->failover.runtime.estimated_recompute_tokens = 0;
  request->failover.runtime.generated_tokens =
      std::max<int64_t>(0, request->num_generated_tokens);
  request->failover.runtime.previous_token_time =
      request->failover.runtime.generated_tokens > 0 ? request->latest_generate_time
                                             : absl::InfinitePast();
  request->failover.runtime.failed_decode_offload_batch_size = request->offload_batch_size;

  switch (failover_type) {
    case FailoverType::PREFILL_CRASH:
      request->failover.runtime.estimated_recompute_tokens =
          static_cast<int64_t>(request->token_ids.size());
      request->failover.runtime.estimated_recompute_cost_ms =
          request->failover.runtime.estimated_recompute_tokens *
          config.prefill_recompute_token_cost_ms;
      break;
    case FailoverType::DECODE_CRASH: {
      const int64_t generated_tokens =
          std::max<int64_t>(0, request->num_generated_tokens);
      int64_t offloaded_tokens = 0;
      if (request->offload_batch_size != UINT32_MAX &&
          request->offload_batch_size > 0 && block_size > 0) {
        const int64_t offload_window_tokens =
            static_cast<int64_t>(request->offload_batch_size) * block_size;
        offloaded_tokens =
            generated_tokens / offload_window_tokens * offload_window_tokens;
      }
      const int64_t tail_tokens = generated_tokens - offloaded_tokens;
      const int64_t restore_blocks = DivideRoundUp(offloaded_tokens, block_size);
      request->failover.runtime.estimated_restore_blocks = restore_blocks;
      request->failover.runtime.estimated_recompute_tokens = tail_tokens;
      request->failover.runtime.estimated_restore_cost_ms =
          restore_blocks * config.restore_block_cost_ms;
      request->failover.runtime.estimated_recompute_cost_ms =
          tail_tokens * config.decode_recompute_token_cost_ms;
      break;
    }
    case FailoverType::NONE:
      break;
  }

  request->failover.runtime.estimated_total_recovery_cost_ms =
      request->failover.runtime.estimated_restore_cost_ms +
      request->failover.runtime.estimated_recompute_cost_ms;
  ApplyFailoverSlo(request);
}

inline void ArmFailoverAttempt(Request* request, absl::Time now) {
  request->failover.runtime.last_from_prefill = request->routing.prefill_name;
  request->failover.runtime.last_from_decode = request->routing.decode_name;
  request->failover.runtime.attempt += 1;
  request->failover.runtime.awaiting_first_token = true;
  request->failover.runtime.latest_start_time = now;
  request->offload_batch_size = UINT32_MAX;
  request->decode_offload_batch_size_recorded = false;
}

inline std::optional<FailoverFirstTokenSample> ConsumeFailoverFirstTokenSample(
    Request* request,
    absl::Time now) {
  if (!request->failover.runtime.awaiting_first_token) {
    return std::nullopt;
  }

  request->failover.runtime.awaiting_first_token = false;
  return FailoverFirstTokenSample{
      .attempt = request->failover.runtime.attempt,
      .latency_ms =
          absl::ToInt64Milliseconds(now - request->failover.runtime.latest_start_time),
      .from_prefill = request->failover.runtime.last_from_prefill,
      .from_decode = request->failover.runtime.last_from_decode,
      .to_prefill = request->routing.prefill_name,
      .to_decode = request->routing.decode_name,
  };
}

}  // namespace xllm_service
