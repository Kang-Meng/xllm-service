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

#include <absl/time/time.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace xllm_service {

inline constexpr int32_t kUnsetSloMs = std::numeric_limits<int32_t>::max();

enum class FailoverType : int32_t {
  NONE = 0,
  PREFILL_CRASH = 1,
  DECODE_CRASH = 2,
};

struct FailoverReplayState {
  size_t base_prompt_token_count = 0;
  std::vector<int32_t> committed_output_token_ids;
  std::string committed_output_text;
  std::string accumulated_output_text;
};

struct FailoverRuntimeState {
  int32_t attempt = 0;
  bool awaiting_first_token = false;
  absl::Time latest_start_time;
  std::string last_from_prefill;
  std::string last_from_decode;
  absl::Time detected_time = absl::InfinitePast();
  absl::Time redispatched_time = absl::InfinitePast();
  absl::Time rpc_redispatched_time = absl::InfinitePast();
  absl::Time previous_token_time = absl::InfinitePast();
  FailoverType type = FailoverType::NONE;
  int64_t estimated_restore_cost_ms = 0;
  int64_t estimated_recompute_cost_ms = 0;
  int64_t estimated_total_recovery_cost_ms = 0;
  int64_t estimated_restore_blocks = 0;
  int64_t estimated_recompute_tokens = 0;
  int64_t generated_tokens = 0;
  uint32_t failed_decode_offload_batch_size = UINT32_MAX;
};

struct FailoverSloState {
  int32_t original_ttft_ms = kUnsetSloMs;
  int32_t original_tpot_ms = kUnsetSloMs;
  int32_t original_ttlt_ms = kUnsetSloMs;
  int32_t prefill_recovery_ttft_ms = kUnsetSloMs;
  int32_t decode_recovery_ttft_ms = kUnsetSloMs;
  int32_t effective_ttft_ms = kUnsetSloMs;
  int32_t effective_tpot_ms = kUnsetSloMs;
  int32_t effective_ttlt_ms = kUnsetSloMs;
};

struct FailoverState {
  FailoverReplayState replay;
  FailoverRuntimeState runtime;
  FailoverSloState slo;
};

}  // namespace xllm_service
