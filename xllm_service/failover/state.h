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
  // Set when the most recent response carried finished_on_prefill_instance=true.
  // The xllm decode side enables checking_prefill_token_, which advances the
  // incremental decoder's output_offset past the prefill-sampled token T0 for
  // text purposes but still slices token_ids from the pre-advance position
  // (sequence.cpp:475-516 / incremental_decoder.cpp:57-65). The net effect is
  // that decode's first streaming response repeats T0 (and any other tokens
  // emitted by prefill before handoff). We dedup that single occurrence by
  // matching the longest suffix of committed_output_token_ids against the
  // prefix of decode's first delta and skipping it.
  bool pending_decode_handoff_dedup = false;
  // True once we've already consumed (or armed and consumed) the
  // prefill->decode handoff dedup for this attempt. xllm may emit more than
  // one response with finished_on_prefill_instance=true (e.g. a streaming
  // batch path that sets the flag on both the per-output and the
  // sequence-level path). Without this latch, a later such response would
  // re-arm pending_decode_handoff_dedup, and if the head of the next decode
  // delta coincidentally matched the tail of committed_output_token_ids
  // (common: repeated words, whitespace, punctuation), HandoffOverlapPrefixLength
  // would silently drop a real token and shift every downstream KV-cache
  // block-hash boundary by one position - which is exactly the R-01
  // "block hash all wrong" failure mode. Reset by ClearCommittedReplayState
  // on rehandle so the next attempt's first prefill->decode boundary still
  // dedups correctly.
  bool decode_handoff_dedup_consumed = false;
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
  std::string planned_prefill_name;
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
