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
==============================================================================
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace xllm_service {

inline int64_t NormalizeEstimatedLatencyMs(double predicted_ms,
                                           int32_t fallback_ms) {
  const int64_t normalized_fallback = std::max<int64_t>(1, fallback_ms);
  if (!std::isfinite(predicted_ms) || predicted_ms <= 0.0) {
    return normalized_fallback;
  }
  if (predicted_ms >=
      static_cast<double>(std::numeric_limits<int64_t>::max())) {
    return std::numeric_limits<int64_t>::max();
  }
  return static_cast<int64_t>(std::ceil(predicted_ms));
}

inline size_t RotatingCandidateOrder(size_t candidate_index,
                                     uint64_t next_index,
                                     size_t candidate_count) {
  if (candidate_count == 0 || candidate_index >= candidate_count) {
    return std::numeric_limits<size_t>::max();
  }
  const size_t start_index = next_index % candidate_count;
  return (candidate_index + candidate_count - start_index) % candidate_count;
}

struct PrefillCandidateScore {
  int64_t estimated_prefill_time = std::numeric_limits<int64_t>::max();
  int64_t prefill_request_num = std::numeric_limits<int64_t>::max();
  int64_t prefill_token_num = std::numeric_limits<int64_t>::max();
  size_t order = std::numeric_limits<size_t>::max();
};

inline bool IsBetterPrefillCandidate(const PrefillCandidateScore& candidate,
                                     const PrefillCandidateScore& current) {
  if (candidate.estimated_prefill_time != current.estimated_prefill_time) {
    return candidate.estimated_prefill_time < current.estimated_prefill_time;
  }
  if (candidate.prefill_request_num != current.prefill_request_num) {
    return candidate.prefill_request_num < current.prefill_request_num;
  }
  if (candidate.prefill_token_num != current.prefill_token_num) {
    return candidate.prefill_token_num < current.prefill_token_num;
  }
  return candidate.order < current.order;
}

struct DecodeCandidateScore {
  int64_t estimated_tpot = std::numeric_limits<int64_t>::max();
  int64_t decode_request_num = std::numeric_limits<int64_t>::max();
  int64_t decode_token_num = std::numeric_limits<int64_t>::max();
  size_t order = std::numeric_limits<size_t>::max();
};

inline bool IsBetterDecodeCandidate(const DecodeCandidateScore& candidate,
                                    const DecodeCandidateScore& current) {
  if (candidate.estimated_tpot != current.estimated_tpot) {
    return candidate.estimated_tpot < current.estimated_tpot;
  }
  if (candidate.decode_request_num != current.decode_request_num) {
    return candidate.decode_request_num < current.decode_request_num;
  }
  if (candidate.decode_token_num != current.decode_token_num) {
    return candidate.decode_token_num < current.decode_token_num;
  }
  return candidate.order < current.order;
}

}  // namespace xllm_service
