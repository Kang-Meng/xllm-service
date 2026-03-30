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

#include <optional>
#include <string>

#include "request.h"

namespace xllm_service {

struct FailoverFirstTokenSample {
  int32_t attempt = 0;
  int64_t latency_ms = 0;
  std::string from_prefill;
  std::string from_decode;
  std::string to_prefill;
  std::string to_decode;
};

inline void ArmFailoverAttempt(Request* request, absl::Time now) {
  request->last_failover_from_prefill = request->routing.prefill_name;
  request->last_failover_from_decode = request->routing.decode_name;
  request->failover_attempt += 1;
  request->awaiting_failover_first_token = true;
  request->latest_failover_start_time = now;
}

inline std::optional<FailoverFirstTokenSample> ConsumeFailoverFirstTokenSample(
    Request* request,
    absl::Time now) {
  if (!request->awaiting_failover_first_token) {
    return std::nullopt;
  }

  request->awaiting_failover_first_token = false;
  return FailoverFirstTokenSample{
      .attempt = request->failover_attempt,
      .latency_ms =
          absl::ToInt64Milliseconds(now - request->latest_failover_start_time),
      .from_prefill = request->last_failover_from_prefill,
      .from_decode = request->last_failover_from_decode,
      .to_prefill = request->routing.prefill_name,
      .to_decode = request->routing.decode_name,
  };
}

}  // namespace xllm_service
