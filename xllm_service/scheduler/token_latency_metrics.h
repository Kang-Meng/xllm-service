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
#include <glog/logging.h>

#include <optional>
#include <string>

#include "common/metrics.h"
#include "request/failover_tracker.h"

namespace xllm_service {

inline std::string BuildFailoverFirstTokenMessage(
    const Request& request,
    const FailoverFirstTokenSample& sample) {
  return "failover_first_token request_id=" + request.service_request_id +
         " attempt=" + std::to_string(sample.attempt) +
         " latency_ms=" + std::to_string(sample.latency_ms) +
         " from_prefill=" + sample.from_prefill +
         " from_decode=" + sample.from_decode +
         " to_prefill=" + sample.to_prefill +
         " to_decode=" + sample.to_decode;
}

inline std::optional<FailoverFirstTokenSample> ObserveTokenLatencyMetrics(
    Request* request,
    bool finished_on_prefill_instance,
    absl::Time now) {
  const int64_t tbt_milliseconds =
      absl::ToInt64Milliseconds(now - request->latest_generate_time);
  request->latest_generate_time = now;

  if (finished_on_prefill_instance) {
    HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                      tbt_milliseconds);
    auto sample = ConsumeFailoverFirstTokenSample(request, now);
    if (sample.has_value()) {
      HISTOGRAM_OBSERVE(failover_time_to_first_token_latency_milliseconds,
                        sample->latency_ms);
      const std::string message =
          BuildFailoverFirstTokenMessage(*request, *sample);
      LOG(INFO) << message;
      if (request->trace_callback) {
        request->trace_callback(message);
      }
    }
    return sample;
  }

  HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
  return std::nullopt;
}

}  // namespace xllm_service
