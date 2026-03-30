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

#include <gtest/gtest.h>

#include "common/metrics.h"
#include "scheduler/token_latency_metrics.h"

namespace xllm_service {
namespace {

TEST(TokenLatencyMetrics, ObserveFailoverFirstTokenRecordsDedicatedHistogram) {
  Request request;
  request.routing.prefill_name = "prefill-new";
  request.routing.decode_name = "decode-new";
  request.failover_attempt = 2;
  request.awaiting_failover_first_token = true;
  request.latest_failover_start_time =
      absl::UnixEpoch() + absl::Milliseconds(2000);
  request.latest_generate_time = absl::UnixEpoch() + absl::Milliseconds(2025);
  request.last_failover_from_prefill = "prefill-old";
  request.last_failover_from_decode = "decode-old";

  std::string trace_message;
  request.trace_callback = [&](const std::string& message) {
    trace_message = message;
  };

  const auto ttft_count_before =
      HISTOGRAM_time_to_first_token_latency_milliseconds.count();
  const auto failover_count_before =
      HISTOGRAM_failover_time_to_first_token_latency_milliseconds.count();

  auto sample = ObserveTokenLatencyMetrics(
      &request, true, absl::UnixEpoch() + absl::Milliseconds(2037));

  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->attempt, 2);
  EXPECT_EQ(sample->latency_ms, 37);
  EXPECT_FALSE(request.awaiting_failover_first_token);
  EXPECT_EQ(HISTOGRAM_time_to_first_token_latency_milliseconds.count(),
            ttft_count_before + 1);
  EXPECT_EQ(HISTOGRAM_failover_time_to_first_token_latency_milliseconds.count(),
            failover_count_before + 1);
  EXPECT_NE(trace_message.find("failover_first_token"), std::string::npos);
  EXPECT_NE(trace_message.find("from_prefill=prefill-old"), std::string::npos);
  EXPECT_NE(trace_message.find("to_prefill=prefill-new"), std::string::npos);
}

TEST(TokenLatencyMetrics, NormalFirstTokenDoesNotRecordFailoverHistogram) {
  Request request;
  request.awaiting_failover_first_token = false;
  request.latest_generate_time = absl::UnixEpoch() + absl::Milliseconds(1000);

  const auto ttft_count_before =
      HISTOGRAM_time_to_first_token_latency_milliseconds.count();
  const auto failover_count_before =
      HISTOGRAM_failover_time_to_first_token_latency_milliseconds.count();

  auto sample = ObserveTokenLatencyMetrics(
      &request, true, absl::UnixEpoch() + absl::Milliseconds(1011));

  EXPECT_FALSE(sample.has_value());
  EXPECT_EQ(HISTOGRAM_time_to_first_token_latency_milliseconds.count(),
            ttft_count_before + 1);
  EXPECT_EQ(HISTOGRAM_failover_time_to_first_token_latency_milliseconds.count(),
            failover_count_before);
}

}  // namespace
}  // namespace xllm_service
