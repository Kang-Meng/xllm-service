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

#include "request/failover_tracker.h"

namespace xllm_service {
namespace {

TEST(FailoverTracker, ArmFailoverAttemptCapturesPreviousRouteAndAttempt) {
  Request request;
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";

  const absl::Time now = absl::UnixEpoch() + absl::Milliseconds(1234);
  ArmFailoverAttempt(&request, now);

  EXPECT_EQ(request.failover_attempt, 1);
  EXPECT_TRUE(request.awaiting_failover_first_token);
  EXPECT_EQ(request.latest_failover_start_time, now);
  EXPECT_EQ(request.last_failover_from_prefill, "prefill-a");
  EXPECT_EQ(request.last_failover_from_decode, "decode-a");
}

TEST(FailoverTracker, ConsumeFailoverFirstTokenSampleReturnsLatencyOnce) {
  Request request;
  request.routing.prefill_name = "prefill-new";
  request.routing.decode_name = "decode-new";
  request.failover_attempt = 2;
  request.awaiting_failover_first_token = true;
  request.latest_failover_start_time =
      absl::UnixEpoch() + absl::Milliseconds(2000);
  request.last_failover_from_prefill = "prefill-old";
  request.last_failover_from_decode = "decode-old";

  const absl::Time first_token_time =
      absl::UnixEpoch() + absl::Milliseconds(2037);
  auto sample = ConsumeFailoverFirstTokenSample(&request, first_token_time);

  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->attempt, 2);
  EXPECT_EQ(sample->latency_ms, 37);
  EXPECT_EQ(sample->from_prefill, "prefill-old");
  EXPECT_EQ(sample->from_decode, "decode-old");
  EXPECT_EQ(sample->to_prefill, "prefill-new");
  EXPECT_EQ(sample->to_decode, "decode-new");
  EXPECT_FALSE(request.awaiting_failover_first_token);
  EXPECT_FALSE(
      ConsumeFailoverFirstTokenSample(&request, first_token_time).has_value());
}

TEST(FailoverTracker, ConsumeFailoverFirstTokenSampleIgnoresUnarmedRequest) {
  Request request;
  request.failover_attempt = 5;
  request.awaiting_failover_first_token = false;

  EXPECT_FALSE(ConsumeFailoverFirstTokenSample(
                   &request, absl::UnixEpoch() + absl::Milliseconds(10))
                   .has_value());
}

}  // namespace
}  // namespace xllm_service
