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
#include "http_service/failover_rehandle.h"

namespace xllm_service {
namespace {

TEST(FailoverRehandle, SuccessfulRehandleArmsTimingUpdatesProtoAndDispatches) {
  Request request;
  request.service_request_id = "req-1";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";

  std::string trace_message;
  request.trace_callback = [&](const std::string& message) {
    trace_message = message;
  };

  xllm::proto::CompletionRequest req_pb;
  const double rehandle_count_before =
      COUNTER_failover_rehandle_total.get_value();
  int dispatch_count = 0;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        request.routing.prefill_name = "prefill-b";
        request.routing.decode_name = "decode-b";
        return true;
      },
      [&]() { ++dispatch_count; });

  EXPECT_TRUE(ok);
  EXPECT_EQ(request.failover_attempt, 1);
  EXPECT_TRUE(request.awaiting_failover_first_token);
  EXPECT_EQ(request.last_failover_from_prefill, "prefill-a");
  EXPECT_EQ(request.last_failover_from_decode, "decode-a");
  EXPECT_EQ(request.latest_failover_start_time,
            absl::UnixEpoch() + absl::Milliseconds(1000));
  EXPECT_EQ(req_pb.routing().prefill_name(), "prefill-b");
  EXPECT_EQ(req_pb.routing().decode_name(), "decode-b");
  EXPECT_EQ(dispatch_count, 1);
  EXPECT_DOUBLE_EQ(COUNTER_failover_rehandle_total.get_value(),
                   rehandle_count_before + 1);
  EXPECT_NE(trace_message.find("failover_rehandle_start"), std::string::npos);
  EXPECT_NE(trace_message.find("from_prefill=prefill-a"), std::string::npos);
}

TEST(FailoverRehandle, FailedRescheduleDoesNotDispatch) {
  Request request;
  request.service_request_id = "req-2";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";

  xllm::proto::ChatRequest req_pb;
  const double rehandle_count_before =
      COUNTER_failover_rehandle_total.get_value();
  int dispatch_count = 0;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(2000),
      [&]() { return false; },
      [&]() { ++dispatch_count; });

  EXPECT_FALSE(ok);
  EXPECT_EQ(dispatch_count, 0);
  EXPECT_TRUE(request.awaiting_failover_first_token);
  EXPECT_EQ(req_pb.routing().prefill_name(), "");
  EXPECT_EQ(req_pb.routing().decode_name(), "");
  EXPECT_DOUBLE_EQ(COUNTER_failover_rehandle_total.get_value(),
                   rehandle_count_before + 1);
}

}  // namespace
}  // namespace xllm_service
