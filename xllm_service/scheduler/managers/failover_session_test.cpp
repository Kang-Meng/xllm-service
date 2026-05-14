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

#include "scheduler/scheduler.h"
#include "scheduler/managers/instance_mgr.h"

namespace xllm_service {

size_t Scheduler::clear_requests_on_failed_instance(const std::string&,
                                                    const std::string&,
                                                    InstanceType) {
  return 0;
}

void Scheduler::rehandle_removed_request() {}

namespace {

TEST(FailoverSession, DisconnectTriggersSameInstanceAndIncarnation) {
  const FailoverSessionState session{
      .instance_name = "decode-a:8010",
      .incarnation_id = "inc-1",
      .stream_id = 101,
      .connected_at_ms = 1234,
  };
  EXPECT_TRUE(ShouldTriggerFailoverOnSessionDisconnect(
      session, "decode-a:8010", "inc-1", 101));
}

TEST(FailoverSession, DisconnectIgnoresDifferentInstance) {
  const FailoverSessionState session{
      .instance_name = "decode-a:8010",
      .incarnation_id = "inc-1",
      .stream_id = 101,
      .connected_at_ms = 1234,
  };
  EXPECT_FALSE(ShouldTriggerFailoverOnSessionDisconnect(
      session, "decode-b:8010", "inc-1", 101));
}

TEST(FailoverSession, DisconnectIgnoresDifferentIncarnation) {
  const FailoverSessionState session{
      .instance_name = "decode-a:8010",
      .incarnation_id = "inc-1",
      .stream_id = 101,
      .connected_at_ms = 1234,
  };
  EXPECT_FALSE(ShouldTriggerFailoverOnSessionDisconnect(
      session, "decode-a:8010", "inc-2", 101));
}

TEST(FailoverSession, DisconnectIgnoresSupersededStream) {
  const FailoverSessionState session{
      .instance_name = "decode-a:8010",
      .incarnation_id = "inc-1",
      .stream_id = 202,
      .connected_at_ms = 1234,
  };
  EXPECT_FALSE(ShouldTriggerFailoverOnSessionDisconnect(
      session, "decode-a:8010", "inc-1", 101));
}

}  // namespace
}  // namespace xllm_service
