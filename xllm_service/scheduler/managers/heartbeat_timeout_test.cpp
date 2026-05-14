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

bool ShouldMarkInstanceSuspectOnHeartbeatTimeout(
    InstanceRuntimeState runtime_state,
    uint64_t latest_timestamp_ms,
    uint64_t now_ms,
    int64_t heartbeat_timeout_ms);

InstanceRuntimeState RestoreRuntimeStateAfterHeartbeat(
    InstanceRuntimeState previous_runtime_state);

bool IsInstanceSchedulableForTest(InstanceRuntimeState runtime_state);

namespace {

TEST(HeartbeatTimeout, ActiveHeartbeatTimeoutHelperStillMatchesLegacyRule) {
  EXPECT_TRUE(ShouldMarkInstanceSuspectOnHeartbeatTimeout(
      InstanceRuntimeState::ACTIVE,
      /*latest_timestamp_ms=*/1000,
      /*now_ms=*/4001,
      /*heartbeat_timeout_ms=*/3000));
}

TEST(HeartbeatTimeout, ActiveHeartbeatTimeoutHelperStaysFalseBeforeThreshold) {
  EXPECT_FALSE(ShouldMarkInstanceSuspectOnHeartbeatTimeout(
      InstanceRuntimeState::ACTIVE,
      /*latest_timestamp_ms=*/1000,
      /*now_ms=*/3999,
      /*heartbeat_timeout_ms=*/3000));
}

TEST(HeartbeatTimeout, LeaseLostHeartbeatTimeoutHelperDoesNotFire) {
  EXPECT_FALSE(ShouldMarkInstanceSuspectOnHeartbeatTimeout(
      InstanceRuntimeState::LEASE_LOST,
      /*latest_timestamp_ms=*/1000,
      /*now_ms=*/10000,
      /*heartbeat_timeout_ms=*/3000));
}

TEST(HeartbeatTimeout, RecoveredActiveSuspectReturnsToActive) {
  EXPECT_EQ(RestoreRuntimeStateAfterHeartbeat(InstanceRuntimeState::ACTIVE),
            InstanceRuntimeState::ACTIVE);
}

TEST(HeartbeatTimeout, RecoveredLeaseLostSuspectReturnsToLeaseLost) {
  EXPECT_EQ(RestoreRuntimeStateAfterHeartbeat(InstanceRuntimeState::LEASE_LOST),
            InstanceRuntimeState::LEASE_LOST);
}

TEST(HeartbeatTimeout, LeaseLostInstanceIsNotSchedulable) {
  EXPECT_FALSE(IsInstanceSchedulableForTest(InstanceRuntimeState::LEASE_LOST));
}

TEST(HeartbeatTimeout, ActiveInstanceRemainsSchedulable) {
  EXPECT_TRUE(IsInstanceSchedulableForTest(InstanceRuntimeState::ACTIVE));
}

}  // namespace
}  // namespace xllm_service
