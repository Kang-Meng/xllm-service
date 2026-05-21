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

#include "scheduler/managers/failover_partition.h"

namespace xllm_service {
namespace {

FailoverPartitionTarget MakeTarget(const std::string& name,
                                   int64_t base_compute_load_ms,
                                   size_t order) {
  FailoverPartitionTarget target;
  target.name = name;
  target.base_compute_load_ms = base_compute_load_ms;
  target.order = order;
  return target;
}

FailoverPartitionItem MakeItem(size_t index,
                               int64_t communication_cost_ms,
                               int64_t compute_cost_ms) {
  FailoverPartitionItem item;
  item.index = index;
  item.communication_cost_ms = communication_cost_ms;
  item.compute_cost_ms = compute_cost_ms;
  return item;
}

TEST(FailoverPartition, UsesAllPrefillTargets) {
  const std::vector<FailoverPartitionTarget> targets = {
      MakeTarget("prefill-a", 0, 0),
      MakeTarget("prefill-b", 0, 1),
      MakeTarget("prefill-c", 0, 2),
  };
  const std::vector<FailoverPartitionItem> items = {
      MakeItem(0, 0, 100),
      MakeItem(1, 0, 100),
      MakeItem(2, 0, 100),
  };

  const auto assignments = PlanFailoverRecoveryPartition(items, targets);

  ASSERT_EQ(assignments.size(), 3u);
  EXPECT_EQ(assignments[0], 0u);
  EXPECT_EQ(assignments[1], 1u);
  EXPECT_EQ(assignments[2], 2u);
}

TEST(FailoverPartition, AssignsLargestComputeToLowestBaseLoadTarget) {
  const std::vector<FailoverPartitionTarget> targets = {
      MakeTarget("prefill-low", 0, 0),
      MakeTarget("prefill-busy", 300, 1),
  };
  const std::vector<FailoverPartitionItem> items = {
      MakeItem(0, 0, 300),
  };

  const auto assignments = PlanFailoverRecoveryPartition(items, targets);

  ASSERT_EQ(assignments.size(), 1u);
  EXPECT_EQ(assignments[0], 0u);
}

TEST(FailoverPartition, BalancesCommunicationAndComputeDimensions) {
  const std::vector<FailoverPartitionTarget> targets = {
      MakeTarget("prefill-a", 0, 0),
      MakeTarget("prefill-b", 0, 1),
  };
  const std::vector<FailoverPartitionItem> items = {
      MakeItem(0, 100, 0),
      MakeItem(1, 0, 100),
  };

  const auto assignments = PlanFailoverRecoveryPartition(items, targets);

  ASSERT_EQ(assignments.size(), 2u);
  EXPECT_NE(assignments[0], assignments[1]);
}

TEST(FailoverPartition, ReturnsAssignmentsIndexedByOriginalItemIndex) {
  const std::vector<FailoverPartitionTarget> targets = {
      MakeTarget("prefill-a", 0, 0),
      MakeTarget("prefill-b", 0, 1),
  };
  const std::vector<FailoverPartitionItem> items = {
      MakeItem(2, 0, 100),
  };

  const auto assignments = PlanFailoverRecoveryPartition(items, targets);

  ASSERT_EQ(assignments.size(), 3u);
  EXPECT_LT(assignments[2], targets.size());
}

}  // namespace
}  // namespace xllm_service
