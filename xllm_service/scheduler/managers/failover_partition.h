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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace xllm_service {

struct FailoverPartitionItem {
  size_t index = 0;
  int64_t communication_cost_ms = 0;
  int64_t compute_cost_ms = 0;
};

struct FailoverPartitionTarget {
  std::string name;
  int64_t base_communication_load_ms = 0;
  int64_t base_compute_load_ms = 0;
  size_t order = 0;
};

inline double NormalizeLoad(int64_t value, double target) {
  return static_cast<double>(std::max<int64_t>(0, value)) /
         std::max(1.0, target);
}

inline std::vector<size_t> PlanFailoverRecoveryPartition(
    const std::vector<FailoverPartitionItem>& items,
    const std::vector<FailoverPartitionTarget>& targets) {
  const size_t invalid_target = std::numeric_limits<size_t>::max();
  size_t assignment_count = 0;
  for (const auto& item : items) {
    if (item.index == invalid_target) {
      continue;
    }
    assignment_count = std::max(assignment_count, item.index + 1);
  }

  std::vector<size_t> assignments(assignment_count, invalid_target);
  if (items.empty() || targets.empty() || assignments.empty()) {
    return assignments;
  }

  std::vector<int64_t> communication_loads(targets.size(), 0);
  std::vector<int64_t> compute_loads(targets.size(), 0);
  int64_t total_communication = 0;
  int64_t total_compute = 0;
  for (size_t i = 0; i < targets.size(); ++i) {
    communication_loads[i] =
        std::max<int64_t>(0, targets[i].base_communication_load_ms);
    compute_loads[i] = std::max<int64_t>(0, targets[i].base_compute_load_ms);
    total_communication += communication_loads[i];
    total_compute += compute_loads[i];
  }
  for (const auto& item : items) {
    total_communication += std::max<int64_t>(0, item.communication_cost_ms);
    total_compute += std::max<int64_t>(0, item.compute_cost_ms);
  }

  const double target_communication =
      static_cast<double>(total_communication) / targets.size();
  const double target_compute = static_cast<double>(total_compute) / targets.size();

  std::vector<size_t> item_order(items.size());
  std::iota(item_order.begin(), item_order.end(), 0);
  std::stable_sort(item_order.begin(),
                   item_order.end(),
                   [&](size_t lhs, size_t rhs) {
                     const auto& lhs_item = items[lhs];
                     const auto& rhs_item = items[rhs];
                     const double lhs_primary = std::max(
                         NormalizeLoad(lhs_item.communication_cost_ms,
                                       target_communication),
                         NormalizeLoad(lhs_item.compute_cost_ms, target_compute));
                     const double rhs_primary = std::max(
                         NormalizeLoad(rhs_item.communication_cost_ms,
                                       target_communication),
                         NormalizeLoad(rhs_item.compute_cost_ms, target_compute));
                     if (lhs_primary != rhs_primary) {
                       return lhs_primary > rhs_primary;
                     }
                     const int64_t lhs_total =
                         std::max<int64_t>(0, lhs_item.communication_cost_ms) +
                         std::max<int64_t>(0, lhs_item.compute_cost_ms);
                     const int64_t rhs_total =
                         std::max<int64_t>(0, rhs_item.communication_cost_ms) +
                         std::max<int64_t>(0, rhs_item.compute_cost_ms);
                     if (lhs_total != rhs_total) {
                       return lhs_total > rhs_total;
                     }
                     return lhs_item.index < rhs_item.index;
                   });

  for (const size_t item_pos : item_order) {
    const auto& item = items[item_pos];
    const int64_t communication_cost =
        std::max<int64_t>(0, item.communication_cost_ms);
    const int64_t compute_cost = std::max<int64_t>(0, item.compute_cost_ms);

    size_t best_target = 0;
    double best_primary = std::numeric_limits<double>::infinity();
    double best_secondary = std::numeric_limits<double>::infinity();
    int64_t best_base_compute = std::numeric_limits<int64_t>::max();
    size_t best_order = std::numeric_limits<size_t>::max();
    for (size_t target_pos = 0; target_pos < targets.size(); ++target_pos) {
      const double communication_score =
          NormalizeLoad(communication_loads[target_pos] + communication_cost,
                        target_communication);
      const double compute_score =
          NormalizeLoad(compute_loads[target_pos] + compute_cost,
                        target_compute);
      const double primary = std::max(communication_score, compute_score);
      const double secondary = communication_score + compute_score;
      const int64_t base_compute =
          std::max<int64_t>(0, targets[target_pos].base_compute_load_ms);
      const size_t order = targets[target_pos].order;
      if (primary < best_primary ||
          (primary == best_primary && secondary < best_secondary) ||
          (primary == best_primary && secondary == best_secondary &&
           base_compute < best_base_compute) ||
          (primary == best_primary && secondary == best_secondary &&
           base_compute == best_base_compute && order < best_order)) {
        best_target = target_pos;
        best_primary = primary;
        best_secondary = secondary;
        best_base_compute = base_compute;
        best_order = order;
      }
    }

    if (item.index < assignments.size()) {
      assignments[item.index] = best_target;
    }
    communication_loads[best_target] += communication_cost;
    compute_loads[best_target] += compute_cost;
  }

  return assignments;
}

}  // namespace xllm_service
