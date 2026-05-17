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

#include "failover/coordinator.h"

#include <algorithm>

#include <glog/logging.h>

#include "common/metrics.h"
#include "failover/tracker.h"

namespace xllm_service {

void FailoverCoordinator::register_request_rehandle_callback(
    RequestRehandleCallback cb) {
  std::lock_guard<std::mutex> guard(mutex_);
  request_rehandle_cb_ = std::move(cb);
}

void FailoverCoordinator::register_batch_prepare_callback(
    RequestBatchPrepareCallback cb) {
  std::lock_guard<std::mutex> guard(mutex_);
  batch_prepare_cb_ = std::move(cb);
}

void FailoverCoordinator::enqueue_removed_request(std::string service_request_id) {
  std::lock_guard<std::mutex> guard(mutex_);
  removed_requests_.push_back(service_request_id);
  GAUGE_SET(active_failover_removed_requests, removed_requests_.size());
  LOG(INFO) << "failover_removed_request_enqueued"
            << " request_id=" << service_request_id
            << " queue_size=" << removed_requests_.size();
}

std::optional<std::string> FailoverCoordinator::pop_first_removed_request_for_test() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (removed_requests_.empty()) {
    return std::nullopt;
  }
  std::string request_id = removed_requests_.front();
  removed_requests_.pop_front();
  GAUGE_SET(active_failover_removed_requests, removed_requests_.size());
  return request_id;
}

std::vector<std::shared_ptr<RequestContext>>
FailoverCoordinator::drain_removed_contexts(
    const RequestContextMap& request_contexts) {
  std::vector<std::shared_ptr<RequestContext>> removed_contexts;
  std::lock_guard<std::mutex> guard(mutex_);
  while (!removed_requests_.empty()) {
    std::string service_request_id = removed_requests_.front();
    removed_requests_.pop_front();

    auto it = request_contexts.find(service_request_id);
    if (it != request_contexts.end()) {
      removed_contexts.emplace_back(it->second);
    } else {
      LOG(ERROR) << "failover_rehandle_context_missing"
                 << " request_id=" << service_request_id;
    }
  }
  GAUGE_SET(active_failover_removed_requests, removed_requests_.size());
  return removed_contexts;
}

void FailoverCoordinator::rehandle_removed_requests(
    const RequestContextMap& request_contexts) {
  RequestRehandleCallback callback;
  RequestBatchPrepareCallback batch_prepare;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    callback = request_rehandle_cb_;
    batch_prepare = batch_prepare_cb_;
    if (!callback) {
      if (!removed_requests_.empty()) {
        LOG(ERROR) << "failover_rehandle_callback_missing"
                   << " count=" << removed_requests_.size();
      }
      return;
    }
  }

  auto removed_contexts = drain_removed_contexts(request_contexts);
  if (removed_contexts.empty()) {
    return;
  }

  LOG(INFO) << "failover_rehandle_batch_start"
            << " count=" << removed_contexts.size();
  std::stable_sort(removed_contexts.begin(),
                   removed_contexts.end(),
                   [](const std::shared_ptr<RequestContext>& lhs,
                      const std::shared_ptr<RequestContext>& rhs) {
                     const auto& lhs_request = lhs->request();
                     const auto& rhs_request = rhs->request();
                     return lhs_request->failover.runtime.estimated_total_recovery_cost_ms <
                            rhs_request->failover.runtime.estimated_total_recovery_cost_ms;
                   });

  if (batch_prepare && !batch_prepare(removed_contexts)) {
    // The batch-level prefill partition planner refused this batch (e.g. no
    // schedulable prefill at planning time, or partition algorithm failed).
    // We must not drop the requests on the floor: they have already been
    // drained out of the queue. Fall through to per-request dispatch -- the
    // per-request schedule_failover path (select_instance_pair_on_failover)
    // is the canonical fallback and is robust to a missing planned_prefill.
    LOG(ERROR) << "failover_rehandle_batch_prepare_failed_fallback"
               << " count=" << removed_contexts.size();
    for (const auto& req_context : removed_contexts) {
      if (req_context != nullptr && req_context->request() != nullptr) {
        req_context->request()->failover.runtime.planned_prefill_name.clear();
      }
    }
  }

  for (const auto& req_context : removed_contexts) {
    const auto& request = req_context->request();
    DLOG(INFO) << "failover_rehandle_dispatch"
              << " request_id=" << request->service_request_id
              << " failover_type=" << FailoverTypeName(request->failover.runtime.type)
              << " recovery_cost_ms="
              << request->failover.runtime.estimated_total_recovery_cost_ms
              << " from_prefill=" << request->routing.prefill_name
              << " from_decode=" << request->routing.decode_name;
    callback(req_context);
  }
}

}  // namespace xllm_service
