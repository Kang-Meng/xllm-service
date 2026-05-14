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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <string>
#include <utility>

#include "chat.pb.h"
#include "common/metrics.h"
#include "completion.pb.h"
#include "failover/tracker.h"

namespace xllm_service {

inline std::string BuildFailoverRehandleMessage(const Request& request) {
  return "failover_rehandle_start request_id=" + request.service_request_id +
         " attempt=" + std::to_string(request.failover.runtime.attempt) +
         " from_prefill=" + request.failover.runtime.last_from_prefill +
         " from_decode=" + request.failover.runtime.last_from_decode +
         " failover_type=" + FailoverTypeName(request.failover.runtime.type) +
         " recovery_cost_ms=" +
         std::to_string(request.failover.runtime.estimated_total_recovery_cost_ms);
}

template <typename RequestProto>
void WriteFailoverSloToProto(const Request& request, RequestProto* req_pb) {
  if (request.failover.slo.effective_ttft_ms == kUnsetSloMs) {
    req_pb->clear_ttft_slo_ms();
  } else {
    req_pb->set_ttft_slo_ms(request.failover.slo.effective_ttft_ms);
  }

  if (request.failover.slo.effective_tpot_ms == kUnsetSloMs) {
    req_pb->clear_tpot_slo_ms();
  } else {
    req_pb->set_tpot_slo_ms(request.failover.slo.effective_tpot_ms);
  }

  if (request.failover.slo.effective_ttlt_ms == kUnsetSloMs) {
    req_pb->clear_ttlt_slo_ms();
  } else {
    req_pb->set_ttlt_slo_ms(request.failover.slo.effective_ttlt_ms);
  }
}

template <typename RequestProto, typename ScheduleFn, typename DispatchFn>
bool RehandleScheduledRequest(Request* request,
                              RequestProto* req_pb,
                              absl::Time now,
                              ScheduleFn&& schedule_request,
                              DispatchFn&& dispatch_request) {
  ArmFailoverAttempt(request, now);
  COUNTER_INC(failover_rehandle_total);

  const std::string message = BuildFailoverRehandleMessage(*request);
  DLOG(INFO) << message;
  if (request->trace_callback) {
    request->trace_callback(message);
  }

  request->routing.prefill_name.clear();
  request->routing.decode_name.clear();
  if (!std::forward<ScheduleFn>(schedule_request)()) {
    LOG(ERROR) << "failover_rehandle_schedule_failed"
               << " request_id=" << request->service_request_id
               << " attempt=" << request->failover.runtime.attempt
               << " failover_type=" << FailoverTypeName(request->failover.runtime.type)
               << " from_prefill=" << request->failover.runtime.last_from_prefill
               << " from_decode=" << request->failover.runtime.last_from_decode;
    return false;
  }

  req_pb->mutable_routing()->set_prefill_name(request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(request->routing.decode_name);
  WriteFailoverSloToProto(*request, req_pb);
  request->failover.runtime.redispatched_time = absl::Now();
  const int64_t detected_to_redispatch_ms =
      request->failover.runtime.detected_time == absl::InfinitePast()
          ? 0
          : absl::ToInt64Milliseconds(request->failover.runtime.redispatched_time -
                                      request->failover.runtime.detected_time);
  LOG(INFO) << "failover_rehandle_redispatch"
            << " request_id=" << request->service_request_id
            << " attempt=" << request->failover.runtime.attempt
            << " failover_type=" << FailoverTypeName(request->failover.runtime.type)
            << " to_prefill=" << request->routing.prefill_name
            << " to_decode=" << request->routing.decode_name
            << " detected_to_redispatch_ms="
            << detected_to_redispatch_ms;
  std::forward<DispatchFn>(dispatch_request)();
  return true;
}

}  // namespace xllm_service
