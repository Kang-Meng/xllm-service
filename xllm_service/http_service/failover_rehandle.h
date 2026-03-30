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

#include <string>
#include <utility>

#include "chat.pb.h"
#include "common/metrics.h"
#include "completion.pb.h"
#include "request/failover_tracker.h"

namespace xllm_service {

inline std::string BuildFailoverRehandleMessage(const Request& request) {
  return "failover_rehandle_start request_id=" + request.service_request_id +
         " attempt=" + std::to_string(request.failover_attempt) +
         " from_prefill=" + request.last_failover_from_prefill +
         " from_decode=" + request.last_failover_from_decode;
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
  LOG(INFO) << message;
  if (request->trace_callback) {
    request->trace_callback(message);
  }

  request->routing.prefill_name.clear();
  request->routing.decode_name.clear();
  if (!std::forward<ScheduleFn>(schedule_request)()) {
    return false;
  }

  req_pb->mutable_routing()->set_prefill_name(request->routing.prefill_name);
  req_pb->mutable_routing()->set_decode_name(request->routing.decode_name);
  std::forward<DispatchFn>(dispatch_request)();
  return true;
}

}  // namespace xllm_service
