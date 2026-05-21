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

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "scheduler/request_context.h"

namespace xllm_service {

using RequestContextMap =
    std::unordered_map<std::string, std::shared_ptr<RequestContext>>;
using RequestRehandleCallback = std::function<void(std::shared_ptr<RequestContext>)>;
using RequestBatchPrepareCallback =
    std::function<bool(const std::vector<std::shared_ptr<RequestContext>>&)>;

class FailoverCoordinator {
 public:
  void register_request_rehandle_callback(RequestRehandleCallback cb);

  void register_batch_prepare_callback(RequestBatchPrepareCallback cb);

  void enqueue_removed_request(std::string service_request_id);

  std::optional<std::string> pop_first_removed_request_for_test();

  void rehandle_removed_requests(const RequestContextMap& request_contexts);

 private:
  std::vector<std::shared_ptr<RequestContext>> drain_removed_contexts(
      const RequestContextMap& request_contexts);

  RequestRehandleCallback request_rehandle_cb_;
  RequestBatchPrepareCallback batch_prepare_cb_;
  std::deque<std::string> removed_requests_;
  std::mutex mutex_;
};

}  // namespace xllm_service
