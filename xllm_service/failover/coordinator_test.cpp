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

#include "failover/coordinator.h"

namespace xllm_service {
namespace {

std::shared_ptr<RequestContext> MakeRequestContext(
    const std::shared_ptr<Request>& request) {
  return std::make_shared<RequestContext>(
      std::shared_ptr<CallData>(), request, nullptr);
}

TEST(FailoverCoordinator, enqueue_and_pop_removed_request_for_test) {
  FailoverCoordinator coordinator;

  coordinator.enqueue_removed_request("req-1");
  coordinator.enqueue_removed_request("req-2");

  auto first = coordinator.pop_first_removed_request_for_test();
  auto second = coordinator.pop_first_removed_request_for_test();
  auto empty = coordinator.pop_first_removed_request_for_test();

  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(*first, "req-1");
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*second, "req-2");
  EXPECT_FALSE(empty.has_value());
}

TEST(FailoverCoordinator, BatchPrepareRunsBeforeRehandleCallbacks) {
  FailoverCoordinator coordinator;
  RequestContextMap request_contexts;

  auto first = std::make_shared<Request>();
  first->service_request_id = "req-1";
  first->failover.runtime.estimated_total_recovery_cost_ms = 200;
  request_contexts.emplace("req-1", MakeRequestContext(first));

  auto second = std::make_shared<Request>();
  second->service_request_id = "req-2";
  second->failover.runtime.estimated_total_recovery_cost_ms = 100;
  request_contexts.emplace("req-2", MakeRequestContext(second));

  std::vector<std::string> rehandled_ids;
  coordinator.register_batch_prepare_callback(
      [](const std::vector<std::shared_ptr<RequestContext>>& contexts) {
        for (const auto& context : contexts) {
          context->request()->routing.prefill_name = "prefill-planned";
        }
        return true;
      });
  coordinator.register_request_rehandle_callback(
      [&](std::shared_ptr<RequestContext> context) {
        EXPECT_EQ(context->request()->routing.prefill_name, "prefill-planned");
        rehandled_ids.push_back(context->request()->service_request_id);
      });

  coordinator.enqueue_removed_request("req-1");
  coordinator.enqueue_removed_request("req-2");
  coordinator.rehandle_removed_requests(request_contexts);

  ASSERT_EQ(rehandled_ids.size(), 2u);
  EXPECT_EQ(rehandled_ids[0], "req-2");
  EXPECT_EQ(rehandled_ids[1], "req-1");
}

}  // namespace
}  // namespace xllm_service
