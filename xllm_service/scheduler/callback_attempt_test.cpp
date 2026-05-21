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

#include "request/request.h"

namespace xllm_service {
namespace {

TEST(CallbackAttemptTest, CallbackAttemptRemainsBoundToOriginalCallback) {
  Request request;
  request.failover.runtime.attempt = 1;
  request.callback_attempt = request.failover.runtime.attempt;

  request.failover.runtime.attempt = 2;

  EXPECT_EQ(request.callback_attempt, 1);
  EXPECT_EQ(request.failover.runtime.attempt, 2);
}

}  // namespace
}  // namespace xllm_service
