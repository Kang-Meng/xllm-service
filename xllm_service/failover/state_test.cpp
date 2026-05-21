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

TEST(FailoverState, RequestKeepsReplayRuntimeAndSloStateGrouped) {
  Request request;

  EXPECT_EQ(request.failover.replay.base_prompt_token_count, 0u);
  EXPECT_TRUE(request.failover.replay.committed_output_token_ids.empty());
  EXPECT_TRUE(request.failover.replay.committed_output_text.empty());
  EXPECT_TRUE(request.failover.replay.accumulated_output_text.empty());
  EXPECT_FALSE(request.failover.replay.pending_decode_handoff_dedup);
  EXPECT_FALSE(request.failover.replay.decode_handoff_dedup_consumed);

  EXPECT_EQ(request.failover.runtime.attempt, 0);
  EXPECT_FALSE(request.failover.runtime.awaiting_first_token);
  EXPECT_EQ(request.failover.runtime.type, FailoverType::NONE);
  EXPECT_TRUE(request.failover.runtime.planned_prefill_name.empty());
  EXPECT_EQ(request.failover.runtime.detected_time, absl::InfinitePast());

  EXPECT_EQ(request.failover.slo.original_ttft_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.original_tpot_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.original_ttlt_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.prefill_recovery_ttft_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.decode_recovery_ttft_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.effective_ttft_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.effective_tpot_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.effective_ttlt_ms, kUnsetSloMs);
}

}  // namespace
}  // namespace xllm_service
