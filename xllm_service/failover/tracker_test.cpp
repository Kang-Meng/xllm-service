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

#include "failover/tracker.h"
#include "scheduler/decode_offload_tracking.h"

namespace xllm_service {
namespace {

TEST(FailoverTracker, ArmFailoverAttemptCapturesPreviousRouteAndAttempt) {
  Request request;
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";
  request.offload_batch_size = 32;

  const absl::Time now = absl::UnixEpoch() + absl::Milliseconds(1234);
  ArmFailoverAttempt(&request, now);

  EXPECT_EQ(request.failover.runtime.attempt, 1);
  EXPECT_TRUE(request.failover.runtime.awaiting_first_token);
  EXPECT_EQ(request.failover.runtime.latest_start_time, now);
  EXPECT_EQ(request.failover.runtime.last_from_prefill, "prefill-a");
  EXPECT_EQ(request.failover.runtime.last_from_decode, "decode-a");
  EXPECT_EQ(request.offload_batch_size, UINT32_MAX);
}

TEST(FailoverTracker, ConsumeFailoverFirstTokenSampleReturnsLatencyOnce) {
  Request request;
  request.routing.prefill_name = "prefill-new";
  request.routing.decode_name = "decode-new";
  request.failover.runtime.attempt = 2;
  request.failover.runtime.awaiting_first_token = true;
  request.failover.runtime.latest_start_time =
      absl::UnixEpoch() + absl::Milliseconds(2000);
  request.failover.runtime.last_from_prefill = "prefill-old";
  request.failover.runtime.last_from_decode = "decode-old";

  const absl::Time first_token_time =
      absl::UnixEpoch() + absl::Milliseconds(2037);
  auto sample = ConsumeFailoverFirstTokenSample(&request, first_token_time);

  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->attempt, 2);
  EXPECT_EQ(sample->latency_ms, 37);
  EXPECT_EQ(sample->from_prefill, "prefill-old");
  EXPECT_EQ(sample->from_decode, "decode-old");
  EXPECT_EQ(sample->to_prefill, "prefill-new");
  EXPECT_EQ(sample->to_decode, "decode-new");
  EXPECT_FALSE(request.failover.runtime.awaiting_first_token);
  EXPECT_FALSE(
      ConsumeFailoverFirstTokenSample(&request, first_token_time).has_value());
}

TEST(FailoverTracker, ConsumeFailoverFirstTokenSampleIgnoresUnarmedRequest) {
  Request request;
  request.failover.runtime.attempt = 5;
  request.failover.runtime.awaiting_first_token = false;

  EXPECT_FALSE(ConsumeFailoverFirstTokenSample(
                   &request, absl::UnixEpoch() + absl::Milliseconds(10))
                   .has_value());
}

TEST(FailoverTracker, DoesNotRecordDecodeOffloadBatchOnPrefillToken) {
  Request request;
  request.routing.decode_name = "decode-a";

  int getter_calls = 0;
  const bool recorded = MaybeRecordFirstDecodeOffloadBatchSize(
      &request, true, [&](const std::string&) {
        ++getter_calls;
        return 8u;
      });

  EXPECT_FALSE(recorded);
  EXPECT_EQ(getter_calls, 0);
  EXPECT_EQ(request.offload_batch_size, UINT32_MAX);
  EXPECT_FALSE(request.decode_offload_batch_size_recorded);
}

TEST(FailoverTracker, RecordsDecodeOffloadBatchOnFirstDecodeToken) {
  Request request;
  request.prefill_stage_finished = true;
  request.routing.decode_name = "decode-a";

  int getter_calls = 0;
  const bool recorded = MaybeRecordFirstDecodeOffloadBatchSize(
      &request, false, [&](const std::string& instance_name) {
        ++getter_calls;
        EXPECT_EQ(instance_name, "decode-a");
        return 8u;
      });

  EXPECT_TRUE(recorded);
  EXPECT_EQ(getter_calls, 1);
  EXPECT_EQ(request.offload_batch_size, 8u);
  EXPECT_TRUE(request.decode_offload_batch_size_recorded);
}

TEST(FailoverTracker, DoesNotOverwriteDecodeOffloadBatchAfterFirstDecodeToken) {
  Request request;
  request.prefill_stage_finished = true;
  request.routing.decode_name = "decode-a";

  int getter_calls = 0;
  EXPECT_TRUE(MaybeRecordFirstDecodeOffloadBatchSize(
      &request, false, [&](const std::string&) {
        ++getter_calls;
        return 8u;
      }));
  EXPECT_FALSE(MaybeRecordFirstDecodeOffloadBatchSize(
      &request, false, [&](const std::string&) {
        ++getter_calls;
        return 16u;
      }));

  EXPECT_EQ(getter_calls, 1);
  EXPECT_EQ(request.offload_batch_size, 8u);
}

TEST(FailoverTracker, LatchesUnknownDecodeOffloadBatchOnFirstDecodeToken) {
  Request request;
  request.prefill_stage_finished = true;
  request.routing.decode_name = "decode-a";

  int getter_calls = 0;
  EXPECT_TRUE(MaybeRecordFirstDecodeOffloadBatchSize(
      &request, false, [&](const std::string&) {
        ++getter_calls;
        return UINT32_MAX;
      }));
  EXPECT_FALSE(MaybeRecordFirstDecodeOffloadBatchSize(
      &request, false, [&](const std::string&) {
        ++getter_calls;
        return 16u;
      }));

  EXPECT_EQ(getter_calls, 1);
  EXPECT_EQ(request.offload_batch_size, UINT32_MAX);
  EXPECT_TRUE(request.decode_offload_batch_size_recorded);
}

TEST(FailoverTracker, FailoverRearmsDecodeOffloadBatchCapture) {
  Request request;
  request.prefill_stage_finished = true;
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";

  EXPECT_TRUE(MaybeRecordFirstDecodeOffloadBatchSize(
      &request, false, [&](const std::string&) { return 8u; }));
  EXPECT_EQ(request.offload_batch_size, 8u);

  ArmFailoverAttempt(&request, absl::UnixEpoch() + absl::Milliseconds(1000));
  request.routing.decode_name = "decode-b";

  int getter_calls = 0;
  EXPECT_TRUE(MaybeRecordFirstDecodeOffloadBatchSize(
      &request, false, [&](const std::string& instance_name) {
        ++getter_calls;
        EXPECT_EQ(instance_name, "decode-b");
        return 16u;
      }));

  EXPECT_EQ(getter_calls, 1);
  EXPECT_EQ(request.offload_batch_size, 16u);
  EXPECT_TRUE(request.decode_offload_batch_size_recorded);
}

TEST(FailoverRecoveryProfile, PrefillCrashUsesFullPromptRecomputeCost) {
  Request request;
  request.token_ids.resize(256);
  request.failover.slo.original_ttft_ms = 1000;
  request.failover.slo.original_ttlt_ms = 4000;
  request.failover.slo.prefill_recovery_ttft_ms = 640;

  FailoverRecoveryConfig config;
  config.prefill_recompute_token_cost_ms = 2;

  MarkRequestForFailover(
      &request,
      FailoverType::PREFILL_CRASH,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      /*block_size=*/128,
      config);

  EXPECT_EQ(request.failover.runtime.type, FailoverType::PREFILL_CRASH);
  EXPECT_EQ(request.failover.runtime.detected_time,
            absl::UnixEpoch() + absl::Milliseconds(1000));
  EXPECT_EQ(request.failover.runtime.estimated_restore_cost_ms, 0);
  EXPECT_EQ(request.failover.runtime.estimated_recompute_cost_ms, 512);
  EXPECT_EQ(request.failover.runtime.estimated_total_recovery_cost_ms, 512);
  EXPECT_EQ(request.failover.runtime.estimated_restore_blocks, 0);
  EXPECT_EQ(request.failover.runtime.estimated_recompute_tokens, 256);
  EXPECT_EQ(request.failover.slo.effective_ttft_ms, 640);
  EXPECT_EQ(request.failover.slo.effective_tpot_ms, kUnsetSloMs);
  EXPECT_EQ(request.failover.slo.effective_ttlt_ms, 4000);
  EXPECT_EQ(request.failover.runtime.generated_tokens, 0);
  EXPECT_EQ(request.failover.runtime.previous_token_time, absl::InfinitePast());
}

TEST(FailoverRecoveryProfile, DecodeCrashSplitsOffloadedAndTailBlocks) {
  Request request;
  request.num_generated_tokens = 300;
  request.failover.runtime.previous_token_time =
      absl::UnixEpoch() + absl::Milliseconds(500);
  request.latest_generate_time = request.failover.runtime.previous_token_time;
  request.prefill_stage_finished = true;
  request.offload_batch_size = 2;
  request.failover.slo.original_ttft_ms = 500;
  request.failover.slo.original_tpot_ms = 100;
  request.failover.slo.original_ttlt_ms = 10000;
  request.failover.slo.decode_recovery_ttft_ms = 180;

  FailoverRecoveryConfig config;
  config.restore_block_cost_ms = 10;
  config.decode_recompute_token_cost_ms = 3;

  MarkRequestForFailover(
      &request,
      FailoverType::DECODE_CRASH,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      /*block_size=*/100,
      config);

  EXPECT_EQ(request.failover.runtime.type, FailoverType::DECODE_CRASH);
  EXPECT_EQ(request.failover.runtime.failed_decode_offload_batch_size, 2u);
  EXPECT_EQ(request.failover.runtime.estimated_restore_cost_ms, 20);
  EXPECT_EQ(request.failover.runtime.estimated_recompute_cost_ms, 300);
  EXPECT_EQ(request.failover.runtime.estimated_total_recovery_cost_ms, 320);
  EXPECT_EQ(request.failover.runtime.estimated_restore_blocks, 2);
  EXPECT_EQ(request.failover.runtime.estimated_recompute_tokens, 100);
  EXPECT_EQ(request.failover.slo.effective_ttft_ms, 180);
  EXPECT_EQ(request.failover.slo.effective_tpot_ms, 100);
  EXPECT_EQ(request.failover.slo.effective_ttlt_ms, 10000);
  EXPECT_EQ(request.failover.runtime.generated_tokens, 300);
  EXPECT_EQ(request.failover.runtime.previous_token_time,
            absl::UnixEpoch() + absl::Milliseconds(500));
}

TEST(FailoverRecoveryProfile, DecodeCrashUnknownOffloadRecomputesAllTokens) {
  Request request;
  request.num_generated_tokens = 32;
  request.offload_batch_size = UINT32_MAX;
  request.failover.slo.original_ttft_ms = 400;
  request.failover.slo.original_tpot_ms = 100;

  FailoverRecoveryConfig config;
  config.restore_block_cost_ms = 10;
  config.decode_recompute_token_cost_ms = 3;

  MarkRequestForFailover(
      &request,
      FailoverType::DECODE_CRASH,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      /*block_size=*/16,
      config);

  EXPECT_EQ(request.failover.runtime.estimated_restore_cost_ms, 0);
  EXPECT_EQ(request.failover.runtime.estimated_recompute_cost_ms, 96);
  EXPECT_EQ(request.failover.runtime.estimated_total_recovery_cost_ms, 96);
  EXPECT_EQ(request.failover.runtime.estimated_restore_blocks, 0);
  EXPECT_EQ(request.failover.runtime.estimated_recompute_tokens, 32);
  EXPECT_EQ(request.failover.slo.effective_ttft_ms, 400);
  EXPECT_EQ(request.failover.slo.effective_tpot_ms, 100);
}

}  // namespace
}  // namespace xllm_service
