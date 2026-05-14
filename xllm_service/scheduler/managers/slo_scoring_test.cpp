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
==============================================================================
*/

#include <cmath>

#include <gtest/gtest.h>

#include "scheduler/managers/slo_scoring.h"

namespace xllm_service {
namespace {

TEST(SloScoringTest, NormalizesMissingPredictionsToPositiveFallback) {
  EXPECT_EQ(NormalizeEstimatedLatencyMs(0.0, 1000), 1000);
  EXPECT_EQ(NormalizeEstimatedLatencyMs(-1.0, 1000), 1000);
  EXPECT_EQ(NormalizeEstimatedLatencyMs(std::nan(""), 1000), 1000);
  EXPECT_EQ(NormalizeEstimatedLatencyMs(0.0, 0), 1);
}

TEST(SloScoringTest, RoundsPositivePredictionsUp) {
  EXPECT_EQ(NormalizeEstimatedLatencyMs(1.1, 1000), 2);
  EXPECT_EQ(NormalizeEstimatedLatencyMs(9.0, 1000), 9);
}

TEST(SloScoringTest, PrefillTieBreaksOnCurrentLoad) {
  PrefillCandidateScore busy;
  busy.estimated_prefill_time = 100;
  busy.prefill_request_num = 2;
  busy.prefill_token_num = 1024;
  busy.order = 0;

  PrefillCandidateScore idle;
  idle.estimated_prefill_time = 100;
  idle.prefill_request_num = 1;
  idle.prefill_token_num = 2048;
  idle.order = 1;

  EXPECT_TRUE(IsBetterPrefillCandidate(idle, busy));
  EXPECT_FALSE(IsBetterPrefillCandidate(busy, idle));
}

TEST(SloScoringTest, DecodeTieBreaksOnCurrentLoad) {
  DecodeCandidateScore busy;
  busy.estimated_tpot = 50;
  busy.decode_request_num = 2;
  busy.decode_token_num = 128;
  busy.order = 0;

  DecodeCandidateScore idle;
  idle.estimated_tpot = 50;
  idle.decode_request_num = 1;
  idle.decode_token_num = 256;
  idle.order = 1;

  EXPECT_TRUE(IsBetterDecodeCandidate(idle, busy));
  EXPECT_FALSE(IsBetterDecodeCandidate(busy, idle));
}

TEST(SloScoringTest, RotatingOrderStartsAtNextIndex) {
  EXPECT_EQ(RotatingCandidateOrder(0, 0, 3), 0);
  EXPECT_EQ(RotatingCandidateOrder(1, 0, 3), 1);
  EXPECT_EQ(RotatingCandidateOrder(2, 0, 3), 2);

  EXPECT_EQ(RotatingCandidateOrder(1, 1, 3), 0);
  EXPECT_EQ(RotatingCandidateOrder(2, 1, 3), 1);
  EXPECT_EQ(RotatingCandidateOrder(0, 1, 3), 2);

  EXPECT_EQ(RotatingCandidateOrder(2, 5, 3), 0);
}

}  // namespace
}  // namespace xllm_service
