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

#include <filesystem>
#include <fstream>
#include <thread>
#include <nlohmann/json.hpp>
#include <string>

#include "failover/recovery_dump.h"

namespace xllm_service {
namespace {

TEST(FailoverRecoveryDumper, WritesOneJsonLineAsynchronously) {
  const auto dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_service_failover_recovery_dumper_test.jsonl";
  std::filesystem::remove(dump_path);

  FailoverRecoveryDumpConfig config;
  config.enabled = true;
  config.file_path = dump_path.string();

  FailoverRecoveryDumper dumper(config);

  FailoverRecoveryDumpRecord record;
  record.service_request_id = "req-1";
  record.failover_type = "DECODE_CRASH";
  record.failover_attempt = 2;
  record.origin_from_prefill = "prefill-a";
  record.origin_from_decode = "decode-a";
  record.failover_to_prefill = "prefill-b";
  record.failover_to_decode = "decode-b";
  record.prev_token_to_detected_ms = 12;
  record.detected_to_redispatch_ms = 34;
  record.dispatch_to_next_token_ms = 56;
  record.prev_token_to_next_ms = 102;
  record.restore_cost_ms = 23;
  record.recompute_cost_ms = 489;
  record.recovery_cost_ms = 512;
  record.restore_blocks = 3;
  record.recompute_tokens = 111;
  record.prompt_tokens = 257;
  record.generated_tokens = 99;
  record.original_ttft_slo_ms = 500;
  record.original_tpot_slo_ms = 100;
  record.original_ttlt_slo_ms = 60000;
  record.effective_ttft_slo_ms = 180;
  record.effective_tpot_slo_ms = 100;
  record.effective_ttlt_slo_ms = 60000;

  dumper.Dump(record);
  dumper.FlushForTest();

  std::ifstream input(dump_path);
  ASSERT_TRUE(input.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(input, line));
  ASSERT_FALSE(line.empty());
  const std::string first_line = line;
  EXPECT_FALSE(std::getline(input, line));

  const auto json = nlohmann::json::parse(first_line);
  EXPECT_EQ(json.at("request_id"), "req-1");
  EXPECT_EQ(json.size(), 3);
  const auto& origin = json.at("origin");
  const auto& failover = json.at("failover");
  EXPECT_EQ(origin.at("from_decode"), "decode-a");
  EXPECT_EQ(origin.at("prompt_tokens"), 257);
  EXPECT_EQ(origin.at("original_tpot_slo_ms"), 100);
  EXPECT_EQ(failover.at("attempt"), 2);
  EXPECT_EQ(failover.at("type"), "DECODE_CRASH");
  EXPECT_EQ(failover.at("to_decode"), "decode-b");
  EXPECT_FALSE(failover.contains("failover_detected_unix_ms"));
  EXPECT_FALSE(failover.contains("redispatched_unix_ms"));
  EXPECT_FALSE(failover.contains("first_token_unix_ms"));
  EXPECT_EQ(failover.at("prev_token_to_detected_ms"), 12);
  EXPECT_EQ(failover.at("detected_to_redispatch_ms"), 34);
  EXPECT_EQ(failover.at("dispatch_to_next_token_ms"), 56);
  EXPECT_EQ(failover.at("prev_token_to_next_ms"), 102);
  EXPECT_EQ(failover.at("restore_blocks"), 3);
  EXPECT_EQ(failover.at("recompute_tokens"), 111);
  EXPECT_EQ(failover.at("recovery_cost_ms"), 512);
  EXPECT_EQ(failover.at("effective_ttft_slo_ms"), 180);
  EXPECT_EQ(failover.at("effective_tpot_slo_ms"), 100);
  EXPECT_EQ(failover.at("effective_ttlt_slo_ms"), 60000);
  EXPECT_FALSE(failover.contains("failover_bucket"));

  std::filesystem::remove(dump_path);
}

TEST(FailoverRecoveryDumper, MakesRecordVisibleWithoutExplicitFlush) {
  const auto dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_service_failover_recovery_dumper_live_visibility_test.jsonl";
  std::filesystem::remove(dump_path);

  FailoverRecoveryDumpConfig config;
  config.enabled = true;
  config.file_path = dump_path.string();

  FailoverRecoveryDumper dumper(config);

  FailoverRecoveryDumpRecord record;
  record.service_request_id = "req-live";
  record.failover_type = "DECODE_CRASH";
  record.failover_attempt = 1;
  record.origin_from_prefill = "prefill-a";
  record.origin_from_decode = "decode-a";
  record.failover_to_prefill = "prefill-b";
  record.failover_to_decode = "decode-b";

  dumper.Dump(record);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream input(dump_path);
  ASSERT_TRUE(input.is_open());
  std::string line;
  EXPECT_TRUE(std::getline(input, line))
      << "Dumped record should be visible before dumper shutdown";
  EXPECT_FALSE(line.empty());

  std::filesystem::remove(dump_path);
}

TEST(FailoverRecoveryDumper, BuildsRecoveryDurationsFromRequestTimes) {
  Request request;
  request.service_request_id = "req-2";
  request.failover.runtime.type = FailoverType::DECODE_CRASH;
  request.failover.runtime.previous_token_time =
      absl::UnixEpoch() + absl::Milliseconds(950);
  request.failover.runtime.detected_time =
      absl::UnixEpoch() + absl::Milliseconds(1000);
  request.failover.runtime.rpc_redispatched_time =
      absl::UnixEpoch() + absl::Milliseconds(1060);
  request.failover.runtime.estimated_restore_cost_ms = 11;
  request.failover.runtime.estimated_recompute_cost_ms = 22;
  request.failover.runtime.estimated_total_recovery_cost_ms = 33;
  request.failover.runtime.estimated_restore_blocks = 7;
  request.failover.runtime.estimated_recompute_tokens = 9;
  request.failover.runtime.failed_decode_offload_batch_size = 4;
  request.token_ids.resize(128);
  request.num_generated_tokens = 10;
  request.failover.runtime.generated_tokens = 9;
  request.failover.slo.original_ttft_ms = 500;
  request.failover.slo.original_tpot_ms = 100;
  request.failover.slo.original_ttlt_ms = 6000;
  request.failover.slo.effective_ttft_ms = 180;
  request.failover.slo.effective_tpot_ms = 50;
  request.failover.slo.effective_ttlt_ms = 6000;

  FailoverFirstTokenSample sample;
  sample.attempt = 3;
  sample.from_prefill = "prefill-old";
  sample.from_decode = "decode-old";
  sample.to_prefill = "prefill-new";
  sample.to_decode = "decode-new";

  const auto record = BuildFailoverRecoveryDumpRecord(
      request,
      sample,
      absl::UnixEpoch() + absl::Milliseconds(1125));

  EXPECT_EQ(record.service_request_id, "req-2");
  EXPECT_EQ(record.failover_attempt, 3);
  EXPECT_EQ(record.failover_type, "DECODE_CRASH");
  EXPECT_EQ(record.origin_from_decode, "decode-old");
  EXPECT_EQ(record.failover_to_decode, "decode-new");
  EXPECT_EQ(record.prev_token_to_detected_ms, 50);
  EXPECT_EQ(record.detected_to_redispatch_ms, 60);
  EXPECT_EQ(record.dispatch_to_next_token_ms, 65);
  EXPECT_EQ(record.prev_token_to_next_ms, 175);
  EXPECT_EQ(record.restore_cost_ms, 11);
  EXPECT_EQ(record.recompute_cost_ms, 22);
  EXPECT_EQ(record.recovery_cost_ms, 33);
  EXPECT_EQ(record.restore_blocks, 7);
  EXPECT_EQ(record.recompute_tokens, 9);
  EXPECT_EQ(record.failed_decode_offload_batch_size, 4u);
  EXPECT_EQ(record.prompt_tokens, 128);
  EXPECT_EQ(record.generated_tokens, 9);
  EXPECT_EQ(record.original_ttft_slo_ms, 500);
  EXPECT_EQ(record.original_tpot_slo_ms, 100);
  EXPECT_EQ(record.original_ttlt_slo_ms, 6000);
  EXPECT_EQ(record.effective_ttft_slo_ms, 180);
  EXPECT_EQ(record.effective_tpot_slo_ms, 50);
  EXPECT_EQ(record.effective_ttlt_slo_ms, 6000);
}

TEST(FailoverRecoveryDumper,
     PreviousTokenDurationsAreZeroWithoutPreviousTokenTime) {
  Request request;
  request.service_request_id = "req-3";
  request.failover.runtime.type = FailoverType::DECODE_CRASH;
  request.failover.runtime.detected_time =
      absl::UnixEpoch() + absl::Milliseconds(1000);
  request.failover.runtime.rpc_redispatched_time =
      absl::UnixEpoch() + absl::Milliseconds(1060);
  request.failover.runtime.previous_token_time = absl::InfinitePast();

  FailoverFirstTokenSample sample;
  sample.attempt = 1;
  sample.from_prefill = "prefill-old";
  sample.from_decode = "decode-old";
  sample.to_prefill = "prefill-new";
  sample.to_decode = "decode-new";

  const auto record = BuildFailoverRecoveryDumpRecord(
      request,
      sample,
      absl::UnixEpoch() + absl::Milliseconds(1125));

  EXPECT_EQ(record.prev_token_to_detected_ms, 0);
  EXPECT_EQ(record.detected_to_redispatch_ms, 60);
  EXPECT_EQ(record.dispatch_to_next_token_ms, 65);
  EXPECT_EQ(record.prev_token_to_next_ms, 0);
}

}  // namespace
}  // namespace xllm_service
