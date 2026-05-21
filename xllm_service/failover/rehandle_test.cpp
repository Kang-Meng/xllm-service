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

#include "common/metrics.h"
#include "failover/debug.h"
#include "failover/local_finish.h"
#include "failover/rehandle.h"
#include "failover/replay.h"

namespace xllm_service {
namespace {

TEST(FailoverRehandle, SuccessfulRehandleArmsTimingUpdatesProtoAndDispatches) {
  Request request;
  request.service_request_id = "req-1";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";
  request.failover.runtime.type = FailoverType::PREFILL_CRASH;
  request.failover.runtime.estimated_total_recovery_cost_ms = 512;
  request.failover.slo.effective_ttft_ms = 750;
  request.failover.slo.effective_tpot_ms = 40;
  request.failover.slo.effective_ttlt_ms = 1500;

  std::string trace_message;
  request.trace_callback = [&](const std::string& message) {
    trace_message = message;
  };

  xllm::proto::CompletionRequest req_pb;
  const double rehandle_count_before =
      COUNTER_failover_rehandle_total.get_value();
  int dispatch_count = 0;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        request.routing.prefill_name = "prefill-b";
        request.routing.decode_name = "decode-b";
        return true;
      },
      [&]() { ++dispatch_count; });

  EXPECT_TRUE(ok);
  EXPECT_EQ(request.failover.runtime.attempt, 1);
  EXPECT_TRUE(request.failover.runtime.awaiting_first_token);
  EXPECT_EQ(request.failover.runtime.last_from_prefill, "prefill-a");
  EXPECT_EQ(request.failover.runtime.last_from_decode, "decode-a");
  EXPECT_EQ(request.failover.runtime.latest_start_time,
            absl::UnixEpoch() + absl::Milliseconds(1000));
  EXPECT_GE(request.failover.runtime.redispatched_time,
            absl::UnixEpoch() + absl::Milliseconds(1000));
  EXPECT_EQ(request.failover.runtime.rpc_redispatched_time, absl::InfinitePast());
  EXPECT_EQ(req_pb.routing().prefill_name(), "prefill-b");
  EXPECT_EQ(req_pb.routing().decode_name(), "decode-b");
  EXPECT_EQ(req_pb.ttft_slo_ms(), 750);
  EXPECT_EQ(req_pb.tpot_slo_ms(), 40);
  EXPECT_EQ(req_pb.ttlt_slo_ms(), 1500);
  EXPECT_EQ(dispatch_count, 1);
  EXPECT_DOUBLE_EQ(COUNTER_failover_rehandle_total.get_value(),
                   rehandle_count_before + 1);
  EXPECT_NE(trace_message.find("failover_rehandle_start"), std::string::npos);
  EXPECT_NE(trace_message.find("from_prefill=prefill-a"), std::string::npos);
  EXPECT_NE(trace_message.find("failover_type=PREFILL_CRASH"),
            std::string::npos);
  EXPECT_NE(trace_message.find("recovery_cost_ms=512"), std::string::npos);
  EXPECT_EQ(trace_message.find("bucket="), std::string::npos);
}

TEST(FailoverRehandle, LeavesUnsetFailoverSloAbsentFromProto) {
  Request request;
  request.service_request_id = "req-1b";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";
  request.failover.slo.effective_ttft_ms = kUnsetSloMs;
  request.failover.slo.effective_tpot_ms = 25;
  request.failover.slo.effective_ttlt_ms = kUnsetSloMs;

  xllm::proto::ChatRequest req_pb;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        request.routing.prefill_name = "prefill-b";
        request.routing.decode_name = "decode-b";
        return true;
      },
      [&]() {});

  EXPECT_TRUE(ok);
  EXPECT_FALSE(req_pb.has_ttft_slo_ms());
  EXPECT_EQ(req_pb.tpot_slo_ms(), 25);
  EXPECT_FALSE(req_pb.has_ttlt_slo_ms());
}

TEST(FailoverRehandle, DecodeCrashPreservesPrefillWhenRescheduling) {
  Request request;
  request.service_request_id = "req-decode-crash";
  request.routing.prefill_name = "prefill-original";
  request.routing.decode_name = "decode-failed";
  request.failover.runtime.type = FailoverType::DECODE_CRASH;

  xllm::proto::CompletionRequest req_pb;
  bool schedule_called = false;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        schedule_called = true;
        EXPECT_EQ(request.routing.prefill_name, "prefill-original");
        EXPECT_TRUE(request.routing.decode_name.empty());
        request.routing.decode_name = "decode-new";
        return true;
      },
      [&]() {});

  EXPECT_TRUE(ok);
  EXPECT_TRUE(schedule_called);
  EXPECT_EQ(request.failover.runtime.last_from_prefill, "prefill-original");
  EXPECT_EQ(request.failover.runtime.last_from_decode, "decode-failed");
  EXPECT_EQ(req_pb.routing().prefill_name(), "prefill-original");
  EXPECT_EQ(req_pb.routing().decode_name(), "decode-new");
}

TEST(FailoverRehandle, PlannedPrefillIsAppliedAfterCapturingOriginalRoute) {
  Request request;
  request.service_request_id = "req-planned-prefill";
  request.routing.prefill_name = "prefill-original";
  request.routing.decode_name = "decode-original";
  request.failover.runtime.type = FailoverType::PREFILL_CRASH;
  request.failover.runtime.planned_prefill_name = "prefill-planned";

  xllm::proto::CompletionRequest req_pb;
  bool schedule_called = false;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        schedule_called = true;
        EXPECT_EQ(request.failover.runtime.last_from_prefill,
                  "prefill-original");
        EXPECT_EQ(request.failover.runtime.last_from_decode,
                  "decode-original");
        EXPECT_EQ(request.routing.prefill_name, "prefill-planned");
        EXPECT_TRUE(request.routing.decode_name.empty());
        request.routing.decode_name = "decode-new";
        return true;
      },
      [&]() {});

  EXPECT_TRUE(ok);
  EXPECT_TRUE(schedule_called);
  EXPECT_TRUE(request.failover.runtime.planned_prefill_name.empty());
  EXPECT_EQ(req_pb.routing().prefill_name(), "prefill-planned");
  EXPECT_EQ(req_pb.routing().decode_name(), "decode-new");
}

TEST(FailoverRehandle, WritesDecodeRecoveryUsingTtftAndKeepsTpot) {
  Request request;
  request.service_request_id = "req-1d";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";
  request.failover.slo.effective_ttft_ms = 180;
  request.failover.slo.effective_tpot_ms = 100;
  request.failover.slo.effective_ttlt_ms = 25000;

  xllm::proto::CompletionRequest req_pb;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        request.routing.prefill_name = "prefill-b";
        request.routing.decode_name = "decode-b";
        return true;
      },
      [&]() {});

  EXPECT_TRUE(ok);
  EXPECT_EQ(req_pb.ttft_slo_ms(), 180);
  EXPECT_EQ(req_pb.tpot_slo_ms(), 100);
  EXPECT_EQ(req_pb.ttlt_slo_ms(), 25000);
}

TEST(FailoverRehandle, RecordsSchedulingRedispatchTimeBeforeDispatch) {
  Request request;
  request.service_request_id = "req-1c";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";

  xllm::proto::CompletionRequest req_pb;
  bool saw_schedule_timestamp = false;
  bool saw_rpc_timestamp = false;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        request.routing.prefill_name = "prefill-b";
        request.routing.decode_name = "decode-b";
        return true;
      },
      [&]() {
        saw_schedule_timestamp =
            request.failover.runtime.redispatched_time != absl::InfinitePast();
        saw_rpc_timestamp =
            request.failover.runtime.rpc_redispatched_time == absl::InfinitePast();
      });

  EXPECT_TRUE(ok);
  EXPECT_TRUE(saw_schedule_timestamp);
  EXPECT_TRUE(saw_rpc_timestamp);
}

TEST(FailoverRehandle, EmptyPrefillNameAfterScheduleIsRejected) {
  // schedule_failover should never return success with an empty prefill_name,
  // but if it does (or some other code path zeroes the field), we must NOT
  // ship a proto with has_routing()==true && prefill_name=="" downstream:
  // on the xllm side that can fall through to re-rendering messages and
  // re-tokenizing, which silently changes the token prefix and breaks the
  // mooncake prefix-hash chain from block 0 (total KV-cache miss).
  Request request;
  request.service_request_id = "req-empty-prefill";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";
  request.failover.runtime.type = FailoverType::DECODE_CRASH;

  xllm::proto::CompletionRequest req_pb;
  int dispatch_count = 0;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(1000),
      [&]() {
        // Pathological schedule_failover: returns success but leaves
        // routing.prefill_name empty.
        request.routing.prefill_name.clear();
        request.routing.decode_name = "decode-new";
        return true;
      },
      [&]() { ++dispatch_count; });

  EXPECT_FALSE(ok);
  EXPECT_EQ(dispatch_count, 0);
  EXPECT_TRUE(req_pb.routing().prefill_name().empty());
}

TEST(FailoverRehandle, FailedRescheduleDoesNotDispatch) {
  Request request;
  request.service_request_id = "req-2";
  request.routing.prefill_name = "prefill-a";
  request.routing.decode_name = "decode-a";

  xllm::proto::ChatRequest req_pb;
  const double rehandle_count_before =
      COUNTER_failover_rehandle_total.get_value();
  int dispatch_count = 0;

  const bool ok = RehandleScheduledRequest(
      &request,
      &req_pb,
      absl::UnixEpoch() + absl::Milliseconds(2000),
      [&]() { return false; },
      [&]() { ++dispatch_count; });

  EXPECT_FALSE(ok);
  EXPECT_EQ(dispatch_count, 0);
  EXPECT_TRUE(request.failover.runtime.awaiting_first_token);
  EXPECT_EQ(req_pb.routing().prefill_name(), "");
  EXPECT_EQ(req_pb.routing().decode_name(), "");
  EXPECT_DOUBLE_EQ(COUNTER_failover_rehandle_total.get_value(),
                   rehandle_count_before + 1);
}

TEST(FailoverRehandle, RewriteCompletionProtoCarriesReplayTokensAndPrompt) {
  Request request;
  request.prompt = "hello";
  request.token_ids = {10, 11};

  xllm::proto::CompletionRequest req_pb;
  req_pb.set_prompt("hello");
  req_pb.set_max_tokens(5);
  req_pb.mutable_token_ids()->Add(10);
  req_pb.mutable_token_ids()->Add(11);

  request.failover.replay.committed_output_text = " world";
  request.failover.replay.committed_output_token_ids = {12};
  request.failover.replay.accumulated_output_text = " world";

  FoldCommittedOutputIntoRequest(&request);
  RewriteFailoverReplayToProto(request, &req_pb);
  ClearCommittedReplayState(&request);

  ASSERT_EQ(request.token_ids.size(), 3);
  EXPECT_EQ(request.prompt, "hello world");
  EXPECT_EQ(request.num_generated_tokens, 0);
  EXPECT_FALSE(request.prefill_stage_finished);
  EXPECT_TRUE(request.failover.replay.committed_output_text.empty());
  EXPECT_TRUE(request.failover.replay.committed_output_token_ids.empty());
  EXPECT_EQ(request.failover.replay.accumulated_output_text, " world");

  EXPECT_EQ(req_pb.prompt(), "hello world");
  ASSERT_TRUE(req_pb.has_max_tokens());
  EXPECT_EQ(req_pb.max_tokens(), 4);
  ASSERT_EQ(req_pb.token_ids_size(), 3);
  EXPECT_EQ(req_pb.token_ids(0), 10);
  EXPECT_EQ(req_pb.token_ids(1), 11);
  EXPECT_EQ(req_pb.token_ids(2), 12);
}

TEST(FailoverRehandle, RewriteChatProtoCarriesReplayTokensOnly) {
  Request request;
  request.token_ids = {101, 102};
  request.messages.emplace_back("user", "hello");
  request.failover.replay.committed_output_token_ids = {103, 104};

  xllm::proto::ChatRequest req_pb;
  auto* msg = req_pb.add_messages();
  msg->set_role("user");
  msg->set_content("hello");
  req_pb.set_max_tokens(6);
  req_pb.mutable_token_ids()->Add(1);
  req_pb.mutable_token_ids()->Add(2);
  req_pb.mutable_token_ids()->Add(3);

  FoldCommittedOutputIntoRequest(&request);
  RewriteFailoverReplayToProto(request, &req_pb);
  ClearCommittedReplayState(&request);

  ASSERT_EQ(request.token_ids.size(), 4);
  EXPECT_TRUE(request.failover.replay.committed_output_token_ids.empty());
  ASSERT_EQ(req_pb.messages_size(), 1);
  EXPECT_EQ(req_pb.messages(0).role(), "user");
  EXPECT_EQ(req_pb.messages(0).content(), "hello");
  ASSERT_TRUE(req_pb.has_max_tokens());
  EXPECT_EQ(req_pb.max_tokens(), 4);
  ASSERT_EQ(req_pb.token_ids_size(), 4);
  EXPECT_EQ(req_pb.token_ids(0), 101);
  EXPECT_EQ(req_pb.token_ids(1), 102);
  EXPECT_EQ(req_pb.token_ids(2), 103);
  EXPECT_EQ(req_pb.token_ids(3), 104);
}

TEST(FailoverRehandle, RemainingMaxTokensAfterReplayClampsAtZero) {
  Request request;
  request.failover.replay.committed_output_token_ids = {10, 11, 12};

  xllm::proto::CompletionRequest req_pb;
  req_pb.set_max_tokens(2);

  const auto remaining = RemainingMaxTokensAfterReplay(request, req_pb);

  ASSERT_TRUE(remaining.has_value());
  EXPECT_EQ(*remaining, 0);
  EXPECT_TRUE(ShouldFinishFailoverLocally(request, req_pb));
}

TEST(FailoverRehandle, LocalCompletionResponseUsesAccumulatedOutputAndUsage) {
  Request request;
  request.model = "model-a";
  request.token_ids = {1, 2, 3, 4};
  request.failover.replay.base_prompt_token_count = 2;
  request.failover.replay.committed_output_token_ids = {5, 6};
  request.failover.replay.accumulated_output_text = "done";

  const auto response =
      BuildLocalCompletionResponse(request, "svc-1", 1234);

  EXPECT_EQ(response.id(), "svc-1");
  EXPECT_EQ(response.object(), "text_completion");
  EXPECT_EQ(response.created(), 1234u);
  EXPECT_EQ(response.model(), "model-a");
  ASSERT_EQ(response.choices_size(), 1);
  EXPECT_EQ(response.choices(0).text(), "done");
  EXPECT_EQ(response.choices(0).finish_reason(), "length");
  EXPECT_EQ(response.usage().prompt_tokens(), 2);
  EXPECT_EQ(response.usage().completion_tokens(), 4);
  EXPECT_EQ(response.usage().total_tokens(), 6);
}

TEST(FailoverRehandle, LocalChatFinishChunkUsesLengthReason) {
  Request request;
  request.model = "model-b";

  const auto response = BuildLocalChatFinishChunk(request, "svc-2", 5678);

  EXPECT_EQ(response.id(), "svc-2");
  EXPECT_EQ(response.object(), "chat.completion.chunk");
  EXPECT_EQ(response.created(), 5678u);
  EXPECT_EQ(response.model(), "model-b");
  ASSERT_EQ(response.choices_size(), 1);
  EXPECT_EQ(response.choices(0).finish_reason(), "length");
  EXPECT_TRUE(response.choices(0).has_delta());
}

TEST(FailoverRehandle,
     CompleteLocalFailoverLifecycleFinishesSchedulerStateOnSuccess) {
  Request request;
  request.service_request_id = "svc-local-success";
  bool finish_request_called = false;
  bool finish_request_error = true;
  bool finish_request_context_called = false;
  bool finish_with_error_called = false;

  CompleteLocalFailoverLifecycle(
      request,
      /*response_ok=*/true,
      [&](const std::string& service_request_id, bool error) {
        finish_request_called = true;
        finish_request_error = error;
        EXPECT_EQ(service_request_id, "svc-local-success");
      },
      [&](const std::string& service_request_id) {
        finish_request_context_called = true;
        EXPECT_EQ(service_request_id, "svc-local-success");
      },
      [&](const std::string& error_message) {
        finish_with_error_called = true;
        EXPECT_EQ(error_message, "Internal runtime error.");
      });

  EXPECT_TRUE(finish_request_called);
  EXPECT_FALSE(finish_request_error);
  EXPECT_TRUE(finish_request_context_called);
  EXPECT_FALSE(finish_with_error_called);
}

TEST(FailoverRehandle, CompleteFailedRequestLifecycleClosesSchedulerContext) {
  std::string finished_request_id;
  bool finish_request_error = false;
  std::string finished_context_id;
  std::string error_message;

  CompleteFailedRequestLifecycle(
      "req-error",
      "Schedule request failed!",
      [&](const std::string& service_request_id, bool error) {
        finished_request_id = service_request_id;
        finish_request_error = error;
      },
      [&](const std::string& service_request_id) {
        finished_context_id = service_request_id;
      },
      [&](const std::string& message) { error_message = message; });

  EXPECT_EQ(finished_request_id, "req-error");
  EXPECT_TRUE(finish_request_error);
  EXPECT_EQ(finished_context_id, "req-error");
  EXPECT_EQ(error_message, "Schedule request failed!");
}

TEST(FailoverRehandle, AccumulateReplayTokensTracksPrimarySequenceOnly) {
  Request request;
  llm::RequestOutput output;
  output.outputs.push_back(
      llm::SequenceOutput{.index = 0, .text = "ab", .token_ids = {7, 8}});
  output.outputs.push_back(
      llm::SequenceOutput{.index = 1, .text = "skip", .token_ids = {9}});

  AccumulateCompletionReplayState(&request, output);

  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 2);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[0], 7);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[1], 8);
  EXPECT_EQ(request.failover.replay.committed_output_text, "ab");
  EXPECT_EQ(request.failover.replay.accumulated_output_text, "ab");
}

TEST(FailoverRehandle, AccumulateReplayTokensAppendsDeltaVerbatim) {
  Request request;
  request.failover.replay.committed_output_token_ids = {41986};

  llm::RequestOutput output;
  output.outputs.push_back(llm::SequenceOutput{
      .index = 0,
      .text = "continued",
      .token_ids = {41986, 313, 17624}});

  AccumulateReplayTokens(&request, output);

  // Delta tokens are appended verbatim; the streaming layer guarantees deltas
  // are non-overlapping, so we no longer dedup on apparent prefix overlap
  // (which would silently drop real repeated tokens).
  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 4);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[0], 41986);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[1], 41986);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[2], 313);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[3], 17624);
}

TEST(FailoverRehandle, AccumulateReplayTokensKeepsRealRepeatedTokens) {
  Request request;
  request.failover.replay.committed_output_token_ids = {100, 200};

  llm::RequestOutput output;
  output.outputs.push_back(llm::SequenceOutput{
      .index = 0,
      .text = "repeat",
      .token_ids = {200, 200, 300}});

  AccumulateReplayTokens(&request, output);

  // Outside of the prefill->decode handoff window, deltas are appended
  // verbatim. The leading 200 in the delta is a genuine repeated token (not a
  // resend), so all five tokens must be preserved.
  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 5);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[0], 100);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[1], 200);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[2], 200);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[3], 200);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[4], 300);
}

TEST(FailoverRehandle, FinishedOnPrefillArmsHandoffDedup) {
  Request request;

  llm::RequestOutput prefill_output;
  prefill_output.finished_on_prefill_instance = true;
  prefill_output.outputs.push_back(llm::SequenceOutput{
      .index = 0, .text = "abc", .token_ids = {7}});

  AccumulateReplayTokens(&request, prefill_output);

  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 1);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[0], 7);
  EXPECT_TRUE(request.failover.replay.pending_decode_handoff_dedup);
}

TEST(FailoverRehandle, HandoffDedupSkipsRepeatedPrefillToken) {
  Request request;
  request.failover.replay.committed_output_token_ids = {7};
  request.failover.replay.pending_decode_handoff_dedup = true;

  llm::RequestOutput decode_first_output;
  decode_first_output.finished_on_prefill_instance = false;
  decode_first_output.outputs.push_back(llm::SequenceOutput{
      .index = 0, .text = "de", .token_ids = {7, 8}});

  AccumulateReplayTokens(&request, decode_first_output);

  // The leading 7 from decode is the prefill-sampled token T0 that prefill
  // already reported; it must be skipped exactly once at the handoff.
  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 2);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[0], 7);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[1], 8);
  EXPECT_FALSE(request.failover.replay.pending_decode_handoff_dedup);
}

TEST(FailoverRehandle, FoldDiscardsCommittedOutputOnPrefillCrash) {
  Request request;
  request.failover.runtime.type = FailoverType::PREFILL_CRASH;
  request.failover.replay.base_prompt_token_count = 3;
  request.token_ids = {1, 2, 3};
  request.prompt = "hello";
  request.failover.replay.committed_output_token_ids = {99};
  request.failover.replay.committed_output_text = "x";

  FoldCommittedOutputIntoRequest(&request);

  // For PREFILL_CRASH the committed output (T0 that the dead prefill emitted
  // before crashing) must NOT be appended back into the next prefill input;
  // the new prefill will resample its own first token.
  ASSERT_EQ(request.token_ids.size(), 3u);
  EXPECT_EQ(request.token_ids[2], 3);
  EXPECT_EQ(request.prompt, "hello");
  EXPECT_TRUE(request.failover.replay.committed_output_token_ids.empty());
  EXPECT_TRUE(request.failover.replay.committed_output_text.empty());
}

TEST(FailoverRehandle, FoldAppendsCommittedOutputOnDecodeCrash) {
  Request request;
  request.failover.runtime.type = FailoverType::DECODE_CRASH;
  request.failover.replay.base_prompt_token_count = 3;
  request.token_ids = {1, 2, 3};
  request.failover.replay.committed_output_token_ids = {99, 100};

  FoldCommittedOutputIntoRequest(&request);

  ASSERT_EQ(request.token_ids.size(), 5u);
  EXPECT_EQ(request.token_ids[3], 99);
  EXPECT_EQ(request.token_ids[4], 100);
}

TEST(FailoverRehandle, HandoffDedupFiresOnlyOnce) {
  Request request;
  request.failover.replay.committed_output_token_ids = {7};
  request.failover.replay.pending_decode_handoff_dedup = true;

  llm::RequestOutput first;
  first.outputs.push_back(llm::SequenceOutput{
      .index = 0, .text = "de", .token_ids = {7, 8}});
  AccumulateReplayTokens(&request, first);

  // Subsequent decode deltas must be appended verbatim, even if a coincidental
  // tail/head match exists. 8 here is a real token, not a duplicate.
  llm::RequestOutput second;
  second.outputs.push_back(llm::SequenceOutput{
      .index = 0, .text = "fg", .token_ids = {8, 9}});
  AccumulateReplayTokens(&request, second);

  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 4);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[0], 7);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[1], 8);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[2], 8);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[3], 9);
}

TEST(FailoverRehandle, HandoffDedupLatchIgnoresLaterPrefillFinishedFlag) {
  // Regression for the Risk-C path: xllm may emit more than one response
  // tagged finished_on_prefill_instance=true (the streaming batch path sets
  // the flag on both per-output and sequence-level branches). Once we've
  // already consumed a dedup pass for this attempt, a later such response
  // must NOT re-arm pending_decode_handoff_dedup, or the next decode delta
  // whose head happens to equal the tail of committed would be silently
  // truncated by one token and every downstream KV-cache block hash would
  // shift by one position.
  Request request;

  // Step 1: prefill emits T0 with finished_on_prefill_instance=true.
  llm::RequestOutput prefill_first;
  prefill_first.finished_on_prefill_instance = true;
  prefill_first.outputs.push_back(
      llm::SequenceOutput{.index = 0, .text = "T0", .token_ids = {7}});
  AccumulateReplayTokens(&request, prefill_first);
  ASSERT_TRUE(request.failover.replay.pending_decode_handoff_dedup);
  ASSERT_FALSE(request.failover.replay.decode_handoff_dedup_consumed);

  // Step 2: decode emits the first delta repeating T0; dedup fires once.
  llm::RequestOutput decode_first;
  decode_first.outputs.push_back(
      llm::SequenceOutput{.index = 0, .text = "T1", .token_ids = {7, 8}});
  AccumulateReplayTokens(&request, decode_first);
  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 2u);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[1], 8);
  EXPECT_FALSE(request.failover.replay.pending_decode_handoff_dedup);
  EXPECT_TRUE(request.failover.replay.decode_handoff_dedup_consumed);

  // Step 3: anomalous "prefill done again" tail frame - empty outputs but
  // finished_on_prefill_instance=true. With the latch, this must NOT re-arm.
  llm::RequestOutput prefill_tail;
  prefill_tail.finished_on_prefill_instance = true;
  AccumulateReplayTokens(&request, prefill_tail);
  EXPECT_FALSE(request.failover.replay.pending_decode_handoff_dedup);
  EXPECT_TRUE(request.failover.replay.decode_handoff_dedup_consumed);

  // Step 4: next decode delta whose first token equals committed's tail
  // (real repeated 8). Must be appended verbatim, NOT deduped.
  llm::RequestOutput decode_second;
  decode_second.outputs.push_back(
      llm::SequenceOutput{.index = 0, .text = "T2", .token_ids = {8, 9}});
  AccumulateReplayTokens(&request, decode_second);

  ASSERT_EQ(request.failover.replay.committed_output_token_ids.size(), 4u);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[0], 7);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[1], 8);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[2], 8);
  EXPECT_EQ(request.failover.replay.committed_output_token_ids[3], 9);
}

TEST(FailoverRehandle, ClearCommittedReplayResetsDedupLatch) {
  // After rehandle clears replay state, a fresh attempt's first prefill-done
  // response must be able to arm dedup again.
  Request request;
  request.failover.replay.committed_output_token_ids = {7, 8};
  request.failover.replay.pending_decode_handoff_dedup = false;
  request.failover.replay.decode_handoff_dedup_consumed = true;

  ClearCommittedReplayState(&request);

  EXPECT_TRUE(request.failover.replay.committed_output_token_ids.empty());
  EXPECT_FALSE(request.failover.replay.pending_decode_handoff_dedup);
  EXPECT_FALSE(request.failover.replay.decode_handoff_dedup_consumed);

  llm::RequestOutput prefill_first;
  prefill_first.finished_on_prefill_instance = true;
  prefill_first.outputs.push_back(
      llm::SequenceOutput{.index = 0, .text = "T0", .token_ids = {11}});
  AccumulateReplayTokens(&request, prefill_first);

  EXPECT_TRUE(request.failover.replay.pending_decode_handoff_dedup);
  EXPECT_FALSE(request.failover.replay.decode_handoff_dedup_consumed);
}

TEST(FailoverRehandle, ReplayStateLogCapturesBasePromptAndPendingReplay) {
  Request request;
  request.service_request_id = "req-log";
  request.failover.runtime.attempt = 2;
  request.failover.replay.base_prompt_token_count = 512;
  request.token_ids = {10, 11, 12, 13, 14, 15};
  request.failover.replay.committed_output_token_ids = {16, 17, 18};
  request.failover.replay.committed_output_text = "abc";
  request.num_generated_tokens = 19;
  request.prefill_stage_finished = true;

  const std::string log_line = BuildFailoverReplayStateLog(request);

  EXPECT_NE(log_line.find("request_id=req-log"), std::string::npos);
  EXPECT_NE(log_line.find("attempt=2"), std::string::npos);
  EXPECT_NE(log_line.find("base_prompt_tokens=512"), std::string::npos);
  EXPECT_NE(log_line.find("prompt_tokens=6"), std::string::npos);
  EXPECT_NE(log_line.find("replay_tokens_pending=3"), std::string::npos);
  EXPECT_NE(log_line.find("replay_text_pending=3"), std::string::npos);
  EXPECT_NE(log_line.find("num_generated_tokens=19"), std::string::npos);
  EXPECT_NE(log_line.find("prefill_finished=1"), std::string::npos);
}

TEST(FailoverRehandle, RequestProtoSummaryShowsReplayTokenCountAndRouting) {
  xllm::proto::CompletionRequest req_pb;
  req_pb.set_service_request_id("req-proto");
  req_pb.set_prompt("hello world");
  req_pb.mutable_token_ids()->Add(100);
  req_pb.mutable_token_ids()->Add(101);
  req_pb.mutable_token_ids()->Add(102);
  req_pb.mutable_routing()->set_prefill_name("prefill-x");
  req_pb.mutable_routing()->set_decode_name("decode-y");

  const std::string summary = BuildRequestProtoSummary(req_pb);
  const std::string json = BuildRequestProtoJson(req_pb);

  EXPECT_NE(summary.find("type=completion"), std::string::npos);
  EXPECT_NE(summary.find("service_request_id=req-proto"), std::string::npos);
  EXPECT_NE(summary.find("token_ids_size=3"), std::string::npos);
  EXPECT_NE(summary.find("prompt_size=11"), std::string::npos);
  EXPECT_NE(summary.find("routing.prefill=prefill-x"), std::string::npos);
  EXPECT_NE(summary.find("routing.decode=decode-y"), std::string::npos);
  EXPECT_NE(json.find("\"serviceRequestId\":\"req-proto\""), std::string::npos);
  EXPECT_NE(json.find("\"tokenIds\":[100,101,102]"), std::string::npos);
}

TEST(FailoverRehandle, OutputTokenTraceLogsUsePromptPlusCommittedTokens) {
  Request request;
  request.service_request_id = "svc-1";
  request.token_ids = {10, 11, 12};
  request.failover.runtime.attempt = 1;
  request.failover.replay.committed_output_token_ids = {20, 21, 22, 23};

  llm::RequestOutput output;
  output.request_id = "req-1";
  output.outputs.push_back(llm::SequenceOutput{
      .index = 0,
      .text = "ab",
      .token_ids = {22, 23},
      .finish_reason = std::string("stop")});

  const auto logs = BuildOutputTokenTraceLogs(
      request, output, /*committed_token_count_before=*/2);

  ASSERT_EQ(logs.size(), 2);
  EXPECT_NE(logs[0].find("service_request_id=svc-1"), std::string::npos);
  EXPECT_NE(logs[0].find("request_id=req-1"), std::string::npos);
  EXPECT_NE(logs[0].find("emitted_token_ordinal=0"), std::string::npos);
  EXPECT_NE(logs[0].find("emitted_token_id=22"), std::string::npos);
  EXPECT_NE(logs[0].find("full_request_tokens_size=6"), std::string::npos);
  EXPECT_NE(logs[0].find("full_request_tokens=[10,11,12,20,21,22]"),
            std::string::npos);
  EXPECT_NE(logs[1].find("emitted_token_ordinal=1"), std::string::npos);
  EXPECT_NE(logs[1].find("emitted_token_id=23"), std::string::npos);
  EXPECT_NE(logs[1].find("full_request_tokens_size=7"), std::string::npos);
  EXPECT_NE(logs[1].find("full_request_tokens=[10,11,12,20,21,22,23]"),
            std::string::npos);
  EXPECT_NE(logs[1].find("finish_reason=stop"), std::string::npos);
}

}  // namespace
}  // namespace xllm_service
