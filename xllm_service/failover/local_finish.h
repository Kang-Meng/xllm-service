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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "chat.pb.h"
#include "completion.pb.h"
#include "failover/replay.h"
#include "request/request.h"

namespace xllm_service {

inline int64_t TotalGeneratedTokensAfterReplay(const Request& request) {
  const int64_t folded_output_tokens =
      static_cast<int64_t>(request.token_ids.size()) -
      static_cast<int64_t>(request.failover.replay.base_prompt_token_count);
  const int64_t pending_output_tokens = static_cast<int64_t>(
      request.failover.replay.committed_output_token_ids.size());
  return std::max<int64_t>(0, folded_output_tokens) + pending_output_tokens;
}

inline xllm::proto::Usage BuildFailoverReplayUsage(const Request& request) {
  xllm::proto::Usage usage;
  usage.set_prompt_tokens(
      static_cast<int32_t>(request.failover.replay.base_prompt_token_count));
  usage.set_completion_tokens(
      static_cast<int32_t>(TotalGeneratedTokensAfterReplay(request)));
  usage.set_total_tokens(
      usage.prompt_tokens() + usage.completion_tokens());
  return usage;
}

inline xllm::proto::CompletionResponse BuildLocalCompletionResponse(
    const Request& request,
    const std::string& response_id,
    uint32_t created) {
  xllm::proto::CompletionResponse response;
  response.set_id(response_id);
  response.set_object("text_completion");
  response.set_created(created);
  response.set_model(request.model);
  auto* choice = response.add_choices();
  choice->set_index(0);
  choice->set_text(request.failover.replay.accumulated_output_text);
  choice->set_finish_reason("length");
  response.mutable_usage()->CopyFrom(BuildFailoverReplayUsage(request));
  return response;
}

inline xllm::proto::CompletionResponse BuildLocalCompletionFinishChunk(
    const Request& request,
    const std::string& response_id,
    uint32_t created) {
  xllm::proto::CompletionResponse response;
  response.set_id(response_id);
  response.set_object("text_completion");
  response.set_created(created);
  response.set_model(request.model);
  auto* choice = response.add_choices();
  choice->set_index(0);
  choice->set_text("");
  choice->set_finish_reason("length");
  return response;
}

inline xllm::proto::CompletionResponse BuildLocalCompletionUsageChunk(
    const Request& request,
    const std::string& response_id,
    uint32_t created) {
  xllm::proto::CompletionResponse response;
  response.set_id(response_id);
  response.set_object("text_completion");
  response.set_created(created);
  response.set_model(request.model);
  response.mutable_choices();
  response.mutable_usage()->CopyFrom(BuildFailoverReplayUsage(request));
  return response;
}

inline xllm::proto::ChatResponse BuildLocalChatResponse(
    const Request& request,
    const std::string& response_id,
    uint32_t created) {
  xllm::proto::ChatResponse response;
  response.set_id(response_id);
  response.set_object("chat.completion");
  response.set_created(created);
  response.set_model(request.model);
  auto* choice = response.add_choices();
  choice->set_index(0);
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  message->set_content(request.failover.replay.accumulated_output_text);
  choice->set_finish_reason("length");
  response.mutable_usage()->CopyFrom(BuildFailoverReplayUsage(request));
  return response;
}

inline xllm::proto::ChatResponse BuildLocalChatFinishChunk(
    const Request& request,
    const std::string& response_id,
    uint32_t created) {
  xllm::proto::ChatResponse response;
  response.set_id(response_id);
  response.set_object("chat.completion.chunk");
  response.set_created(created);
  response.set_model(request.model);
  auto* choice = response.add_choices();
  choice->set_index(0);
  choice->mutable_delta();
  choice->set_finish_reason("length");
  return response;
}

inline xllm::proto::ChatResponse BuildLocalChatUsageChunk(
    const Request& request,
    const std::string& response_id,
    uint32_t created) {
  xllm::proto::ChatResponse response;
  response.set_id(response_id);
  response.set_object("chat.completion.chunk");
  response.set_created(created);
  response.set_model(request.model);
  response.mutable_usage()->CopyFrom(BuildFailoverReplayUsage(request));
  return response;
}

template <typename RequestProto>
inline bool ShouldFinishFailoverLocally(const Request& request,
                                        const RequestProto& req_pb) {
  const auto remaining = RemainingMaxTokensAfterReplay(request, req_pb);
  return remaining.has_value() && *remaining == 0;
}

template <typename FinishRequestFn,
          typename FinishRequestContextFn,
          typename FinishWithErrorFn>
inline void CompleteLocalFailoverLifecycle(
    const Request& request,
    bool response_ok,
    FinishRequestFn&& finish_request,
    FinishRequestContextFn&& finish_request_context,
    FinishWithErrorFn&& finish_with_error) {
  if (!response_ok) {
    std::forward<FinishWithErrorFn>(finish_with_error)(
        "Internal runtime error.");
  }
  std::forward<FinishRequestFn>(finish_request)(request.service_request_id,
                                                !response_ok);
  std::forward<FinishRequestContextFn>(finish_request_context)(
      request.service_request_id);
}

}  // namespace xllm_service
