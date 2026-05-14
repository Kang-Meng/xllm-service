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
#include <optional>
#include <type_traits>

#include "chat.pb.h"
#include "completion.pb.h"
#include "request/request.h"

namespace xllm_service {

inline size_t ReplayTokenOverlapPrefixLength(
    const std::vector<int32_t>& committed_tokens,
    const std::vector<int32_t>& new_tokens) {
  if (committed_tokens.empty() || new_tokens.empty()) {
    return 0;
  }

  const size_t max_overlap =
      std::min(committed_tokens.size(), new_tokens.size());
  for (size_t overlap = max_overlap; overlap > 0; --overlap) {
    if (std::equal(committed_tokens.end() - static_cast<long>(overlap),
                   committed_tokens.end(),
                   new_tokens.begin())) {
      return overlap;
    }
  }
  return 0;
}

inline void AccumulateReplayTokens(Request* request,
                                   const llm::RequestOutput& output) {
  if (request == nullptr) {
    return;
  }

  for (const auto& seq_output : output.outputs) {
    if (seq_output.index != 0 || seq_output.token_ids.empty()) {
      continue;
    }
    const size_t overlap = ReplayTokenOverlapPrefixLength(
        request->failover.replay.committed_output_token_ids, seq_output.token_ids);
    request->failover.replay.committed_output_token_ids.insert(
        request->failover.replay.committed_output_token_ids.end(),
        seq_output.token_ids.begin() + static_cast<long>(overlap),
        seq_output.token_ids.end());
  }
}

inline void AccumulateCompletionReplayState(
    Request* request,
    const llm::RequestOutput& output) {
  if (request == nullptr) {
    return;
  }

  AccumulateReplayTokens(request, output);
  for (const auto& seq_output : output.outputs) {
    if (seq_output.index != 0 || seq_output.text.empty()) {
      continue;
    }
    request->failover.replay.committed_output_text.append(seq_output.text);
    request->failover.replay.accumulated_output_text.append(seq_output.text);
  }
}

inline void AccumulateReplayState(Request* request,
                                  const llm::RequestOutput& output) {
  AccumulateCompletionReplayState(request, output);
}

inline void FoldCommittedOutputIntoRequest(Request* request) {
  if (request == nullptr) {
    return;
  }

  if (!request->failover.replay.committed_output_token_ids.empty()) {
    request->token_ids.insert(request->token_ids.end(),
                              request->failover.replay.committed_output_token_ids.begin(),
                              request->failover.replay.committed_output_token_ids.end());
  }
  if (!request->failover.replay.committed_output_text.empty()) {
    request->prompt.append(request->failover.replay.committed_output_text);
  }

  request->num_generated_tokens = 0;
  request->prefill_stage_finished = false;
}

inline void ClearCommittedReplayState(Request* request) {
  if (request == nullptr) {
    return;
  }

  request->failover.replay.committed_output_token_ids.clear();
  request->failover.replay.committed_output_text.clear();
}

template <typename RequestProto>
std::optional<uint32_t> RemainingMaxTokensAfterReplay(
    const Request& request,
    const RequestProto& req_pb) {
  if (!req_pb.has_max_tokens()) {
    return std::nullopt;
  }

  const int64_t remaining =
      static_cast<int64_t>(req_pb.max_tokens()) -
      static_cast<int64_t>(request.failover.replay.committed_output_token_ids.size());
  return static_cast<uint32_t>(std::max<int64_t>(0, remaining));
}

template <typename RequestProto>
void RewriteFailoverReplayToProto(const Request& request, RequestProto* req_pb) {
  if (req_pb == nullptr) {
    return;
  }

  req_pb->clear_token_ids();
  req_pb->mutable_token_ids()->Add(request.token_ids.begin(),
                                   request.token_ids.end());

  if constexpr (std::is_same_v<RequestProto, xllm::proto::CompletionRequest>) {
    req_pb->set_prompt(request.prompt);
  }

  if (const auto remaining = RemainingMaxTokensAfterReplay(request, *req_pb);
      remaining.has_value()) {
    req_pb->set_max_tokens(*remaining);
  }
}

}  // namespace xllm_service
