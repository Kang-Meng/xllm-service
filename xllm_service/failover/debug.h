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

#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "chat.pb.h"
#include "completion.pb.h"
#include "common/xllm/output.h"
#include "request/request.h"

namespace xllm_service {

template <typename Iterator>
inline std::string FormatTokenPreviewFromIterators(Iterator begin,
                                                   Iterator end,
                                                   size_t limit = 4) {
  const size_t size = static_cast<size_t>(std::distance(begin, end));
  std::ostringstream oss;
  oss << "[";
  if (size == 0) {
    oss << "]";
    return oss.str();
  }

  const size_t head_count = std::min(size, limit);
  Iterator it = begin;
  for (size_t i = 0; i < head_count; ++i, ++it) {
    if (i != 0) {
      oss << ",";
    }
    oss << *it;
  }

  if (size > limit * 2) {
    oss << ",...,";
    Iterator tail = end;
    std::advance(tail, -static_cast<long>(limit));
    bool first_tail = true;
    for (; tail != end; ++tail) {
      if (!first_tail) {
        oss << ",";
      }
      oss << *tail;
      first_tail = false;
    }
  } else {
    for (; it != end; ++it) {
      oss << ",";
      oss << *it;
    }
  }

  oss << "]";
  return oss.str();
}

template <typename Container>
inline std::string FormatTokenPreview(const Container& container,
                                      size_t limit = 4) {
  return FormatTokenPreviewFromIterators(container.begin(),
                                         container.end(),
                                         limit);
}

inline std::string FormatAllTokens(const std::vector<int32_t>& tokens) {
  return FormatTokenPreview(tokens, std::max<size_t>(tokens.size(), 1));
}

inline std::string FormatAccumulatedRequestTokens(
    const Request& request,
    size_t committed_output_token_count) {
  std::ostringstream oss;
  oss << "[";
  bool first = true;
  for (const auto token_id : request.token_ids) {
    if (!first) {
      oss << ",";
    }
    oss << token_id;
    first = false;
  }

  const size_t committed_count =
      std::min(committed_output_token_count,
               request.failover.replay.committed_output_token_ids.size());
  for (size_t i = 0; i < committed_count; ++i) {
    if (!first) {
      oss << ",";
    }
    oss << request.failover.replay.committed_output_token_ids[i];
    first = false;
  }
  oss << "]";
  return oss.str();
}

inline std::vector<std::string> BuildOutputTokenTraceLogs(
    const Request& request,
    const llm::RequestOutput& output,
    size_t committed_token_count_before) {
  std::vector<std::string> logs;
  size_t emitted_token_count = 0;
  for (const auto& seq_output : output.outputs) {
    if (seq_output.index != 0 || seq_output.token_ids.empty()) {
      continue;
    }

    logs.reserve(logs.size() + seq_output.token_ids.size());
    for (size_t i = 0; i < seq_output.token_ids.size(); ++i) {
      ++emitted_token_count;
      const size_t committed_token_count_now =
          std::min(request.failover.replay.committed_output_token_ids.size(),
                   committed_token_count_before + emitted_token_count);

      std::ostringstream oss;
      oss << "service_request_id=" << request.service_request_id
          << " request_id=" << output.request_id
          << " attempt=" << request.failover.runtime.attempt
          << " seq_index=" << seq_output.index
          << " emitted_token_ordinal=" << (emitted_token_count - 1)
          << " emitted_token_id=" << seq_output.token_ids[i]
          << " prompt_tokens=" << request.token_ids.size()
          << " committed_tokens=" << committed_token_count_now
          << " full_request_tokens_size="
          << (request.token_ids.size() + committed_token_count_now)
          << " full_request_tokens="
          << FormatAccumulatedRequestTokens(request, committed_token_count_now);
      if (seq_output.finish_reason.has_value()) {
        oss << " finish_reason=" << seq_output.finish_reason.value();
      }
      logs.emplace_back(oss.str());
    }
  }
  return logs;
}

inline std::string BuildFailoverReplayStateLog(const Request& request) {
  std::ostringstream oss;
  oss << "request_id=" << request.service_request_id
      << " attempt=" << request.failover.runtime.attempt
      << " base_prompt_tokens=" << request.failover.replay.base_prompt_token_count
      << " prompt_tokens=" << request.token_ids.size()
      << " replay_tokens_pending=" << request.failover.replay.committed_output_token_ids.size()
      << " replay_text_pending=" << request.failover.replay.committed_output_text.size()
      << " num_generated_tokens=" << request.num_generated_tokens
      << " prefill_finished=" << request.prefill_stage_finished
      << " token_ids=" << FormatTokenPreview(request.token_ids)
      << " replay_token_ids="
      << FormatTokenPreview(request.failover.replay.committed_output_token_ids);
  return oss.str();
}

template <typename RequestProto>
inline std::string BuildRequestProtoSummary(const RequestProto& req_pb) {
  std::ostringstream oss;
  if constexpr (std::is_same_v<RequestProto, xllm::proto::CompletionRequest>) {
    oss << "type=completion";
  } else if constexpr (std::is_same_v<RequestProto, xllm::proto::ChatRequest>) {
    oss << "type=chat";
  } else {
    oss << "type=unknown";
  }

  oss << " service_request_id=" << req_pb.service_request_id()
      << " token_ids_size=" << req_pb.token_ids_size()
      << " token_ids=" << FormatTokenPreview(req_pb.token_ids());

  if constexpr (std::is_same_v<RequestProto, xllm::proto::CompletionRequest>) {
    oss << " prompt_size=" << req_pb.prompt().size();
  } else if constexpr (std::is_same_v<RequestProto, xllm::proto::ChatRequest>) {
    oss << " messages_size=" << req_pb.messages_size();
  }

  if (req_pb.has_routing()) {
    oss << " routing.prefill=" << req_pb.routing().prefill_name()
        << " routing.decode=" << req_pb.routing().decode_name();
  } else {
    oss << " routing=<unset>";
  }
  return oss.str();
}

inline std::string BuildRequestProtoJson(
    const google::protobuf::Message& request_proto) {
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = false;
  options.preserve_proto_field_names = false;

  std::string json;
  const auto status = google::protobuf::util::MessageToJsonString(
      request_proto, &json, options);
  if (!status.ok()) {
    return "MessageToJsonString failed: " + status.ToString();
  }
  return json;
}

}  // namespace xllm_service
