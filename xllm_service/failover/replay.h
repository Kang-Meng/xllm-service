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

#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <type_traits>

#include "chat.pb.h"
#include "completion.pb.h"
#include "request/request.h"

namespace xllm_service {

// Only used at the prefill->decode handoff to remove the T0 (and any other
// pre-handoff tokens) that decode's first streaming response repeats on top of
// what prefill already reported. Returns the length of the longest suffix of
// `committed` that matches a prefix of `new_tokens`.
inline size_t HandoffOverlapPrefixLength(
    const std::vector<int32_t>& committed,
    const std::vector<int32_t>& new_tokens) {
  if (committed.empty() || new_tokens.empty()) {
    return 0;
  }
  const size_t max_overlap = std::min(committed.size(), new_tokens.size());
  for (size_t overlap = max_overlap; overlap > 0; --overlap) {
    if (std::equal(committed.end() - static_cast<long>(overlap),
                   committed.end(),
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
    auto& buf = request->failover.replay.committed_output_token_ids;
    size_t skip = 0;
    if (request->failover.replay.pending_decode_handoff_dedup) {
      skip = HandoffOverlapPrefixLength(buf, seq_output.token_ids);
      request->failover.replay.pending_decode_handoff_dedup = false;
      // Latch: once this attempt has gone through a dedup pass, refuse to
      // re-arm even if xllm later emits another finished_on_prefill_instance=true
      // response. See FailoverReplayState::decode_handoff_dedup_consumed for
      // the failure mode this prevents.
      request->failover.replay.decode_handoff_dedup_consumed = true;
    }
    buf.insert(buf.end(),
               seq_output.token_ids.begin() + static_cast<long>(skip),
               seq_output.token_ids.end());
  }

  // Arm dedup for the next response if this one was the prefill's last emit
  // AND we have not already consumed a dedup pass in this attempt. The next
  // response is decode's first delta, which will redundantly carry the
  // tokens prefill just reported.
  if (output.finished_on_prefill_instance &&
      !request->failover.replay.decode_handoff_dedup_consumed) {
    request->failover.replay.pending_decode_handoff_dedup = true;
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

  // Sanity check: on DECODE_CRASH the replay invariant is that every token
  // we counted via update_request_metrics shows up in committed (modulo the
  // single prefill->decode handoff dedup). A larger gap usually means the
  // *upstream* xllm side skipped delivering some output token to us - the
  // known offender is incremental_decoder.cpp's checking_prefill_token_
  // branch advancing output_offset_ past a prefill-sampled token whose text
  // is empty (special / control id), combined with the next
  // generate_streaming_output returning nullopt because delta is empty,
  // which strands T0 past the slice start of every later frame. The
  // committed buffer then has every position from T1 onward, mooncake's
  // hash chain breaks at the prompt/output boundary block, and the
  // redispatched request misses every block past the original prompt.
  // We cannot recover the missing token here (it never reached us) - log
  // loud so postmortems can correlate to the upstream bug.
  if (request->failover.runtime.type == FailoverType::DECODE_CRASH) {
    const int64_t generated_tokens =
        std::max<int64_t>(0, request->num_generated_tokens);
    const int64_t committed_tokens = static_cast<int64_t>(
        request->failover.replay.committed_output_token_ids.size());
    const int64_t gap = generated_tokens - committed_tokens;
    if (gap > 1) {
      LOG(ERROR) << "failover_replay_token_gap"
                 << " request_id=" << request->service_request_id
                 << " attempt=" << request->failover.runtime.attempt
                 << " num_generated_tokens=" << generated_tokens
                 << " committed_tokens=" << committed_tokens
                 << " gap=" << gap
                 << " note=likely_xllm_incremental_decoder_dropped_empty_text_token"
                 << " effect=mooncake_prefix_hash_chain_breaks_at_prompt_boundary_block";
    }
  }

  // For PREFILL_CRASH we deliberately discard any tokens that the failed
  // prefill instance had already emitted (e.g. the T0 it sampled before
  // crashing). The new prefill will re-prefill the original prompt and
  // sample its own first token; folding the failed instance's T0 back as
  // prompt would (a) waste one token of prefill compute and (b) bias the
  // re-sampled distribution since T0 was an arbitrary draw from the dead
  // prefill that the client never observed.
  if (request->failover.runtime.type == FailoverType::PREFILL_CRASH) {
    request->failover.replay.committed_output_token_ids.clear();
    request->failover.replay.committed_output_text.clear();
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
  request->failover.replay.pending_decode_handoff_dedup = false;
  // Reset the per-attempt dedup latch so the next attempt's first
  // prefill->decode boundary will arm and consume one dedup pass again.
  request->failover.replay.decode_handoff_dedup_consumed = false;
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
