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

#include "failover/recovery_dump.h"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

namespace xllm_service {
namespace {

int64_t DurationMillis(absl::Time end, absl::Time start) {
  if (end == absl::InfinitePast() || end == absl::InfiniteFuture() ||
      start == absl::InfinitePast() || start == absl::InfiniteFuture()) {
    return 0;
  }
  return std::max<int64_t>(0, absl::ToInt64Milliseconds(end - start));
}

void SetSlo(nlohmann::json* json, const char* key, int32_t slo_ms) {
  if (slo_ms == kUnsetSloMs) {
    (*json)[key] = nullptr;
    return;
  }
  (*json)[key] = slo_ms;
}

}  // namespace

FailoverRecoveryDumpRecord BuildFailoverRecoveryDumpRecord(
    const Request& request,
    const FailoverFirstTokenSample& sample,
    absl::Time first_token_time) {
  FailoverRecoveryDumpRecord record;
  record.service_request_id = request.service_request_id;
  record.failover_attempt = sample.attempt;
  record.failover_type = FailoverTypeName(request.failover.runtime.type);
  record.origin_from_prefill = sample.from_prefill;
  record.origin_from_decode = sample.from_decode;
  record.failover_to_prefill = sample.to_prefill;
  record.failover_to_decode = sample.to_decode;
  record.prev_token_to_detected_ms =
      DurationMillis(request.failover.runtime.detected_time,
                     request.failover.runtime.previous_token_time);
  record.detected_to_redispatch_ms =
      DurationMillis(request.failover.runtime.rpc_redispatched_time,
                     request.failover.runtime.detected_time);
  record.dispatch_to_next_token_ms =
      DurationMillis(first_token_time, request.failover.runtime.rpc_redispatched_time);
  record.prev_token_to_next_ms =
      DurationMillis(first_token_time, request.failover.runtime.previous_token_time);
  record.restore_cost_ms = request.failover.runtime.estimated_restore_cost_ms;
  record.recompute_cost_ms = request.failover.runtime.estimated_recompute_cost_ms;
  record.recovery_cost_ms = request.failover.runtime.estimated_total_recovery_cost_ms;
  record.restore_blocks = request.failover.runtime.estimated_restore_blocks;
  record.recompute_tokens = request.failover.runtime.estimated_recompute_tokens;
  record.failed_decode_offload_batch_size =
      request.failover.runtime.failed_decode_offload_batch_size;
  record.prompt_tokens = static_cast<int64_t>(request.token_ids.size());
  record.generated_tokens = request.failover.runtime.generated_tokens;
  record.original_ttft_slo_ms = request.failover.slo.original_ttft_ms;
  record.original_tpot_slo_ms = request.failover.slo.original_tpot_ms;
  record.original_ttlt_slo_ms = request.failover.slo.original_ttlt_ms;
  record.effective_ttft_slo_ms = request.failover.slo.effective_ttft_ms;
  record.effective_tpot_slo_ms = request.failover.slo.effective_tpot_ms;
  record.effective_ttlt_slo_ms = request.failover.slo.effective_ttlt_ms;
  return record;
}

FailoverRecoveryDumper::FailoverRecoveryDumper(
    FailoverRecoveryDumpConfig config)
    : enabled_(config.enabled),
      max_queue_size_(std::max<size_t>(1, config.max_queue_size)) {
  if (!enabled_) {
    return;
  }

  const std::filesystem::path output_path(config.file_path);
  if (output_path.has_parent_path()) {
    std::error_code error;
    std::filesystem::create_directories(output_path.parent_path(), error);
    if (error) {
      LOG(ERROR) << "Failed to create failover recovery dump directory: "
                 << output_path.parent_path().string()
                 << ", error: " << error.message();
      enabled_ = false;
      return;
    }
  }
  output_.open(config.file_path, std::ios::app);
  if (!output_.is_open()) {
    LOG(ERROR) << "Failed to open failover recovery dump file: "
               << config.file_path << ", error: " << std::strerror(errno);
    enabled_ = false;
    return;
  }

  writer_thread_ = std::thread([this]() { WriterLoop(); });
}

FailoverRecoveryDumper::~FailoverRecoveryDumper() { Stop(); }

void FailoverRecoveryDumper::Dump(FailoverRecoveryDumpRecord record) {
  if (!enabled_) {
    return;
  }

  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (stopped_) {
      return;
    }
    if (queue_.size() >= max_queue_size_) {
      LOG_EVERY_N(WARNING, 100)
          << "Failover recovery dump queue is full; dropping records.";
      return;
    }
    queue_.emplace_back(std::move(record));
  }
  cv_.notify_one();
}

void FailoverRecoveryDumper::FlushForTest() {
  if (!enabled_) {
    return;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  drained_cv_.wait(lock, [this]() { return queue_.empty() && !writing_; });
  output_.flush();
}

void FailoverRecoveryDumper::Stop() {
  if (!enabled_) {
    return;
  }
  {
    std::lock_guard<std::mutex> guard(mutex_);
    stopped_ = true;
  }
  cv_.notify_one();
  if (writer_thread_.joinable()) {
    writer_thread_.join();
  }
  output_.flush();
  output_.close();
  enabled_ = false;
}

void FailoverRecoveryDumper::WriterLoop() {
  while (true) {
    FailoverRecoveryDumpRecord record;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return stopped_ || !queue_.empty(); });
      if (queue_.empty()) {
        if (stopped_) {
          drained_cv_.notify_all();
          return;
        }
        continue;
      }
      record = std::move(queue_.front());
      queue_.pop_front();
      writing_ = true;
    }
    WriteRecord(record);
    {
      std::lock_guard<std::mutex> guard(mutex_);
      writing_ = false;
      if (queue_.empty()) {
        drained_cv_.notify_all();
      }
    }
  }
}

void FailoverRecoveryDumper::WriteRecord(
    const FailoverRecoveryDumpRecord& record) {
  nlohmann::json json;
  nlohmann::json origin;
  nlohmann::json failover;
  json["request_id"] = record.service_request_id;

  origin["from_prefill"] = record.origin_from_prefill;
  origin["from_decode"] = record.origin_from_decode;
  origin["prompt_tokens"] = record.prompt_tokens;
  origin["generated_tokens"] = record.generated_tokens;
  SetSlo(&origin, "original_ttft_slo_ms", record.original_ttft_slo_ms);
  SetSlo(&origin, "original_tpot_slo_ms", record.original_tpot_slo_ms);
  SetSlo(&origin, "original_ttlt_slo_ms", record.original_ttlt_slo_ms);
  json["origin"] = std::move(origin);

  failover["attempt"] = record.failover_attempt;
  failover["type"] = record.failover_type;
  failover["to_prefill"] = record.failover_to_prefill;
  failover["to_decode"] = record.failover_to_decode;
  failover["prev_token_to_detected_ms"] = record.prev_token_to_detected_ms;
  failover["detected_to_redispatch_ms"] = record.detected_to_redispatch_ms;
  failover["dispatch_to_next_token_ms"] = record.dispatch_to_next_token_ms;
  failover["prev_token_to_next_ms"] = record.prev_token_to_next_ms;
  failover["restore_blocks"] = record.restore_blocks;
  failover["recompute_tokens"] = record.recompute_tokens;
  failover["restore_cost_ms"] = record.restore_cost_ms;
  failover["recompute_cost_ms"] = record.recompute_cost_ms;
  failover["recovery_cost_ms"] = record.recovery_cost_ms;
  failover["failed_decode_offload_batch_size"] =
      record.failed_decode_offload_batch_size;
  SetSlo(&failover, "effective_ttft_slo_ms", record.effective_ttft_slo_ms);
  SetSlo(&failover, "effective_tpot_slo_ms", record.effective_tpot_slo_ms);
  SetSlo(&failover, "effective_ttlt_slo_ms", record.effective_ttlt_slo_ms);
  json["failover"] = std::move(failover);

  output_ << json.dump() << '\n';
  output_.flush();
}

}  // namespace xllm_service
