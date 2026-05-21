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

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include "common/macros.h"
#include "failover/tracker.h"

namespace xllm_service {

struct FailoverRecoveryDumpConfig {
  bool enabled = false;
  std::string file_path = "trace/failover_recovery.jsonl";
  size_t max_queue_size = 4096;
};

struct FailoverRecoveryDumpRecord {
  std::string service_request_id;
  int32_t failover_attempt = 0;
  std::string failover_type;
  std::string origin_from_prefill;
  std::string origin_from_decode;
  std::string failover_to_prefill;
  std::string failover_to_decode;
  int64_t prev_token_to_detected_ms = 0;
  int64_t detected_to_redispatch_ms = 0;
  int64_t dispatch_to_next_token_ms = 0;
  int64_t prev_token_to_next_ms = 0;
  int64_t restore_cost_ms = 0;
  int64_t recompute_cost_ms = 0;
  int64_t recovery_cost_ms = 0;
  int64_t restore_blocks = 0;
  int64_t recompute_tokens = 0;
  uint32_t failed_decode_offload_batch_size = UINT32_MAX;
  int64_t prompt_tokens = 0;
  int64_t generated_tokens = 0;
  int32_t original_ttft_slo_ms = kUnsetSloMs;
  int32_t original_tpot_slo_ms = kUnsetSloMs;
  int32_t original_ttlt_slo_ms = kUnsetSloMs;
  int32_t effective_ttft_slo_ms = kUnsetSloMs;
  int32_t effective_tpot_slo_ms = kUnsetSloMs;
  int32_t effective_ttlt_slo_ms = kUnsetSloMs;
};

FailoverRecoveryDumpRecord BuildFailoverRecoveryDumpRecord(
    const Request& request,
    const FailoverFirstTokenSample& sample,
    absl::Time first_token_time);

class FailoverRecoveryDumper {
 public:
  explicit FailoverRecoveryDumper(FailoverRecoveryDumpConfig config);
  ~FailoverRecoveryDumper();

  FailoverRecoveryDumper(const FailoverRecoveryDumper&) = delete;
  FailoverRecoveryDumper& operator=(const FailoverRecoveryDumper&) = delete;
  FailoverRecoveryDumper(FailoverRecoveryDumper&&) = delete;
  FailoverRecoveryDumper& operator=(FailoverRecoveryDumper&&) = delete;

  bool enabled() const { return enabled_; }
  void Dump(FailoverRecoveryDumpRecord record);
  void FlushForTest();

 private:
  void WriterLoop();
  void WriteRecord(const FailoverRecoveryDumpRecord& record);
  void Stop();

  bool enabled_ = false;
  size_t max_queue_size_ = 0;
  std::ofstream output_;
  std::thread writer_thread_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable drained_cv_;
  std::deque<FailoverRecoveryDumpRecord> queue_ GUARDED_BY(mutex_);
  bool writing_ GUARDED_BY(mutex_) = false;
  bool stopped_ GUARDED_BY(mutex_) = false;
};

}  // namespace xllm_service
