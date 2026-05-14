/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <cerrno>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "disagg_pd.pb.h"
#include "scheduler/managers/failover_partition.h"
#include "scheduler/managers/slo_scoring.h"
#include "scheduler/scheduler.h"

namespace {
using xllm_service::InstanceRuntimeState;
using xllm_service::InstanceType;
std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};

std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";

uint64_t current_time_ms() {
  return static_cast<uint64_t>(
      absl::ToInt64Milliseconds(absl::Now() - absl::UnixEpoch()));
}

bool is_instance_schedulable(const xllm_service::InstanceMetaInfo& info) {
  return info.runtime_state == InstanceRuntimeState::ACTIVE;
}

const char* runtime_state_name(InstanceRuntimeState state) {
  switch (state) {
    case InstanceRuntimeState::ACTIVE:
      return "ACTIVE";
    case InstanceRuntimeState::LEASE_LOST:
      return "LEASE_LOST";
    case InstanceRuntimeState::SUSPECT:
      return "SUSPECT";
  }
  return "UNKNOWN";
}

bool select_next_schedulable_instance(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index,
    uint64_t* next_index,
    std::string* instance_name) {
  if (index.empty()) {
    return false;
  }

  const uint64_t start_index = *next_index % index.size();
  for (uint64_t offset = 0; offset < index.size(); ++offset) {
    const uint64_t index_pos = (start_index + offset) % index.size();
    auto it = instances.find(index[index_pos]);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    *instance_name = index[index_pos];
    *next_index = index_pos + 1;
    return true;
  }
  return false;
}

size_t count_schedulable_instances(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index) {
  size_t count = 0;
  for (const auto& name : index) {
    auto it = instances.find(name);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    ++count;
  }
  return count;
}

InstanceType get_cleanup_type(const xllm_service::InstanceMetaInfo& info) {
  if (info.type == InstanceType::DEFAULT) {
    return InstanceType::PREFILL;
  }
  if (info.type == InstanceType::MIX) {
    return info.current_type;
  }
  return info.type;
}
}  // namespace

namespace xllm_service {

bool IsInstanceSchedulableForTest(InstanceRuntimeState runtime_state) {
  InstanceMetaInfo info;
  info.runtime_state = runtime_state;
  return is_instance_schedulable(info);
}

class FailoverSessionStreamHandler
    : public brpc::StreamInputHandler,
      public std::enable_shared_from_this<FailoverSessionStreamHandler> {
 public:
  FailoverSessionStreamHandler(InstanceMgr* instance_mgr,
                               std::string instance_name,
                               std::string incarnation_id)
      : instance_mgr_(instance_mgr),
        instance_name_(std::move(instance_name)),
        incarnation_id_(std::move(incarnation_id)) {}

  int on_received_messages(brpc::StreamId id,
                           butil::IOBuf* const messages[],
                           size_t size) override {
    VLOG(1) << "failover_session_keepalive"
            << " instance=" << instance_name_
            << " incarnation_id=" << incarnation_id_
            << " stream_id=" << id << " message_count=" << size;
    return 0;
  }

  void on_idle_timeout(brpc::StreamId id) override {
    auto self = shared_from_this();
    set_error(ETIMEDOUT, "failover session idle timeout");
    instance_mgr_->handle_failover_session_disconnect(
        instance_name_, incarnation_id_, id, ETIMEDOUT,
        "failover session idle timeout");
  }

  void on_closed(brpc::StreamId id) override {
    auto self = shared_from_this();
    int error_code = 0;
    std::string error_text;
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_code = error_code_;
      error_text = error_text_;
    }
    instance_mgr_->handle_failover_session_disconnect(
        instance_name_, incarnation_id_, id, error_code, error_text);
  }

  void on_failed(brpc::StreamId id,
                 int error_code,
                 const std::string& error_text) override {
    auto self = shared_from_this();
    set_error(error_code, error_text);
  }

 private:
  void set_error(int error_code, const std::string& error_text) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_code_ = error_code;
    error_text_ = error_text;
  }

  InstanceMgr* instance_mgr_;
  const std::string instance_name_;
  const std::string incarnation_id_;
  std::mutex error_mutex_;
  int error_code_ = 0;
  std::string error_text_;
};

bool ShouldMarkInstanceSuspectOnHeartbeatTimeout(
    InstanceRuntimeState runtime_state,
    uint64_t latest_timestamp_ms,
    uint64_t now_ms,
    int64_t heartbeat_timeout_ms) {
  if (runtime_state != InstanceRuntimeState::ACTIVE ||
      heartbeat_timeout_ms <= 0 || now_ms < latest_timestamp_ms) {
    return false;
  }
  return now_ms - latest_timestamp_ms >=
         static_cast<uint64_t>(heartbeat_timeout_ms);
}

InstanceRuntimeState RestoreRuntimeStateAfterHeartbeat(
    InstanceRuntimeState previous_runtime_state) {
  if (previous_runtime_state == InstanceRuntimeState::LEASE_LOST) {
    return InstanceRuntimeState::LEASE_LOST;
  }
  return InstanceRuntimeState::ACTIVE;
}

bool ShouldTriggerFailoverOnSessionDisconnect(
    const FailoverSessionState& session,
    const std::string& disconnected_instance_name,
    const std::string& disconnected_incarnation_id,
    brpc::StreamId disconnected_stream_id) {
  return session.stream_id != brpc::INVALID_STREAM_ID &&
         session.instance_name == disconnected_instance_name &&
         session.incarnation_id == disconnected_incarnation_id &&
         session.stream_id == disconnected_stream_id;
}

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service,
                         Scheduler* scheduler)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client),
      scheduler_(scheduler) {
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();

  if (!CanUseDynamicPdFlip(options_)) {
    LOG(INFO) << "Dynamic PD flips are disabled; SLO-aware routing remains "
                 "enabled.";
  }

  state_reconcile_thread_ = std::make_unique<std::thread>(
      &InstanceMgr::reconcile_instance_states, this);
}

void InstanceMgr::init() {
  std::unordered_map<std::string, InstanceMetaInfo> loaded_instances;
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->get_prefix(it.second, &loaded_instances);
  }
  LOG(INFO) << "Load instance info from etcd:" << loaded_instances.size();

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    prefill_index_.reserve(loaded_instances.size());
    decode_index_.reserve(loaded_instances.size());
  }

  for (auto& pair : loaded_instances) {
    if (!register_instance(pair.first, pair.second)) {
      LOG(ERROR) << "Fail to register instance: " << pair.first;
    }
  }

  std::unordered_map<std::string, LoadMetrics> loaded_metrics;
  etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &loaded_metrics);
  {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    load_metrics_ = std::move(loaded_metrics);
  }

  {
    std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
    for (int i = 0; i < prefill_index_.size(); i++) {
      LOG(INFO) << i << " : " << prefill_index_[i];
    }
  }
}

InstanceMgr::~InstanceMgr() {
  exited_ = true;
  if (state_reconcile_thread_ && state_reconcile_thread_->joinable()) {
    state_reconcile_thread_->join();
  }

  std::vector<brpc::StreamId> stream_ids;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    failover_sessions_.clear();
    stream_ids.reserve(failover_session_handlers_.size());
    for (const auto& [stream_id, handler] : failover_session_handlers_) {
      stream_ids.push_back(stream_id);
    }
  }
  for (brpc::StreamId stream_id : stream_ids) {
    brpc::StreamClose(stream_id);
  }
  for (int attempt = 0; attempt < 50; ++attempt) {
    {
      std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
      if (failover_session_handlers_.empty()) {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::can_route_prefill_without_decode_locked(
    const std::string& prefill_name) const {
  auto selected_it = instances_.find(prefill_name);
  if (selected_it == instances_.end() ||
      selected_it->second.type != InstanceType::DEFAULT) {
    LOG(ERROR) << "No decode instance and selected prefill is not default, "
               << "instance: " << prefill_name;
    return false;
  }
  return true;
}

bool InstanceMgr::get_next_instance_pair(Routing* routing) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  if (prefill_index_.empty()) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }

  routing->decode_name.clear();
  if (suspect_instances_.empty()) {
    // Fast path for the common case: no suspect instances, keep plain RR.
    next_prefill_index_ = next_prefill_index_ % prefill_index_.size();
    routing->prefill_name = prefill_index_[next_prefill_index_];
    next_prefill_index_++;

    if (decode_index_.empty()) {
      return can_route_prefill_without_decode_locked(routing->prefill_name);
    }

    next_decode_index_ = next_decode_index_ % decode_index_.size();
    routing->decode_name = decode_index_[next_decode_index_];
    next_decode_index_++;
    return true;
  }

  if (!select_next_schedulable_instance(instances_,
                                        prefill_index_,
                                        &next_prefill_index_,
                                        &routing->prefill_name)) {
    LOG(ERROR) << "No schedulable prefill or default instance found!";
    return false;
  }

  if (decode_index_.empty()) {
    return can_route_prefill_without_decode_locked(routing->prefill_name);
  }

  select_next_schedulable_instance(
      instances_, decode_index_, &next_decode_index_, &routing->decode_name);
  return true;
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE &&
        is_instance_schedulable(inst.second)) {
      decode_list.emplace_back(inst.second.name);
    }
  }

  return decode_list;
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  for (auto& inst : instances_) {
    if ((inst.second.type == InstanceType::PREFILL ||
         inst.second.type == InstanceType::DEFAULT) &&
        is_instance_schedulable(inst.second)) {
      prefill_list.emplace_back(inst.second.name);
    }
  }

  return prefill_list;
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos) {
  std::shared_lock<std::shared_mutex> inst_lock(cluster_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(metrics_mutex_);

  for (auto name : infos->overlap_scores.instances) {
    auto it = load_metrics_.find(name);
    if (it == load_metrics_.end()) {
      continue;
    }
    auto instance_it = instances_.find(name);
    if (instance_it == instances_.end() ||
        !is_instance_schedulable(instance_it->second)) {
      continue;
    }

    if (instance_it->second.type == InstanceType::DECODE) {
      infos->decode_load_metrics.insert(std::make_pair(name, it->second));
      infos->decode_max_waiting_requests_num =
          std::max(infos->decode_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    } else {
      infos->prefill_load_metrics.insert(std::make_pair(name, it->second));
      infos->prefill_max_waiting_requests_num =
          std::max(infos->prefill_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    }
  }

  std::string least_loaded_prefill_instance;
  float least_loaded_prefill_gpu_cache_usage_perc = 1;
  std::string least_loaded_decode_instance;
  float least_loaded_decode_gpu_cache_usage_perc = 1;

  if (infos->prefill_load_metrics.size() == 0 ||
      infos->decode_load_metrics.size() == 0) {
    for (const auto& metric : load_metrics_) {
      auto instance_it = instances_.find(metric.first);
      if (instance_it == instances_.end() ||
          !is_instance_schedulable(instance_it->second)) {
        continue;
      }
      if (instance_it->second.type != InstanceType::DECODE) {
        if (metric.second.gpu_cache_usage_perc <
            least_loaded_prefill_gpu_cache_usage_perc) {
          least_loaded_prefill_gpu_cache_usage_perc =
              metric.second.gpu_cache_usage_perc;
          least_loaded_prefill_instance = metric.first;
        }
      } else {
        if (metric.second.gpu_cache_usage_perc <
            least_loaded_decode_gpu_cache_usage_perc) {
          least_loaded_decode_gpu_cache_usage_perc =
              metric.second.gpu_cache_usage_perc;
          least_loaded_decode_instance = metric.first;
        }
      }
    }
  }

  if (infos->prefill_load_metrics.size() == 0 &&
      !least_loaded_prefill_instance.empty()) {
    infos->prefill_load_metrics.insert(
        std::make_pair(least_loaded_prefill_instance,
                       load_metrics_[least_loaded_prefill_instance]));
  }

  if (infos->decode_load_metrics.size() == 0 &&
      !least_loaded_decode_instance.empty()) {
    infos->decode_load_metrics.insert(
        std::make_pair(least_loaded_decode_instance,
                       load_metrics_[least_loaded_decode_instance]));
  }
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc(),
                  load_metrics.offload_batch_size()));
}

bool InstanceMgr::upload_load_metrics() {
  std::unordered_map<std::string, LoadMetrics> upload_snapshot;
  std::unordered_set<std::string> remove_snapshot;
  {
    std::unique_lock<std::shared_mutex> lk(metrics_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, iter.second);
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
    upload_snapshot = updated_metrics_;
    remove_snapshot = removed_instance_;
    updated_metrics_.clear();
    removed_instance_.clear();
  }
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, upload_snapshot);
  status = status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, remove_snapshot);
  return status;
}

void InstanceMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::bind_request_instance_incarnations(
    const std::shared_ptr<Request>& request) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);

  // Bind the selected routing to a concrete incarnation before dispatch.
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();

  if (!request->routing.prefill_name.empty()) {
    auto prefill_it = instances_.find(request->routing.prefill_name);
    if (prefill_it == instances_.end()) {
      LOG(ERROR) << "Prefill instance is not registered when binding request: "
                 << request->routing.prefill_name;
      return false;
    }
    if (!is_instance_schedulable(prefill_it->second)) {
      LOG(ERROR) << "Prefill instance is not schedulable when binding request: "
                 << request->routing.prefill_name << ", state: "
                 << runtime_state_name(prefill_it->second.runtime_state);
      return false;
    }
    request->prefill_incarnation_id = prefill_it->second.incarnation_id;
  }

  if (!request->routing.decode_name.empty()) {
    auto decode_it = instances_.find(request->routing.decode_name);
    if (decode_it == instances_.end()) {
      LOG(ERROR) << "Decode instance is not registered when binding request: "
                 << request->routing.decode_name;
      return false;
    }
    if (!is_instance_schedulable(decode_it->second)) {
      LOG(ERROR) << "Decode instance is not schedulable when binding request: "
                 << request->routing.decode_name << ", state: "
                 << runtime_state_name(decode_it->second.runtime_state);
      return false;
    }
    request->decode_incarnation_id = decode_it->second.incarnation_id;
  }

  return true;
}

bool InstanceMgr::record_instance_heartbeat(const std::string& instance_name,
                                            const std::string& incarnation_id) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  auto it = instances_.find(instance_name);
  if (it == instances_.end()) {
    LOG(WARNING) << "Ignore heartbeat from unknown instance: " << instance_name;
    return false;
  }

  if (it->second.incarnation_id != incarnation_id) {
    LOG(WARNING) << "Ignore stale heartbeat from instance: " << instance_name
                 << ", current incarnation_id: " << it->second.incarnation_id
                 << ", heartbeat incarnation_id: " << incarnation_id;
    return false;
  }

  it->second.latest_timestamp = current_time_ms();
  if (it->second.runtime_state == InstanceRuntimeState::SUSPECT) {
    const auto suspect_it = suspect_instances_.find(instance_name);
    const InstanceRuntimeState restored_state =
        suspect_it == suspect_instances_.end()
            ? InstanceRuntimeState::ACTIVE
            : RestoreRuntimeStateAfterHeartbeat(
                  suspect_it->second.previous_runtime_state);
    clear_suspect_instance(instance_name, incarnation_id);
    it->second.runtime_state = restored_state;
    LOG(WARNING) << "Heartbeat recovered for suspect instance, move to "
                 << runtime_state_name(restored_state) << ": " << instance_name
                 << ", incarnation_id: " << incarnation_id;
  }
  return true;
}

bool InstanceMgr::open_failover_session(
    const proto::FailoverSessionRequest& request,
    brpc::Controller* cntl) {
  if (cntl == nullptr || request.name().empty() ||
      request.incarnation_id().empty()) {
    return false;
  }
  if (!is_master_service_.load()) {
    LOG(WARNING) << "Reject failover session on non-master service, instance="
                 << request.name()
                 << " incarnation_id=" << request.incarnation_id();
    return false;
  }

  {
    std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
    auto instance_it = instances_.find(request.name());
    if (instance_it == instances_.end()) {
      LOG(WARNING) << "Reject failover session from unknown instance: "
                   << request.name();
      return false;
    }
    if (instance_it->second.incarnation_id != request.incarnation_id()) {
      LOG(WARNING) << "Reject failover session from stale instance, instance="
                   << request.name()
                   << " current_incarnation_id="
                   << instance_it->second.incarnation_id
                   << " request_incarnation_id=" << request.incarnation_id();
      return false;
    }
  }

  auto handler = std::make_shared<FailoverSessionStreamHandler>(
      this, request.name(), request.incarnation_id());
  brpc::StreamOptions stream_options;
  stream_options.handler = handler.get();
  stream_options.idle_timeout_ms = -1;

  brpc::StreamId stream_id = brpc::INVALID_STREAM_ID;
  if (brpc::StreamAccept(&stream_id, *cntl, &stream_options) != 0) {
    LOG(ERROR) << "Failed to accept failover session, instance="
               << request.name()
               << " incarnation_id=" << request.incarnation_id();
    return false;
  }

  brpc::StreamId previous_stream_id = brpc::INVALID_STREAM_ID;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    failover_session_handlers_[stream_id] = handler;

    auto instance_it = instances_.find(request.name());
    if (instance_it == instances_.end() ||
        instance_it->second.incarnation_id != request.incarnation_id()) {
      lock.unlock();
      brpc::StreamClose(stream_id);
      LOG(WARNING) << "Close failover session after stale registration check, "
                   << "instance=" << request.name()
                   << " incarnation_id=" << request.incarnation_id()
                   << " stream_id=" << stream_id;
      return false;
    }

    auto session_it = failover_sessions_.find(request.name());
    if (session_it != failover_sessions_.end() &&
        session_it->second.stream_id != stream_id) {
      previous_stream_id = session_it->second.stream_id;
    }

    failover_sessions_[request.name()] = FailoverSessionState{
        .instance_name = request.name(),
        .incarnation_id = request.incarnation_id(),
        .stream_id = stream_id,
        .connected_at_ms = current_time_ms(),
    };
  }

  if (previous_stream_id != brpc::INVALID_STREAM_ID &&
      previous_stream_id != stream_id) {
    brpc::StreamClose(previous_stream_id);
  }

  LOG(INFO) << "failover_session_opened"
            << " instance=" << request.name()
            << " incarnation_id=" << request.incarnation_id()
            << " stream_id=" << stream_id
            << " replaced_stream_id=" << previous_stream_id;
  return true;
}

void InstanceMgr::handle_failover_session_disconnect(
    const std::string& instance_name,
    const std::string& incarnation_id,
    brpc::StreamId stream_id,
    int error_code,
    const std::string& error_text) {
  InstanceType cleanup_type = InstanceType::DEFAULT;
  bool should_queue_failover = false;
  bool ignored_disconnect = false;
  bool already_suspect = false;

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    failover_session_handlers_.erase(stream_id);

    auto session_it = failover_sessions_.find(instance_name);
    if (session_it == failover_sessions_.end()) {
      ignored_disconnect = true;
    } else if (!ShouldTriggerFailoverOnSessionDisconnect(session_it->second,
                                                         instance_name,
                                                         incarnation_id,
                                                         stream_id)) {
      ignored_disconnect = true;
    } else {
      failover_sessions_.erase(session_it);
      auto instance_it = instances_.find(instance_name);
      if (instance_it == instances_.end() ||
          instance_it->second.incarnation_id != incarnation_id) {
        ignored_disconnect = true;
      } else if (instance_it->second.runtime_state ==
                 InstanceRuntimeState::SUSPECT) {
        already_suspect = true;
      } else {
        cleanup_type = get_cleanup_type(instance_it->second);
        mark_instance_suspect(instance_name, incarnation_id);
        should_queue_failover = true;
      }
    }
  }

  if (ignored_disconnect) {
    LOG(INFO) << "failover_session_closed_ignored"
              << " instance=" << instance_name
              << " incarnation_id=" << incarnation_id
              << " stream_id=" << stream_id
              << " error_code=" << error_code
              << " error_text=" << error_text;
    return;
  }

  if (already_suspect) {
    LOG(INFO) << "failover_session_closed_on_suspect_instance"
              << " instance=" << instance_name
              << " incarnation_id=" << incarnation_id
              << " stream_id=" << stream_id
              << " error_code=" << error_code
              << " error_text=" << error_text;
    return;
  }

  LOG(WARNING) << "failover_instance_suspect"
               << " instance=" << instance_name
               << " incarnation_id=" << incarnation_id
               << " cleanup_type=" << static_cast<int32_t>(cleanup_type)
               << " reason=rpc_session_closed"
               << " stream_id=" << stream_id
               << " error_code=" << error_code
               << " error_text=" << error_text;
  const size_t failover_count = scheduler_->clear_requests_on_failed_instance(
      instance_name, incarnation_id, cleanup_type);
  if (failover_count > 0) {
    LOG(INFO) << "failover_rehandle_trigger"
              << " instance=" << instance_name
              << " incarnation_id=" << incarnation_id
              << " count=" << failover_count
              << " reason=rpc_session_closed";
    scheduler_->rehandle_removed_request();
  }
}

bool InstanceMgr::init_brpc_channel(
    const std::string& instance_name,
    std::shared_ptr<brpc::Channel>* out_channel) {
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  // Add to params
  // options.protocol = "http";
  options.timeout_ms = options_.timeout_ms(); /*milliseconds*/
  options.max_retry = 3;
  options.connect_timeout_ms = options_.connect_timeout_ms();
  std::string load_balancer = "";
  if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
      0) {
    LOG(ERROR) << "Fail to initialize channel for " << instance_name;
    return false;
  }
  *out_channel = std::move(channel);
  return true;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    for (const auto& event : response.events()) {
      const std::string instance_name = get_event_key_suffix(event, prefix_len);
      if (instance_name.empty()) {
        continue;
      }

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = get_event_value(event);
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "Parse instance json failed: " << json_str;
          continue;
        }

        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          lock.unlock();
          if (!register_instance(instance_name, metainfo)) {
            LOG(ERROR) << "Fail to register instance: " << instance_name;
          }
          continue;
        }

        if (existing_it->second.incarnation_id == metainfo.incarnation_id) {
          const auto previous_state = existing_it->second.runtime_state;
          refresh_instance_registration(instance_name, metainfo);
          clear_suspect_instance(instance_name, metainfo.incarnation_id);
          if (previous_state != InstanceRuntimeState::ACTIVE) {
            LOG(INFO) << "Instance registration restored, back to active: "
                      << instance_name
                      << ", incarnation_id: " << metainfo.incarnation_id
                      << ", previous_state: "
                      << runtime_state_name(previous_state);
          }
          continue;
        }

        const std::string old_incarnation_id =
            existing_it->second.incarnation_id;
        LOG(WARNING) << "Detected instance replacement, instance_name: "
                     << instance_name
                     << ", old incarnation_id: " << old_incarnation_id
                     << ", new incarnation_id: " << metainfo.incarnation_id;
        lock.unlock();
        deregister_instance(instance_name, old_incarnation_id);
        if (!register_instance(instance_name, metainfo)) {
          LOG(ERROR) << "Fail to register replacement instance: "
                     << instance_name;
        }
        continue;
      }

      if (event.event_type() != etcd::Event::EventType::DELETE_) {
        continue;
      }

      InstanceMetaInfo deleted_info;
      std::string deleted_incarnation_id;
      const auto deleted_value = get_event_value(event);
      if (!deleted_value.empty() &&
          deleted_info.parse_from_json(deleted_value)) {
        deleted_incarnation_id = deleted_info.incarnation_id;
      }
      std::string tracked_incarnation_id;
      InstanceRuntimeState tracked_runtime_state = InstanceRuntimeState::ACTIVE;
      uint64_t tracked_latest_timestamp_ms = 0;
      {
        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          continue;
        }

        if (!deleted_incarnation_id.empty() &&
            existing_it->second.incarnation_id != deleted_incarnation_id) {
          LOG(INFO) << "Ignore stale delete for replaced instance: "
                    << instance_name
                    << ", deleted incarnation_id: " << deleted_incarnation_id
                    << ", current incarnation_id: "
                    << existing_it->second.incarnation_id;
          continue;
        }
        tracked_incarnation_id = existing_it->second.incarnation_id;
        tracked_runtime_state = existing_it->second.runtime_state;
        tracked_latest_timestamp_ms = existing_it->second.latest_timestamp;
      }

      const uint64_t delete_event_seen_at_ms = current_time_ms();
      const uint64_t silence_before_delete_event_ms =
          delete_event_seen_at_ms >= tracked_latest_timestamp_ms
              ? delete_event_seen_at_ms - tracked_latest_timestamp_ms
              : 0;
      LOG(INFO) << "failover_delete_event_received"
                << " instance=" << instance_name
                << " tracked_incarnation_id=" << tracked_incarnation_id
                << " deleted_incarnation_id=" << deleted_incarnation_id
                << " runtime_state=" << runtime_state_name(tracked_runtime_state)
                << " last_heartbeat_unix_ms=" << tracked_latest_timestamp_ms
                << " silence_before_delete_event_ms="
                << silence_before_delete_event_ms;

      {
        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end() ||
            existing_it->second.incarnation_id != tracked_incarnation_id) {
          continue;
        }
        if (existing_it->second.runtime_state ==
            InstanceRuntimeState::SUSPECT) {
          LOG(INFO) << "Keep suspect state after etcd delete event, instance="
                    << instance_name
                    << ", incarnation_id=" << tracked_incarnation_id;
          continue;
        }
        clear_suspect_instance(instance_name, tracked_incarnation_id);
        existing_it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
        existing_it->second.latest_timestamp = current_time_ms();
      }

      LOG(WARNING) << "failover_instance_lease_lost"
                   << " instance=" << instance_name
                   << " incarnation_id=" << tracked_incarnation_id
                   << " reason=etcd_delete";
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  latency_metrics_.insert_or_assign(
      instance_name,
      LatencyMetrics(latency_metrics.recent_max_ttft(),
                     latency_metrics.recent_max_tbt()));
}

void InstanceMgr::reconcile_instance_states() {
  const auto suspect_interval_ms =
      std::max<int64_t>(1, options_.detect_disconnected_instance_interval()) *
      1000;
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<std::pair<std::string, std::string>> to_deregister;

    {
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      if (exited_) {
        return;
      }

      const uint64_t now_ms = current_time_ms();
      for (auto& [instance_name, info] : instances_) {
        const uint64_t silence_ms =
            now_ms >= info.latest_timestamp ? now_ms - info.latest_timestamp : 0;
        if (info.runtime_state == InstanceRuntimeState::ACTIVE &&
            silence_ms >= static_cast<uint64_t>(suspect_interval_ms)) {
          LOG(INFO) << "failover_heartbeat_timeout_detected"
                    << " instance=" << instance_name
                    << " incarnation_id=" << info.incarnation_id
                    << " runtime_state="
                    << runtime_state_name(info.runtime_state)
                    << " now_unix_ms=" << now_ms
                    << " last_heartbeat_unix_ms=" << info.latest_timestamp
                    << " silence_ms=" << silence_ms
                    << " reason=active_heartbeat_stalled_waiting_for_session_close";
          continue;
        }

        if (info.runtime_state == InstanceRuntimeState::LEASE_LOST) {
          LOG(INFO) << "failover_heartbeat_timeout_detected"
                    << " instance=" << instance_name
                    << " incarnation_id=" << info.incarnation_id
                    << " runtime_state="
                    << runtime_state_name(info.runtime_state)
                    << " now_unix_ms=" << now_ms
                    << " last_heartbeat_unix_ms=" << info.latest_timestamp
                    << " silence_ms=" << silence_ms
                    << " reason=lease_lost_waiting_for_session_close";
        }
      }

      for (auto it = suspect_instances_.begin();
           it != suspect_instances_.end();) {
        const std::string instance_name = it->first;
        const std::string incarnation_id = it->second.incarnation_id;
        const uint64_t enter_ts_ms = it->second.enter_ts_ms;
        ++it;

        if (now_ms - enter_ts_ms < suspect_interval_ms) {
          continue;
        }

        auto inst_it = instances_.find(instance_name);
        if (inst_it == instances_.end() ||
            inst_it->second.incarnation_id != incarnation_id) {
          suspect_instances_.erase(instance_name);
          continue;
        }

        LOG(WARNING) << "Suspect window expired, deregister instance: "
                     << instance_name << ", incarnation_id: " << incarnation_id;
        to_deregister.emplace_back(instance_name, incarnation_id);
      }
    }

    for (const auto& p : to_deregister) {
      deregister_instance(p.first, p.second);
    }
  }
}

void InstanceMgr::refresh_instance_registration(const std::string& name,
                                                const InstanceMetaInfo& info) {
  auto it = instances_.find(name);
  if (it == instances_.end()) {
    return;
  }

  // Preserve local scheduling/index state across etcd refreshes.
  const auto instance_index = it->second.instance_index;
  const auto current_type = it->second.current_type;

  it->second = info;
  it->second.instance_index = instance_index;
  it->second.current_type = current_type;
  it->second.latest_timestamp = current_time_ms();
  it->second.runtime_state = InstanceRuntimeState::ACTIVE;
}

void InstanceMgr::mark_instance_suspect(const std::string& name,
                                        const std::string& incarnation_id) {
  SuspectInstanceInfo info;
  info.incarnation_id = incarnation_id;
  info.enter_ts_ms = current_time_ms();
  auto it = instances_.find(name);
  if (it != instances_.end() && it->second.incarnation_id == incarnation_id) {
    info.previous_runtime_state = it->second.runtime_state;
    it->second.runtime_state = InstanceRuntimeState::SUSPECT;
  }
  suspect_instances_[name] = std::move(info);
}

void InstanceMgr::clear_suspect_instance(const std::string& name,
                                         const std::string& incarnation_id) {
  auto it = suspect_instances_.find(name);
  if (it == suspect_instances_.end()) {
    return;
  }
  if (!incarnation_id.empty() && it->second.incarnation_id != incarnation_id) {
    return;
  }
  suspect_instances_.erase(it);
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  // skip request metrics update if policy is not SLO_AWARE
  if (options_.load_balance_policy() != "SLO_AWARE") {
    return;
  }

  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);

  const bool needs_prefill_metrics =
      action == RequestAction::SCHEDULE ||
      action == RequestAction::FINISH_PREFILL || action == RequestAction::CANCEL;
  const bool needs_decode_metrics =
      action == RequestAction::SCHEDULE ||
      action == RequestAction::FINISH_PREFILL ||
      action == RequestAction::GENERATE ||
      action == RequestAction::FINISH_DECODE || action == RequestAction::CANCEL;

  auto prefill_it = request_metrics_.end();
  if (needs_prefill_metrics) {
    prefill_it = request_metrics_.find(request->routing.prefill_name);
  }
  const bool has_prefill_metrics = prefill_it != request_metrics_.end();

  auto decode_it = request_metrics_.end();
  if (needs_decode_metrics && !request->routing.decode_name.empty()) {
    decode_it = request_metrics_.find(request->routing.decode_name);
  }
  const bool has_decode_metrics = decode_it != request_metrics_.end();

  if (needs_prefill_metrics && !has_prefill_metrics &&
      !request->routing.prefill_name.empty()) {
    LOG(WARNING) << "Skip missing prefill request metrics, instance name : "
                 << request->routing.prefill_name
                 << ", action: " << static_cast<int32_t>(action);
  }
  if (needs_decode_metrics && !has_decode_metrics &&
      !request->routing.decode_name.empty()) {
    LOG(WARNING) << "Skip missing decode request metrics, instance name : "
                 << request->routing.decode_name
                 << ", action: " << static_cast<int32_t>(action);
  }
  if ((needs_prefill_metrics && !has_prefill_metrics) &&
      (needs_decode_metrics && !has_decode_metrics)) {
    return;
  }

  int64_t num_prompt_tokens = request->token_ids.size();
  int64_t num_generated_tokens = request->num_generated_tokens;
  switch (action) {
    case RequestAction::SCHEDULE:
      // update the request metrics for prefill and decode instances when
      // request is scheduled
      if (has_prefill_metrics) {
        prefill_it->second.prefill_request_num += 1;
        prefill_it->second.prefill_token_num += num_prompt_tokens;
      }

      if (has_decode_metrics) {
        decode_it->second.decode_request_num += 1;
        decode_it->second.decode_token_num += num_prompt_tokens;
      }
      break;
    case RequestAction::FINISH_PREFILL:
      // update the request metrics for prefill and decode instance when request
      // finishes the prefill phase
      if (has_prefill_metrics) {
        prefill_it->second.prefill_request_num -= 1;
        prefill_it->second.prefill_token_num -= num_prompt_tokens;
        prefill_it->second.estimated_prefill_time -= request->estimated_ttft;
      }

      if (has_decode_metrics) {
        decode_it->second.decode_token_num += 1;
      }
      break;
    case RequestAction::GENERATE:
      // update the request metrics for decode instance when request generate a
      // token
      if (has_decode_metrics) {
        decode_it->second.decode_token_num += 1;
      }
      break;
    case RequestAction::FINISH_DECODE:
      // update the request metrics for decode instance when request finishes
      // the decode phase
      if (has_decode_metrics) {
        decode_it->second.decode_request_num -= 1;
        decode_it->second.decode_token_num -=
            (num_prompt_tokens + num_generated_tokens);
      }

      break;
    case RequestAction::CANCEL:
      // update the request metrics for prefill and decode instances when
      // request is cancelled
      if (has_prefill_metrics) {
        prefill_it->second.prefill_request_num -= 1;
        prefill_it->second.prefill_token_num -= num_prompt_tokens;
        prefill_it->second.estimated_prefill_time -= request->estimated_ttft;
      }

      if (has_decode_metrics) {
        decode_it->second.decode_request_num -= 1;
        decode_it->second.decode_token_num -=
            (num_prompt_tokens + num_generated_tokens);
      }

      break;
    default:
      LOG(ERROR) << "Unknown RequestAction: " << static_cast<int32_t>(action);
      break;
  }

  if (has_decode_metrics && CanUseDynamicPdFlip(options_) &&
      decode_it->second.decode_request_num == 0) {
    flip_decode_to_prefill(request->routing.decode_name);
  }
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);
  const bool has_unschedulable_instances = !suspect_instances_.empty();

  std::string min_prefill_instance;
  PrefillCandidateScore best_prefill_score;
  int64_t total_prefill_time = 0;
  size_t schedulable_prefill_count = 0;
  for (size_t i = 0; i < prefill_index_.size(); ++i) {
    const auto& prefill_instance = prefill_index_[i];
    if (has_unschedulable_instances) {
      auto it = instances_.find(prefill_instance);
      if (it == instances_.end() || !is_instance_schedulable(it->second)) {
        continue;
      }
    }

    const auto& metrics = request_metrics_[prefill_instance];
    const int64_t prefill_time = metrics.estimated_prefill_time;
    total_prefill_time += prefill_time;
    PrefillCandidateScore score;
    score.estimated_prefill_time = prefill_time;
    score.prefill_request_num = metrics.prefill_request_num;
    score.prefill_token_num = metrics.prefill_token_num;
    score.order =
        RotatingCandidateOrder(i, next_prefill_index_, prefill_index_.size());
    if (IsBetterPrefillCandidate(score, best_prefill_score)) {
      min_prefill_instance = prefill_instance;
      best_prefill_score = score;
    }
    ++schedulable_prefill_count;
  }

  if (schedulable_prefill_count == 0) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }
  int64_t avg_prefill_time = total_prefill_time / schedulable_prefill_count;
  const int64_t min_prefill_time = best_prefill_score.estimated_prefill_time;

  std::string min_decode_instance;
  DecodeCandidateScore best_decode_score;
  std::string target_decode_instance;
  DecodeCandidateScore target_decode_score;
  size_t schedulable_decode_count = 0;
  for (size_t i = 0; i < decode_index_.size(); ++i) {
    const auto& decode_instance = decode_index_[i];
    if (has_unschedulable_instances) {
      auto it = instances_.find(decode_instance);
      if (it == instances_.end() || !is_instance_schedulable(it->second)) {
        continue;
      }
    }

    const auto& metrics = request_metrics_[decode_instance];
    int64_t token_num = metrics.decode_token_num;
    int64_t request_num = metrics.decode_request_num;
    auto& time_predictor = get_time_predictor(decode_instance);
    int64_t estimated_tpot = NormalizeEstimatedLatencyMs(
        time_predictor.predict_tpot(token_num + request->token_ids.size(),
                                    request_num + 1),
        FLAGS_target_tpot);
    DecodeCandidateScore score;
    score.estimated_tpot = estimated_tpot;
    score.decode_request_num = request_num;
    score.decode_token_num = token_num;
    score.order =
        RotatingCandidateOrder(i, next_decode_index_, decode_index_.size());
    if (estimated_tpot <= FLAGS_target_tpot &&
        IsBetterDecodeCandidate(score, target_decode_score)) {
      target_decode_instance = decode_instance;
      target_decode_score = score;
    }

    if (IsBetterDecodeCandidate(score, best_decode_score)) {
      min_decode_instance = decode_instance;
      best_decode_score = score;
    }
    ++schedulable_decode_count;
  }

  if (schedulable_decode_count == 0) {
    LOG(ERROR) << "No decode instance found!";
    return false;
  }

  if (!target_decode_instance.empty()) {
    request->routing.decode_name = target_decode_instance;
  } else {
    request->routing.decode_name = min_decode_instance;
  }
  next_decode_index_ =
      (next_decode_index_ + best_decode_score.order + 1) %
      decode_index_.size();

  // select prefill instance
  float tpot_threshold =
      (schedulable_decode_count - 1.0f) / schedulable_decode_count;
  // When the prefill instances are already overloaded and there are other
  // instances with lower loads in the decode group, we will dispatch the
  // prefill requests to those instances to alleviate the pressure on the
  // prefill instances.
  if (min_prefill_time > FLAGS_target_ttft &&
      target_decode_instance != min_decode_instance &&
      best_decode_score.estimated_tpot < FLAGS_target_tpot * tpot_threshold &&
      request_metrics_[min_decode_instance].estimated_prefill_time <
          min_prefill_time) {
    request->routing.prefill_name = min_decode_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_decode_instance);
    request->estimated_ttft = NormalizeEstimatedLatencyMs(
        time_predictor.predict_ttft(request->token_ids.size()),
        FLAGS_target_ttft);
    request_metrics_[min_decode_instance].estimated_prefill_time +=
        request->estimated_ttft;
  } else {
    request->routing.prefill_name = min_prefill_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_prefill_instance);
    request->estimated_ttft = NormalizeEstimatedLatencyMs(
        time_predictor.predict_ttft(request->token_ids.size()),
        FLAGS_target_ttft);
    request_metrics_[min_prefill_instance].estimated_prefill_time +=
        request->estimated_ttft;
  }
  next_prefill_index_ =
      (next_prefill_index_ + best_prefill_score.order + 1) %
      prefill_index_.size();

  // If there are no decode instances that meet the requirements, switch a
  // prefill instance to decode if the number of instances allows. Since the
  // current disaggregated PD mode does not support prefill and decode using the
  // same instance, we only switch the instance here, without dispatching the
  // decode request to this instance.
  float ttft_threshold =
      (schedulable_prefill_count - 1.0f) / schedulable_prefill_count;
  if (CanUseDynamicPdFlip(options_) && target_decode_instance.empty() &&
      (avg_prefill_time < FLAGS_target_ttft * ttft_threshold ||
       schedulable_decode_count < schedulable_prefill_count)) {
    flip_prefill_to_decode(request->routing.prefill_name);
  }

  return true;
}

bool InstanceMgr::select_instance_pair_on_failover(
    std::shared_ptr<Request> request) {
  if (options_.load_balance_policy() != "SLO_AWARE" ||
      request->token_ids.empty()) {
    return get_next_instance_pair(&request->routing);
  }

  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);

  std::string best_prefill_instance;
  PrefillCandidateScore best_prefill_score;
  size_t schedulable_prefill_count = 0;
  const bool has_preselected_prefill = !request->routing.prefill_name.empty();
  if (has_preselected_prefill) {
    auto instance_it = instances_.find(request->routing.prefill_name);
    if (instance_it == instances_.end() ||
        !is_instance_schedulable(instance_it->second)) {
      LOG(ERROR) << "Preselected failover prefill is not schedulable, "
                 << "instance=" << request->routing.prefill_name;
      return false;
    }
    best_prefill_instance = request->routing.prefill_name;
    best_prefill_score.order = 0;
    schedulable_prefill_count = count_schedulable_instances(instances_, prefill_index_);
  } else {
    for (size_t i = 0; i < prefill_index_.size(); ++i) {
      const auto& prefill_instance = prefill_index_[i];
      auto instance_it = instances_.find(prefill_instance);
      if (instance_it == instances_.end() ||
          !is_instance_schedulable(instance_it->second)) {
        continue;
      }

      const auto& metrics = request_metrics_[prefill_instance];
      int64_t cost = metrics.estimated_prefill_time;
      if (request->failover.runtime.type == FailoverType::PREFILL_CRASH) {
        auto& time_predictor = get_time_predictor(prefill_instance);
        const int64_t estimated_ttft = NormalizeEstimatedLatencyMs(
            time_predictor.predict_ttft(request->token_ids.size()),
            FLAGS_target_ttft);
        cost += estimated_ttft;
        PrefillCandidateScore score;
        score.estimated_prefill_time = cost;
        score.prefill_request_num = metrics.prefill_request_num;
        score.prefill_token_num = metrics.prefill_token_num;
        score.order =
            RotatingCandidateOrder(i, next_prefill_index_, prefill_index_.size());
        if (IsBetterPrefillCandidate(score, best_prefill_score)) {
          best_prefill_instance = prefill_instance;
          best_prefill_score = score;
          request->estimated_ttft = estimated_ttft;
        }
        ++schedulable_prefill_count;
        continue;
      }

      PrefillCandidateScore score;
      score.estimated_prefill_time = cost;
      score.prefill_request_num = metrics.prefill_request_num;
      score.prefill_token_num = metrics.prefill_token_num;
      score.order =
          RotatingCandidateOrder(i, next_prefill_index_, prefill_index_.size());
      if (IsBetterPrefillCandidate(score, best_prefill_score)) {
        best_prefill_instance = prefill_instance;
        best_prefill_score = score;
      }
      ++schedulable_prefill_count;
    }

    if (best_prefill_instance.empty()) {
      LOG(ERROR) << "No schedulable prefill or default instance found for "
                 << "failover request!";
      return false;
    }
  }

  std::string best_decode_instance;
  DecodeCandidateScore best_decode_score;
  size_t schedulable_decode_count = 0;
  for (size_t i = 0; i < decode_index_.size(); ++i) {
    const auto& decode_instance = decode_index_[i];
    auto instance_it = instances_.find(decode_instance);
    if (instance_it == instances_.end() ||
        !is_instance_schedulable(instance_it->second)) {
      continue;
    }

    const auto& metrics = request_metrics_[decode_instance];
    const int64_t token_num = metrics.decode_token_num;
    const int64_t request_num = metrics.decode_request_num;
    auto& time_predictor = get_time_predictor(decode_instance);
    const int64_t estimated_tpot = NormalizeEstimatedLatencyMs(
        time_predictor.predict_tpot(token_num + request->token_ids.size() +
                                        request->num_generated_tokens,
                                    request_num + 1),
        FLAGS_target_tpot);
    DecodeCandidateScore score;
    score.estimated_tpot = estimated_tpot;
    score.decode_request_num = request_num;
    score.decode_token_num = token_num;
    score.order =
        RotatingCandidateOrder(i, next_decode_index_, decode_index_.size());
    if (IsBetterDecodeCandidate(score, best_decode_score)) {
      best_decode_instance = decode_instance;
      best_decode_score = score;
    }
    ++schedulable_decode_count;
  }

  if (best_decode_instance.empty()) {
    LOG(ERROR) << "No schedulable decode instance found for failover request!";
    return false;
  }

  request->routing.prefill_name = best_prefill_instance;
  request->routing.decode_name = best_decode_instance;
  if (!has_preselected_prefill && schedulable_prefill_count > 0) {
    next_prefill_index_ =
        (next_prefill_index_ + best_prefill_score.order + 1) %
        prefill_index_.size();
  }
  next_decode_index_ =
      (next_decode_index_ + best_decode_score.order + 1) %
      decode_index_.size();

  if (request->failover.runtime.estimated_total_recovery_cost_ms > 0) {
    request->estimated_ttft =
        request->failover.runtime.estimated_total_recovery_cost_ms;
  } else if (request->failover.runtime.type != FailoverType::PREFILL_CRASH) {
    auto& time_predictor = get_time_predictor(best_prefill_instance);
    request->estimated_ttft = NormalizeEstimatedLatencyMs(
        time_predictor.predict_ttft(request->token_ids.size()),
        FLAGS_target_ttft);
  }
  request_metrics_[best_prefill_instance].estimated_prefill_time +=
      request->estimated_ttft;

  return true;
}

bool InstanceMgr::plan_failover_prefill_assignments(
    const std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return true;
  }
  if (options_.load_balance_policy() != "SLO_AWARE") {
    return true;
  }

  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);
  std::vector<FailoverPartitionTarget> targets;
  targets.reserve(prefill_index_.size());
  for (size_t i = 0; i < prefill_index_.size(); ++i) {
    const auto& prefill_instance = prefill_index_[i];
    auto instance_it = instances_.find(prefill_instance);
    if (instance_it == instances_.end() ||
        !is_instance_schedulable(instance_it->second)) {
      continue;
    }
    const auto& metrics = request_metrics_[prefill_instance];
    FailoverPartitionTarget target;
    target.name = prefill_instance;
    target.base_compute_load_ms = metrics.estimated_prefill_time;
    target.order =
        RotatingCandidateOrder(i, next_prefill_index_, prefill_index_.size());
    targets.push_back(target);
  }

  if (targets.empty()) {
    LOG(ERROR) << "No schedulable prefill or default instance found for "
               << "failover batch!";
    return false;
  }

  std::vector<FailoverPartitionItem> items;
  items.reserve(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    if (request == nullptr) {
      continue;
    }
    FailoverPartitionItem item;
    item.index = i;
    item.communication_cost_ms =
        request->failover.runtime.estimated_restore_cost_ms;
    item.compute_cost_ms =
        request->failover.runtime.estimated_recompute_cost_ms;
    items.push_back(item);
  }

  const auto assignments = PlanFailoverRecoveryPartition(items, targets);
  size_t max_assigned_order = 0;
  bool saw_assignment = false;
  for (size_t request_pos = 0; request_pos < requests.size(); ++request_pos) {
    const auto& request = requests[request_pos];
    if (request == nullptr || request_pos >= assignments.size()) {
      continue;
    }
    const size_t target_pos = assignments[request_pos];
    if (target_pos >= targets.size()) {
      LOG(ERROR) << "Failover partition did not assign request, request_id="
                 << request->service_request_id;
      return false;
    }
    request->routing.prefill_name = targets[target_pos].name;
    request->estimated_ttft =
        request->failover.runtime.estimated_total_recovery_cost_ms;
    max_assigned_order = std::max(max_assigned_order, targets[target_pos].order);
    saw_assignment = true;
    DLOG(INFO) << "failover_prefill_partition_assignment"
               << " request_id=" << request->service_request_id
               << " prefill=" << request->routing.prefill_name
               << " communication_cost_ms="
               << request->failover.runtime.estimated_restore_cost_ms
               << " compute_cost_ms="
               << request->failover.runtime.estimated_recompute_cost_ms
               << " recovery_cost_ms="
               << request->failover.runtime.estimated_total_recovery_cost_ms;
  }

  if (saw_assignment && !prefill_index_.empty()) {
    next_prefill_index_ =
        (next_prefill_index_ + max_assigned_order + 1) % prefill_index_.size();
  }
  return true;
}

void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  if (count_schedulable_instances(instances_, prefill_index_) <= 1) {
    // Ensure there is at least one prefill instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from prefill_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to decode_index_
  instances_[instance_name].current_type = InstanceType::DECODE;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  if (count_schedulable_instances(instances_, decode_index_) <= 1) {
    // Ensure there is at least one decode instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from decode_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to prefill_index
  instances_[instance_name].current_type = InstanceType::PREFILL;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

bool InstanceMgr::call_link_instance(const std::string& target_rpc_addr,
                                     const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for LinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& ip : peer_info.device_ips) {
    req.add_device_ips(ip);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  xllm::proto::Status res;
  stub.LinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "LinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::call_unlink_instance(const std::string& target_rpc_addr,
                                       const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for UnlinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& ip : peer_info.device_ips) {
    req.add_device_ips(ip);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  xllm::proto::Status res;
  stub.UnlinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UnlinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::register_instance(const std::string& name,
                                    InstanceMetaInfo& info) {
  info.runtime_state = InstanceRuntimeState::ACTIVE;
  info.latest_timestamp = current_time_ms();
  info.name = name;

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(ERROR) << "Instance is already registered, instance_name: " << name;
      return false;
    }
  }

  std::shared_ptr<brpc::Channel> channel;
  if (!init_brpc_channel(name, &channel)) {
    LOG(ERROR) << "create channel fail: " << name;
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(WARNING) << "Instance registered concurrently during channel init: "
                   << name;
      return false;
    }
    cached_channels_[name] = std::move(channel);
  }

  add_instance_resources(name, info);

  std::vector<std::pair<std::string, InstanceMetaInfo>> link_ops;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (!gather_link_operations(info, &link_ops)) {
      remove_instance_resources(name);
      return false;
    }
  }

  if (!run_link_operations(link_ops)) {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    remove_instance_resources(name);
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    add_instance_to_index(name, info);
    instances_.insert(std::make_pair(name, info));
  }
  return true;
}

void InstanceMgr::deregister_instance(
    const std::string& name,
    const std::string& expected_incarnation_id) {
  InstanceMetaInfo info;
  std::vector<std::pair<std::string, InstanceMetaInfo>> unlink_ops;
  brpc::StreamId failover_session_stream_id = brpc::INVALID_STREAM_ID;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      LOG(ERROR) << "Instance is not registered, instance_name: " << name;
      return;
    }

    if (!expected_incarnation_id.empty() &&
        it->second.incarnation_id != expected_incarnation_id) {
      LOG(INFO) << "Skip deregistering stale incarnation, instance_name: "
                << name
                << ", current incarnation_id: " << it->second.incarnation_id
                << ", expected incarnation_id: " << expected_incarnation_id;
      return;
    }

    info = it->second;
    clear_suspect_instance(name, info.incarnation_id);
    auto session_it = failover_sessions_.find(name);
    if (session_it != failover_sessions_.end() &&
        session_it->second.incarnation_id == info.incarnation_id) {
      failover_session_stream_id = session_it->second.stream_id;
      failover_sessions_.erase(session_it);
    }
    gather_unlink_operations(name, info, &unlink_ops);
  }

  if (failover_session_stream_id != brpc::INVALID_STREAM_ID) {
    brpc::StreamClose(failover_session_stream_id);
  }

  for (const auto& op : unlink_ops) {
    call_unlink_instance(op.first, op.second);
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      return;
    }
    remove_instance_from_index(name, it->second);
  }

  const InstanceType cleanup_type = get_cleanup_type(info);
  const size_t failover_count = scheduler_->clear_requests_on_failed_instance(
      name, info.incarnation_id, cleanup_type);
  if (failover_count > 0) {
    LOG(INFO) << "failover_rehandle_trigger"
              << " instance=" << name
              << " incarnation_id=" << info.incarnation_id
              << " count=" << failover_count
              << " reason=deregister_instance";
    scheduler_->rehandle_removed_request();
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      return;
    }
    remove_instance_resources(name);
    instances_.erase(it);
  }
  LOG(INFO) << "delete instance: " << name;
}

void InstanceMgr::add_instance_resources(const std::string& name,
                                         const InstanceMetaInfo& info) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  time_predictors_.insert_or_assign(
      name, TimePredictor(info.ttft_profiling_data, info.tpot_profiling_data));

  request_metrics_.insert_or_assign(name, RequestMetrics());
}

void InstanceMgr::remove_instance_resources(const std::string& name) {
  // Caller must hold cluster_mutex_ (cached_channels_ is L1).
  cached_channels_.erase(name);
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
  time_predictors_.erase(name);
  request_metrics_.erase(name);
  latency_metrics_.erase(name);
  updated_metrics_.erase(name);
  removed_instance_.insert(name);
  load_metrics_.erase(name);
}

bool InstanceMgr::gather_link_operations(
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  switch (info.type) {
    case InstanceType::DEFAULT:
      break;
    case InstanceType::PREFILL: {
      for (auto& d_name : decode_index_) {
        out_ops->emplace_back(instances_[d_name].rpc_address, info);
      }
      break;
    }
    case InstanceType::DECODE: {
      for (auto& p_name : prefill_index_) {
        out_ops->emplace_back(info.rpc_address, instances_[p_name]);
      }
      break;
    }
    case InstanceType::MIX: {
      for (const auto& [peer_name, peer_info] : instances_) {
        if (peer_name == info.name) {
          continue;
        }
        out_ops->emplace_back(info.rpc_address, peer_info);
      }
      break;
    }
    default:
      LOG(WARNING) << "Unknown InstanceType: " << int(info.type);
      return false;
  }
  return true;
}

bool InstanceMgr::run_link_operations(
    const std::vector<std::pair<std::string, InstanceMetaInfo>>& ops) {
  for (size_t i = 0; i < ops.size(); ++i) {
    if (!call_link_instance(ops[i].first, ops[i].second)) {
      LOG(ERROR) << "Fail to link instance during registration, op index " << i;
      for (size_t j = 0; j < i; ++j) {
        call_unlink_instance(ops[j].first, ops[j].second);
      }
      return false;
    }
  }
  return true;
}

void InstanceMgr::gather_unlink_operations(
    const std::string& name,
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  if (info.type == InstanceType::PREFILL) {
    for (auto& d_name : decode_index_) {
      out_ops->emplace_back(instances_[d_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::DECODE) {
    for (auto& p_name : prefill_index_) {
      out_ops->emplace_back(instances_[p_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::MIX) {
    for (const auto& [peer_name, peer_info] : instances_) {
      if (peer_name == name) {
        continue;
      }
      out_ops->emplace_back(peer_info.rpc_address, info);
    }
  }
}

void InstanceMgr::add_instance_to_index(const std::string& name,
                                        InstanceMetaInfo& info) {
  switch (info.type) {
    case InstanceType::DEFAULT:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new default instance, instance name : " << name;
      break;
    case InstanceType::PREFILL:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new prefill instance, instance name : " << name;
      break;
    case InstanceType::DECODE:
      info.instance_index = decode_index_.size();
      decode_index_.emplace_back(name);
      LOG(INFO) << "Register a new decode instance, instance name : " << name;
      break;
    case InstanceType::MIX:
      if (decode_index_.size() > 0) {
        info.instance_index = prefill_index_.size();
        info.current_type = InstanceType::PREFILL;
        prefill_index_.emplace_back(name);
        LOG(INFO) << "Register a new prefill instance, instance name : "
                  << name;
      } else {
        info.instance_index = decode_index_.size();
        info.current_type = InstanceType::DECODE;
        decode_index_.emplace_back(name);
        LOG(INFO) << "Register a new decode instance, instance name : " << name;
      }
      break;
    default:
      break;
  }
}

void InstanceMgr::remove_instance_from_index(const std::string& name,
                                             const InstanceMetaInfo& info) {
  uint64_t index = info.instance_index;
  if (index == -1) return;

  auto remove_from_vec = [&](std::vector<std::string>& vec) {
    if (index >= vec.size()) return;
    std::swap(vec[index], vec.back());
    instances_[vec[index]].instance_index = index;
    vec.pop_back();
  };

  switch (info.type) {
    case InstanceType::DEFAULT:
    case InstanceType::PREFILL:
      remove_from_vec(prefill_index_);
      break;
    case InstanceType::DECODE:
      remove_from_vec(decode_index_);
      break;
    case InstanceType::MIX:
      if (info.current_type == InstanceType::PREFILL) {
        remove_from_vec(prefill_index_);
      } else {
        remove_from_vec(decode_index_);
      }
      break;
    default:
      break;
  }
}

bool InstanceMgr::has_available_instances() const {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);

  bool has_default = false;
  bool has_prefill = false;
  bool has_decode = false;
  bool has_mix_as_prefill = false;
  bool has_mix_as_decode = false;

  for (const auto& [name, info] : instances_) {
    if (!is_instance_schedulable(info)) continue;

    switch (info.type) {
      case InstanceType::DEFAULT:
        has_default = true;
        break;
      case InstanceType::PREFILL:
        has_prefill = true;
        break;
      case InstanceType::DECODE:
        has_decode = true;
        break;
      case InstanceType::MIX:
        if (info.current_type == InstanceType::PREFILL) {
          has_mix_as_prefill = true;
        } else if (info.current_type == InstanceType::DECODE) {
          has_mix_as_decode = true;
        }
        break;
      default:
        break;
    }

    // Early exit: any satisfied condition is enough
    if (has_default || (has_prefill && has_decode) ||
        (has_mix_as_prefill && has_mix_as_decode)) {
      return true;
    }
  }

  return has_default || (has_prefill && has_decode) ||
         (has_mix_as_prefill && has_mix_as_decode);
}

uint32_t InstanceMgr::get_offload_batch_size(const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> guard(metrics_mutex_);
  auto it = load_metrics_.find(instance_name);
  if (it != load_metrics_.end()) {
    return it->second.offload_batch_size;
  }
  return UINT32_MAX;
}

}  // namespace xllm_service
