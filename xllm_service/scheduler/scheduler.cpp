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

#include "scheduler/scheduler.h"

#include <absl/time/clock.h>

#include <algorithm>
#include <utility>

#include "common/metrics.h"
#include "common/utils.h"
#include "common/xllm/status.h"
#include "failover/debug.h"
#include "failover/replay.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "loadbalance_policy/slo_aware_policy.h"
#include "scheduler/decode_offload_tracking.h"
#include "scheduler/token_latency_metrics.h"
#include "tokenizer/tokenizer_factory.h"

namespace {
constexpr int32_t kHeartbeatInterval = 3;  // in seconds

constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";

bool RouteUsesFailedInstance(const xllm_service::Request& request,
                             const std::string& instance_name,
                             xllm_service::InstanceType type) {
  const bool check_prefill = type == xllm_service::InstanceType::DEFAULT ||
                             type == xllm_service::InstanceType::PREFILL;
  const bool check_decode = type == xllm_service::InstanceType::DEFAULT ||
                            type == xllm_service::InstanceType::DECODE;
  return (check_prefill && request.routing.prefill_name == instance_name) ||
         (check_decode && request.routing.decode_name == instance_name);
}
}  // namespace

namespace xllm_service {

Scheduler::Scheduler(const Options& options) : options_(options) {
  tokenizer_ = TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                                  &tokenizer_args_);
  chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);

  const std::string etcd_username =
      utils::get_optional_string_env(kEtcdUsernameEnvVar).value_or("");
  const std::string etcd_password =
      utils::get_optional_string_env(kEtcdPasswordEnvVar).value_or("");
  const bool has_etcd_auth_user = !etcd_username.empty();
  const bool has_etcd_auth_password = !etcd_password.empty();
  if (has_etcd_auth_user != has_etcd_auth_password) {
    LOG(FATAL) << "Both " << kEtcdUsernameEnvVar << " and "
               << kEtcdPasswordEnvVar << " must be set together.";
  }
  if (has_etcd_auth_user) {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                etcd_username,
                                                etcd_password,
                                                options_.etcd_namespace());
  } else {
    etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr(),
                                                options_.etcd_namespace());
  }

  if (!register_current_service()) {
    LOG(FATAL)
        << "Failed to register current xllm_service in etcd, service_name: "
        << options_.service_name();
  }

  auto handle_xservice = std::bind(&Scheduler::handle_xservice_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
  etcd_client_->add_watch(ETCD_XSERVICE_KEY_PREFIX, handle_xservice);

  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, options_.service_name(), kHeartbeatInterval);
    LOG(INFO) << "Set current service as master!";
  }

  instance_mgr_ = std::make_shared<InstanceMgr>(
      options, etcd_client_, is_master_service_, this);

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options.load_balance_policy() == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_, global_kvcache_mgr_);
  } else if (options.load_balance_policy() == "SLO_AWARE") {
    lb_policy_ = std::make_unique<SloAwarePolicy>(options, instance_mgr_);
  } else {
    lb_policy_ = std::make_unique<RoundRobin>(instance_mgr_);
  }

  if (options_.enable_failover_recovery_dump()) {
    FailoverRecoveryDumpConfig dump_config;
    dump_config.enabled = true;
    dump_config.file_path = options_.failover_recovery_dump_path();
    dump_config.max_queue_size = static_cast<size_t>(
        std::max<int32_t>(1, options_.failover_recovery_dump_max_queue_size()));
    failover_recovery_dumper_ =
        std::make_unique<FailoverRecoveryDumper>(std::move(dump_config));
  }

  if (is_master_service_) {
    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);
  } else {
    auto handle_master = std::bind(&Scheduler::handle_master_service_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
    etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, handle_master);
  }
}

Scheduler::~Scheduler() {
  failover_recovery_dumper_.reset();
  etcd_client_->stop_watch();
}

bool Scheduler::schedule(std::shared_ptr<Request> request) {
  // apply chat template
  if (request->messages.size() > 0) {
    if (chat_template_ == nullptr) {
      LOG(ERROR) << "Chat template has not configured.";
      return false;
    }

    const std::vector<JsonTool> empty_tools;
    const std::vector<JsonTool>& tools_for_template =
        request->tool_choice == "none" ? empty_tools : request->tools;
    auto prompt = chat_template_->apply(
        request->messages, tools_for_template, request->chat_template_kwargs);
    if (!prompt.has_value()) {
      LOG(ERROR) << "Failed to construct prompt from messages";
      return false;
    }
    request->prompt = prompt.value();
  }

  // encode prompt
  if (request->prompt.size() != 0) {
    if (!get_tls_tokenizer()->encode(request->prompt, &request->token_ids)) {
      LOG(ERROR) << "Encode prompt failed: " << request->prompt;
      return false;
    }
  }

  auto ret = lb_policy_->select_instances_pair(request);
  if (!ret) {
    return false;
  }

  if (!instance_mgr_->bind_request_instance_incarnations(request)) {
    LOG(ERROR) << "Failed to bind request to instance incarnation ids. "
               << request->routing.debug_string();
    return false;
  }
  DLOG(INFO) << request->routing.debug_string();

  // update request metrics
  if (request->prompt.size() != 0) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  return true;
}

bool Scheduler::schedule_failover(std::shared_ptr<Request> request) {
  auto ret = instance_mgr_->select_instance_pair_on_failover(request);
  if (!ret) {
    return false;
  }

  if (!instance_mgr_->bind_request_instance_incarnations(request)) {
    LOG(ERROR)
        << "Failed to bind failover request to instance incarnation ids. "
        << request->routing.debug_string();
    return false;
  }
  DLOG(INFO) << request->routing.debug_string();

  if (request->prompt.size() != 0 || !request->token_ids.empty()) {
    instance_mgr_->update_request_metrics(request, RequestAction::SCHEDULE);
  }

  return true;
}

std::shared_ptr<brpc::Channel> Scheduler::get_channel(
    const std::string& target_name) {
  return instance_mgr_->get_channel(target_name);
}

void Scheduler::update_master_service_heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));

    global_kvcache_mgr_->upload_kvcache();

    instance_mgr_->upload_load_metrics();
  }
}

bool Scheduler::register_current_service() {
  const std::string service_key =
      ETCD_XSERVICE_KEY_PREFIX + options_.service_name();

  if (etcd_client_->set(
          service_key, options_.service_name(), kHeartbeatInterval)) {
    return true;
  }

  LOG(ERROR) << "Service key already exists, registration failed: "
             << service_key
             << ". Please ensure service_name is unique across xllm_service "
                "instances.";
  return false;
}

bool Scheduler::handle_instance_heartbeat(const proto::HeartbeatRequest* req) {
  if (exited_) {
    return false;
  }
  if (!instance_mgr_->record_instance_heartbeat(req->name(),
                                                req->incarnation_id())) {
    return false;
  }
  global_kvcache_mgr_->record_updated_kvcaches(req->name(), req->cache_event());
  instance_mgr_->record_load_metrics_update(req->name(), req->load_metrics());
  instance_mgr_->update_latency_metrics(req->name(), req->latency_metrics());
  return true;
}

bool Scheduler::open_failover_session(const proto::FailoverSessionRequest* req,
                                      brpc::Controller* cntl) {
  if (exited_ || req == nullptr || cntl == nullptr) {
    return false;
  }
  return instance_mgr_->open_failover_session(*req, cntl);
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  if (etcd_client_->set(ETCD_MASTER_SERVICE_KEY,
                        options_.service_name(),
                        kHeartbeatInterval)) {
    is_master_service_ = true;

    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);

    global_kvcache_mgr_->set_as_master();
    instance_mgr_->set_as_master();
  }
}

void Scheduler::handle_xservice_watch(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() != etcd::Event::EventType::DELETE_) {
      continue;
    }

    std::string deleted_service;
    if (event.has_prev_kv()) {
      deleted_service = event.prev_kv().key().substr(prefix_len);
    } else if (event.has_kv()) {
      deleted_service = event.kv().key().substr(prefix_len);
    }

    if (deleted_service.empty()) {
      continue;
    }

    if (deleted_service == options_.service_name()) {
      LOG(INFO) << "Current xllm_service registration expired, re-registering";
      register_current_service();
      continue;
    }

    if (deleted_service == ETCD_MASTER_SERVICE_NAME) {
      continue;
    }

    if (!is_master_service_) {
      continue;
    }

    LOG(INFO) << "Detected xllm_service offline: " << deleted_service;
  }
}

InstanceMetaInfo Scheduler::get_instance_info(
    const std::string& instance_name) {
  return instance_mgr_->get_instance_info(instance_name);
}

std::vector<std::string> Scheduler::get_static_decode_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_decode_list(instance_name);
}

std::vector<std::string> Scheduler::get_static_prefill_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_prefill_list(instance_name);
}

Tokenizer* Scheduler::get_tls_tokenizer() {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->service_request_id) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->service_request_id;
      return false;
    }

    request->latest_generate_time = absl::Now();
    request->callback_attempt = request->failover.runtime.attempt;
    auto tools_for_parse =
        (request->tool_choice == "none" ? std::vector<JsonTool>{}
                                        : request->tools);
    auto tool_call_parser_pref = options_.tool_call_parser();
    auto reasoning_parser_pref = options_.reasoning_parser();
    std::shared_ptr<ChatStreamParseState> stream_state;
    if (request->stream) {
      stream_state = response_handler_.create_chat_stream_parse_state(
          tools_for_parse,
          request->model,
          tool_call_parser_pref,
          reasoning_parser_pref);
    }

    request->call_data = call_data;
    auto weak_request = std::weak_ptr<Request>(request);
    request->output_callback =
        [this,
         weak_request,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         tools = std::move(tools_for_parse),
         tool_call_parser = std::move(tool_call_parser_pref),
         reasoning_parser = std::move(reasoning_parser_pref),
         stream_state = std::move(stream_state),
         expected_attempt = request->callback_attempt.load(),
         created_time = absl::ToUnixSeconds(request->latest_generate_time)](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (auto request = weak_request.lock()) {
        if (request->callback_attempt.load() != expected_attempt) {
          return true;
        }
      } else {
        return false;
      }

      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      bool ok = true;
      if (stream) {
        ok = response_handler_.send_delta_to_client(call_data,
                                                    include_usage,
                                                    created_time,
                                                    model,
                                                    req_output,
                                                    stream_state);
      } else if (!req_output.finished_on_prefill_instance) {
        // for non-stream request, only send final result from decode instance
        ok = response_handler_.send_result_to_client(call_data,
                                                     created_time,
                                                     model,
                                                     req_output,
                                                     tools,
                                                     tool_call_parser,
                                                     reasoning_parser);
      }

      // Replay-buffer accumulation has moved into Scheduler::handle_generation
      // under request_mutex_. Doing it here used to race with
      // MarkRequestForFailover and could either drop tokens that were already
      // streamed to the client, or leak tokens from a stale attempt into the
      // next attempt's committed buffer. The first-line attempt check above
      // is preserved so we still silently drop client-side sends from a
      // stale attempt; lifecycle handling stays in the outer
      // output_threadpool closure.
      return ok;
    };
    requests_.emplace(request->service_request_id, request);
    GAUGE_SET(active_service_requests, requests_.size());
    COUNTER_INC(server_request_in_total);
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    if (remote_requests_output_thread_map_.find(request->service_request_id) ==
        remote_requests_output_thread_map_.end()) {
      remote_requests_output_thread_map_[request->service_request_id] =
          next_thread_idx;
      next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
      GAUGE_SET(active_output_thread_mappings,
                remote_requests_output_thread_map_.size());
    }
  }

  return true;
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    if (requests_.find(request->service_request_id) != requests_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << request->service_request_id;
      return false;
    }

    request->latest_generate_time = absl::Now();
    request->callback_attempt = request->failover.runtime.attempt;

    request->call_data = call_data;
    auto weak_request = std::weak_ptr<Request>(request);
    request->output_callback =
        [this,
         weak_request,
         call_data,
         model = request->model,
         stream = request->stream,
         include_usage = request->include_usage,
         expected_attempt = request->callback_attempt.load(),
         created_time = absl::ToUnixSeconds(request->latest_generate_time)](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (auto request = weak_request.lock()) {
        if (request->callback_attempt.load() != expected_attempt) {
          return true;
        }
      } else {
        return false;
      }

      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      bool ok = true;
      if (stream) {
        ok = response_handler_.send_delta_to_client(
            call_data, include_usage, created_time, model, req_output);
      } else if (!req_output.finished_on_prefill_instance) {
        // for non-stream request, only send final result from decode instance
        ok = response_handler_.send_result_to_client(
            call_data, created_time, model, req_output);
      }

      // Replay-buffer accumulation has moved into Scheduler::handle_generation
      // under request_mutex_ for the same reasons explained in the chat
      // callback above (avoid the send-vs-accumulate race against
      // MarkRequestForFailover).
      return ok;
    };
    requests_.emplace(request->service_request_id, request);
    GAUGE_SET(active_service_requests, requests_.size());
    COUNTER_INC(server_request_in_total);
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    if (remote_requests_output_thread_map_.find(request->service_request_id) ==
        remote_requests_output_thread_map_.end()) {
      remote_requests_output_thread_map_[request->service_request_id] =
          next_thread_idx;
      next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
      GAUGE_SET(active_output_thread_mappings,
                remote_requests_output_thread_map_.size());
    }
  }

  return true;
}

void Scheduler::finish_request(const std::string& service_request_id,
                               bool error) {
  std::shared_ptr<Request> request;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it != requests_.end()) {
      request = it->second;
      requests_.erase(it);
      GAUGE_SET(active_service_requests, requests_.size());
    }
  }

  if (request != nullptr) {
    if (error) {
      instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    } else {
      instance_mgr_->update_request_metrics(request,
                                            RequestAction::FINISH_DECODE);
    }
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
    GAUGE_SET(active_output_thread_mappings,
              remote_requests_output_thread_map_.size());
  }
}

size_t Scheduler::clear_requests_on_failed_instance(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType type) {
  std::vector<std::string> cleared_request_ids;
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    for (auto it = requests_.begin(); it != requests_.end();) {
      auto failover_type = MatchFailedInstanceForFailover(
          *it->second, instance_name, incarnation_id, type);
      if (failover_type.has_value()) {
        MarkRequestForFailover(it->second.get(),
                               *failover_type,
                               absl::Now(),
                               options_.block_size(),
                               failover_recovery_config_);
        auto service_request_id = it->second->service_request_id;
        LOG(INFO)
            << "failover_request_marked"
            << " request_id=" << service_request_id
            << " failed_instance=" << instance_name
            << " incarnation_id=" << incarnation_id
            << " cleanup_type=" << static_cast<int32_t>(type)
            << " failover_type=" << FailoverTypeName(*failover_type)
            << " recovery_cost_ms="
            << it->second->failover.runtime.estimated_total_recovery_cost_ms
            << " restore_cost_ms="
            << it->second->failover.runtime.estimated_restore_cost_ms
            << " recompute_cost_ms="
            << it->second->failover.runtime.estimated_recompute_cost_ms
            << " generated_tokens=" << it->second->num_generated_tokens
            << " prompt_tokens=" << it->second->token_ids.size();
        cleared_request_ids.emplace_back(service_request_id);
        it = requests_.erase(it);
        GAUGE_SET(active_service_requests, requests_.size());
      } else {
        if (RouteUsesFailedInstance(*it->second, instance_name, type)) {
          LOG(WARNING)
              << "failover_request_route_matched_but_not_cleared"
              << " request_id=" << it->second->service_request_id
              << " failed_instance=" << instance_name
              << " incarnation_id=" << incarnation_id
              << " cleanup_type=" << static_cast<int32_t>(type)
              << " prefill=" << it->second->routing.prefill_name
              << " prefill_incarnation=" << it->second->prefill_incarnation_id
              << " decode=" << it->second->routing.decode_name
              << " decode_incarnation=" << it->second->decode_incarnation_id
              << " prefill_finished=" << it->second->prefill_stage_finished
              << " generated_tokens=" << it->second->num_generated_tokens
              << " failover_attempt=" << it->second->failover.runtime.attempt
              << " failover_type="
              << FailoverTypeName(it->second->failover.runtime.type);
        }
        ++it;
      }
    }
  }

  if (!cleared_request_ids.empty()) {
    {
      std::lock_guard<std::mutex> guard(thread_map_mutex_);
      for (const auto& service_request_id : cleared_request_ids) {
        remote_requests_output_thread_map_.erase(service_request_id);
      }
      GAUGE_SET(active_output_thread_mappings,
                remote_requests_output_thread_map_.size());
    }
    for (const auto& service_request_id : cleared_request_ids) {
      add_removed_request(service_request_id);
    }
    LOG(INFO) << "failover_requests_queued"
              << " failed_instance=" << instance_name
              << " incarnation_id=" << incarnation_id
              << " cleanup_type=" << static_cast<int32_t>(type)
              << " count=" << cleared_request_ids.size();
  }
  return cleared_request_ids.size();
}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  bool finished_on_prefill_instance =
      request_output.finished_on_prefill_instance;
  const std::string& service_request_id = request_output.service_request_id;
  bool status_error =
      request_output.status.has_value() && !request_output.status.value().ok();

  OutputCallback cb;
  std::shared_ptr<Request> request;
  int32_t expected_attempt = 0;
  bool client_disconnected = false;
  {
    std::lock_guard<std::mutex> guard(request_mutex_);
    auto it = requests_.find(service_request_id);
    if (it == requests_.end()) {
      LOG(ERROR) << "Can not found the callback for the received request "
                    "output, request id is: "
                 << service_request_id;
      return false;
    }
    request = it->second;
    cb = request->output_callback;
    // Snapshot the attempt this response belongs to. The output_threadpool
    // task captures this and re-checks at execution time so a late-arriving
    // response from a dead instance (especially one carrying finished=true
    // or status=error) cannot prematurely terminate the next attempt that
    // record_new_request has since installed.
    expected_attempt = request->callback_attempt;

    // Accumulate the replay buffer synchronously under request_mutex_, BEFORE
    // dispatching the callback to the output_threadpool. MarkRequestForFailover
    // also bumps callback_attempt and clears the active request under this
    // same lock, so this section is mutually exclusive with failover detection.
    // Doing the accumulate here means:
    //   - "client received a delta" (in the cb on the threadpool) and
    //     "committed_output_token_ids grew" (right here) cannot diverge:
    //     either failover mark wins and this response is fully ignored before
    //     it ever reaches the threadpool, or this attempt's accumulation
    //     completes here and any later failover sees the up-to-date buffer.
    //   - The output_callback's previous "double attempt check around
    //     accumulate" race window (where a callback that already sent to the
    //     client could be denied accumulation by a later bump, leaving the
    //     committed buffer one token short and shifting all subsequent block
    //     hashes on rehandle) is closed.
    if (!status_error) {
      AccumulateReplayState(request.get(), request_output);
    }

    // check client connection
    if (request->call_data->is_disconnected()) {
      LOG(INFO) << "Client has disconnected and the request will be cancelled, "
                   "request id: "
                << service_request_id;
      requests_.erase(it);
      GAUGE_SET(active_service_requests, requests_.size());
      client_disconnected = true;
    }
  }

  if (client_disconnected) {
    instance_mgr_->update_request_metrics(request, RequestAction::CANCEL);
    {
      std::lock_guard<std::mutex> guard(thread_map_mutex_);
      remote_requests_output_thread_map_.erase(service_request_id);
      GAUGE_SET(active_output_thread_mappings,
                remote_requests_output_thread_map_.size());
    }
    finish_request_context(service_request_id);
    return false;
  }

  if (!status_error) {
    // no error, update instance request metrics
    update_request_metrics(request, finished_on_prefill_instance);
    update_token_latency_metrics(request, finished_on_prefill_instance);
  }

  size_t req_thread_idx = -1;
  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    auto it = remote_requests_output_thread_map_.find(service_request_id);
    if (it == remote_requests_output_thread_map_.end()) {
      LOG(ERROR) << "Can not found the thread for the received request output, "
                    "request id is: "
                 << service_request_id;
      return false;
    }
    req_thread_idx = it->second;
  }

  output_threadpools_[req_thread_idx].schedule(
      [this,
       service_request_id,
       cb,
       status_error,
       expected_attempt,
       request_output = std::move(request_output)]() mutable {
        const bool cb_ok = cb(request_output);
        // Recheck the request's current attempt. A response that belongs to
        // a stale attempt (e.g. the dead instance flushed a final
        // finished=true / status=error frame just before disconnect, and
        // failover has since re-dispatched the request) must not terminate
        // the live attempt. The inner cb has already suppressed the
        // user-visible work (send_delta / accumulate) via its own attempt
        // check; this outer recheck additionally suppresses the lifecycle
        // transition that the inner cb cannot reach.
        bool is_current_attempt = false;
        {
          std::lock_guard<std::mutex> guard(request_mutex_);
          auto it = requests_.find(service_request_id);
          if (it != requests_.end() &&
              it->second->callback_attempt == expected_attempt) {
            is_current_attempt = true;
          }
        }
        if (!is_current_attempt) {
          DLOG(INFO) << "stale_output_dropped"
                     << " request_id=" << service_request_id
                     << " expected_attempt=" << expected_attempt
                     << " finished=" << request_output.finished
                     << " status_error=" << status_error;
          return;
        }
        if (!cb_ok || status_error) {
          finish_request(service_request_id, true);
          finish_request_context(service_request_id);
          return;
        }
        if (request_output.finished) {
          finish_request(service_request_id);
          finish_request_context(service_request_id);
          return;
        }
      });

  return true;
}

void Scheduler::update_request_metrics(std::shared_ptr<Request> request,
                                       bool finished_on_prefill_instance) {
  request->num_generated_tokens += 1;
  if (finished_on_prefill_instance) {
    request->prefill_stage_finished = true;
    // update instance request metrics for prefill finished request
    instance_mgr_->update_request_metrics(request,
                                          RequestAction::FINISH_PREFILL);
  } else {
    if (!request->prefill_stage_finished) {
      request->prefill_stage_finished = true;
      instance_mgr_->update_request_metrics(request,
                                            RequestAction::FINISH_PREFILL);
    }
    // update instance request metrics
    MaybeRecordFirstDecodeOffloadBatchSize(
        request.get(),
        finished_on_prefill_instance,
        [this](const std::string& instance_name) {
          return instance_mgr_->get_offload_batch_size(instance_name);
        });
    instance_mgr_->update_request_metrics(request, RequestAction::GENERATE);
  }
}

void Scheduler::update_token_latency_metrics(
    std::shared_ptr<Request> request,
    bool finished_on_prefill_instance) {
  FailoverRecoveryDumper* recovery_dumper = nullptr;
  if (failover_recovery_dumper_ != nullptr &&
      failover_recovery_dumper_->enabled()) {
    recovery_dumper = failover_recovery_dumper_.get();
  }
  ObserveTokenLatencyMetrics(request.get(),
                             finished_on_prefill_instance,
                             absl::Now(),
                             recovery_dumper);
}

bool Scheduler::has_available_instances() const {
  return instance_mgr_->has_available_instances();
}

void Scheduler::register_request_rehandle_callback(RequestRehandleCallback cb) {
  failover_coordinator_.register_request_rehandle_callback(std::move(cb));
  failover_coordinator_.register_batch_prepare_callback(
      [this](const std::vector<std::shared_ptr<RequestContext>>& contexts) {
        return prepare_failover_rehandle_batch(contexts);
      });
}

bool Scheduler::record_new_request_context(
    std::shared_ptr<RequestContext> req_context) {
  std::lock_guard<std::mutex> guard(request_context_mutex_);
  if (request_contexts_.find(req_context->request()->service_request_id) !=
      request_contexts_.end()) {
    LOG(ERROR)
        << "The request context ID already exists. Requests with the same ID "
           "are not allowed. "
        << req_context->request()->service_request_id;
    return false;
  }
  request_contexts_[req_context->request()->service_request_id] = req_context;
  GAUGE_SET(active_request_contexts, request_contexts_.size());
  return true;
}

void Scheduler::finish_request_context(const std::string& service_request_id) {
  DLOG(INFO) << "Scheduler::finish_request_context for request id: "
             << service_request_id;
  {
    std::lock_guard<std::mutex> guard(request_context_mutex_);
    const auto erased = request_contexts_.erase(service_request_id);
    GAUGE_SET(active_request_contexts, request_contexts_.size());
    if (erased == 0) {
      LOG(WARNING) << "request_context_finish_missing"
                   << " request_id=" << service_request_id
                   << " active_contexts=" << request_contexts_.size();
    }
  }
}

void Scheduler::add_removed_request(std::string request) {
  failover_coordinator_.enqueue_removed_request(std::move(request));
}

void Scheduler::rehandle_removed_request() {
  RequestContextMap request_contexts_copy;
  {
    std::lock_guard<std::mutex> guard(request_context_mutex_);
    request_contexts_copy = request_contexts_;
  }
  failover_coordinator_.rehandle_removed_requests(request_contexts_copy);
}

bool Scheduler::prepare_failover_rehandle_batch(
    const std::vector<std::shared_ptr<RequestContext>>& request_contexts) {
  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(request_contexts.size());
  for (const auto& request_context : request_contexts) {
    if (request_context == nullptr || request_context->request() == nullptr) {
      continue;
    }
    requests.emplace_back(request_context->request());
  }
  return instance_mgr_->plan_failover_prefill_assignments(requests);
}

}  // namespace xllm_service
