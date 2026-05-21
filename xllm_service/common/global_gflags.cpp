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

#include "common/global_gflags.h"

#include "brpc/reloadable_flags.h"

DEFINE_string(server_host,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(http_server_port, 8888, "Port for xllm http service to listen on");

DEFINE_int32(http_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(http_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(http_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(rpc_server_port, 8889, "Port for xllm rpc service to listen on");

DEFINE_int32(rpc_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(rpc_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(rpc_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_string(etcd_addr,
              "0.0.0.0:2379",
              "etcd adderss for save instance meta info");

DEFINE_string(
    etcd_namespace,
    "",
    "Optional etcd namespace prefix for all xllm-service keys, e.g. prod-a.");

DEFINE_uint32(xxh3_128bits_seed, 1024, "default XXH3 128bits Hash seed");

DEFINE_int32(port, 8888, "Port for xllm service to listen on");

DEFINE_int32(num_threads, 32, "Number of threads to process requests");

DEFINE_int32(max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(
    timeout_ms,
    -1,
    "Max duration (millisecond) of bRPC Channel. -1 means wait indefinitely.");

DEFINE_int32(connect_timeout_ms,
             -1,
             "Max duration (millisecond) of bRPC to establish connections. -1 "
             "means wait "
             "indefinitely.");

DEFINE_string(listen_addr,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_string(load_balance_policy,
              "RR",
              "Disaggregated prefill-decode policy.");

DEFINE_bool(disable_dynamic_pd_flip,
            false,
            "Disable runtime MIX role flips while keeping SLO-aware routing "
            "enabled.");

DEFINE_int32(detect_disconnected_instance_interval,
             15,
             "The interval that server detect the disconnected instance.");

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_string(tokenizer_path, "", "tokenizer config path.");

DEFINE_bool(enable_request_trace, false, "Whether to enable request trace");

DEFINE_bool(enable_failover_recovery_dump,
            false,
            "Whether to dump failover recovery records as JSONL.");

DEFINE_string(failover_recovery_dump_path,
              "trace/failover_recovery.jsonl",
              "Path of failover recovery JSONL dump file.");

DEFINE_int32(failover_recovery_dump_max_queue_size,
             4096,
             "Maximum buffered failover recovery dump records before dropping.");

BRPC_VALIDATE_GFLAG(failover_recovery_dump_max_queue_size,
                    brpc::PositiveInteger);

DEFINE_int32(max_failover_attempts,
             3,
             "Maximum number of failover redispatches per request. After this "
             "many failovers the request is finished with an error instead of "
             "being retried again. Set <= 0 to disable the cap.");

DEFINE_int32(target_ttft,
             1000,
             "Target Time to First Token (TTFT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_ttft, brpc::NonNegativeInteger);

DEFINE_int32(target_tpot,
             50,
             "Target Time Per Output Token (TPOT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_tpot, brpc::NonNegativeInteger);

DEFINE_string(reasoning_parser,
              "",
              "Specify the reasoning parser for handling reasoning "
              "interactions(e.g. auto, glm45, glm47, qwen3, deepseek-r1).");

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions(e.g. "
              "auto, qwen25, qwen3, kimi_k2, deepseekv3, glm45, glm47).");

DEFINE_int32(readiness_check_interval_s,
             3,
             "Interval in seconds to check for available instance groups "
             "before starting and during runtime of the HTTP service.");

BRPC_VALIDATE_GFLAG(readiness_check_interval_s, brpc::PositiveInteger);
