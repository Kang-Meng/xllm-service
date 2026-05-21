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

#include "metrics.h"

DEFINE_COUNTER(server_request_in_total,
               "Total number of request that server received");
DEFINE_COUNTER(failover_rehandle_total,
               "Total number of failover rehandle attempts");

DEFINE_GAUGE(active_service_requests,
             "Current number of requests tracked by scheduler");
DEFINE_GAUGE(active_request_contexts,
             "Current number of request contexts tracked by scheduler");
DEFINE_GAUGE(active_output_thread_mappings,
             "Current number of output thread mappings tracked by scheduler");
DEFINE_GAUGE(active_failover_removed_requests,
             "Current number of removed requests waiting for failover rehandle");

// ttft latency histogram
DEFINE_HISTOGRAM(time_to_first_token_latency_milliseconds,
                 "Histogram of time to first token latency in milliseconds");
// inter token latency histogram
DEFINE_HISTOGRAM(inter_token_latency_milliseconds,
                 "Histogram of inter token latency in milliseconds");
DEFINE_HISTOGRAM(failover_time_to_first_token_latency_milliseconds,
                 "Histogram of failover time to first token latency in "
                 "milliseconds");
