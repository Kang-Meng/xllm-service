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

#include <utility>

#include "request/request.h"

namespace xllm_service {

template <typename GetOffloadBatchSizeFn>
inline bool MaybeRecordFirstDecodeOffloadBatchSize(
    Request* request,
    bool finished_on_prefill_instance,
    GetOffloadBatchSizeFn&& get_offload_batch_size) {
  if (finished_on_prefill_instance || !request->prefill_stage_finished ||
      request->decode_offload_batch_size_recorded) {
    return false;
  }

  request->offload_batch_size = std::forward<GetOffloadBatchSizeFn>(
      get_offload_batch_size)(request->routing.decode_name);
  request->decode_offload_batch_size_recorded = true;
  return true;
}

}  // namespace xllm_service
