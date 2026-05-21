include(cc_binary)
include(cc_library)

cc_library(
  NAME
    request
  HDRS
    request.h
  SRCS
  DEPS
    :common
    absl::time
    brpc-static
    glog::glog
    nlohmann_json::nlohmann_json
)
