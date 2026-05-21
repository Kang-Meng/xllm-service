include(cc_library)
include(cc_test)

cc_library(
  NAME
   managers
  HDRS
    instance_mgr.h
    failover_partition.h
    slo_scoring.h
    global_kvcache_mgr.h
  SRCS
    instance_mgr.cpp
    global_kvcache_mgr.cpp
  DEPS
    :chat_template
    :common
    :etcd_client
    :request
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
)
target_link_libraries(managers PRIVATE brpc-static)

cc_binary(
  NAME
    static_pd_flip_test
  SRCS
    static_pd_flip_test.cpp
  DEPS
    :managers
    GTest::gtest_main
)
add_test(NAME StaticPdFlipTest COMMAND static_pd_flip_test)

cc_binary(
  NAME
    slo_scoring_test
  SRCS
    slo_scoring_test.cpp
  DEPS
    :managers
    GTest::gtest_main
)
add_test(NAME SloScoringTest COMMAND slo_scoring_test)

cc_binary(
  NAME
    heartbeat_timeout_test
  SRCS
    heartbeat_timeout_test.cpp
  DEPS
    :managers
    GTest::gtest_main
)
add_test(NAME HeartbeatTimeoutTest COMMAND heartbeat_timeout_test)

cc_binary(
  NAME
    failover_partition_test
  SRCS
    failover_partition_test.cpp
  DEPS
    :managers
    GTest::gtest_main
)
add_test(NAME FailoverPartitionTest COMMAND failover_partition_test)

cc_binary(
  NAME
    failover_session_test
  SRCS
    failover_session_test.cpp
  DEPS
    :managers
    GTest::gtest_main
)
add_test(NAME FailoverSessionTest COMMAND failover_session_test)
