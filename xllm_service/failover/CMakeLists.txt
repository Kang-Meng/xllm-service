include(cc_binary)
include(cc_library)

cc_library(
  NAME
    failover
  HDRS
    coordinator.h
    debug.h
    local_finish.h
    recovery_dump.h
    rehandle.h
    replay.h
    state.h
    tracker.h
  SRCS
    coordinator.cpp
    recovery_dump.cpp
  DEPS
    :common
    :request
    absl::time
    brpc-static
    glog::glog
    nlohmann_json::nlohmann_json
)

cc_binary(
  NAME
    failover_coordinator_test
  SRCS
    coordinator_test.cpp
  DEPS
    :failover
    GTest::gtest_main
)
add_test(NAME FailoverCoordinatorTest COMMAND failover_coordinator_test)

cc_binary(
  NAME
    failover_state_test
  SRCS
    state_test.cpp
  DEPS
    :failover
    GTest::gtest_main
)
add_test(NAME FailoverStateTest COMMAND failover_state_test)

cc_binary(
  NAME
    failover_tracker_test
  SRCS
    tracker_test.cpp
  DEPS
    :failover
    GTest::gtest_main
)
add_test(NAME FailoverTrackerTest COMMAND failover_tracker_test)

cc_binary(
  NAME
    failover_recovery_dumper_test
  SRCS
    recovery_dump_test.cpp
  DEPS
    :failover
    GTest::gtest_main
)
add_test(NAME FailoverRecoveryDumperTest COMMAND failover_recovery_dumper_test)

cc_binary(
  NAME
    failover_rehandle_test
  SRCS
    rehandle_test.cpp
  DEPS
    :failover
    GTest::gtest_main
)
add_test(NAME FailoverRehandleTest COMMAND failover_rehandle_test)
