include(cc_binary)
include(cc_library)

cc_library(
  NAME
    request
  HDRS
    request.h
    failover_tracker.h
  DEPS
    :common
    absl::time
    brpc-static
)

cc_binary(
  NAME
    failover_tracker_test
  SRCS
    failover_tracker_test.cpp
  DEPS
    :request
    GTest::gtest_main
)
add_test(NAME FailoverTrackerTest COMMAND failover_tracker_test)
