find_package(Catch2 3 CONFIG REQUIRED)

add_executable(agents_framework_tests
  test_version.cpp
  core/test_result.cpp
  core/test_logging.cpp
  core/test_config.cpp
  core/test_rng.cpp
  core/test_clock.cpp
)

target_link_libraries(agents_framework_tests
  PRIVATE
    agents_framework
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(agents_framework_tests)
