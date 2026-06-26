find_package(Catch2 3 CONFIG REQUIRED)

add_executable(agents_framework_tests
  test_version.cpp
  core/test_result.cpp
)

target_link_libraries(agents_framework_tests
  PRIVATE
    agents_framework
    Catch2::Catch2WithMain
)

add_test(NAME agents_framework_tests COMMAND agents_framework_tests)
