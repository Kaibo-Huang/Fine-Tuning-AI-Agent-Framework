find_package(Catch2 3 CONFIG REQUIRED)

add_executable(agents_framework_tests
  test_version.cpp
  core/test_result.cpp
  core/test_logging.cpp
  core/test_config.cpp
  core/test_rng.cpp
  core/test_clock.cpp
  core/test_id.cpp
  core/test_fixed_string.cpp
  http/test_secrets.cpp
  http/test_http_client.cpp
  http/test_sse.cpp
  llm/test_canonical.cpp
  llm/test_mock_backend.cpp
  llm/test_anthropic_backend.cpp
  llm/test_openai_backend.cpp
  graph/test_state.cpp
  graph/test_state_serde.cpp
  graph/test_node.cpp
  graph/test_builder.cpp
  graph/test_compile.cpp
  graph/test_executor.cpp
  graph/test_routing.cpp
  graph/test_llm_node.cpp
  tools/test_registry.cpp
  tools/test_process.cpp
  tools/test_http_tool.cpp
)

target_link_libraries(agents_framework_tests
  PRIVATE
    agents_framework
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(agents_framework_tests)
