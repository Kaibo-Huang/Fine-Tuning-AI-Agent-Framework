#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/tools/process.hpp"
#include "agents_framework/tools/process_tool.hpp"
#include "agents_framework/tools/registry.hpp"

namespace core = agents_framework::core;
namespace tools = agents_framework::tools;

namespace {

#ifdef _WIN32
const std::vector<std::string> kShell{"cmd", "/c"};
const std::vector<std::string> kEcho{"cmd", "/c", "echo hello tool"};
const std::vector<std::string> kExit3{"cmd", "/c", "exit 3"};
const std::vector<std::string> kCat{"cmd", "/c", "more"};
const std::vector<std::string> kSleep{"cmd", "/c", "ping -n 6 127.0.0.1 > NUL"};
#else
const std::vector<std::string> kShell{"sh", "-c"};
const std::vector<std::string> kEcho{"sh", "-c", "echo hello tool"};
const std::vector<std::string> kExit3{"sh", "-c", "exit 3"};
const std::vector<std::string> kCat{"cat"};
const std::vector<std::string> kSleep{"sh", "-c", "sleep 5"};
#endif

}

TEST_CASE("run_process captures output and the exit code", "[tools][process]") {
  const auto result = tools::run_process(kEcho);
  REQUIRE(result);
  REQUIRE(result->exit_code == 0);
  REQUIRE(result->output.find("hello tool") != std::string::npos);
  REQUIRE(!result->truncated);
}

TEST_CASE("run_process reports non-zero exit codes", "[tools][process]") {
  const auto result = tools::run_process(kExit3);
  REQUIRE(result);
  REQUIRE(result->exit_code == 3);
}

TEST_CASE("run_process pipes stdin to the child", "[tools][process]") {
  tools::ProcessOptions options;
  options.stdin_data = "line from stdin\n";
  const auto result = tools::run_process(kCat, options);
  REQUIRE(result);
  REQUIRE(result->output.find("line from stdin") != std::string::npos);
}

TEST_CASE("run_process truncates output beyond the cap", "[tools][process]") {
  tools::ProcessOptions options;
  options.max_output_bytes = 5;
  const auto result = tools::run_process(kEcho, options);
  REQUIRE(result);
  REQUIRE(result->output.size() <= 5);
  REQUIRE(result->truncated);
}

TEST_CASE("run_process kills a child that outlives the timeout", "[tools][process]") {
  tools::ProcessOptions options;
  options.timeout = std::chrono::milliseconds{300};
  const auto started = std::chrono::steady_clock::now();
  const auto result = tools::run_process(kSleep, options);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Timeout);
  REQUIRE(elapsed < std::chrono::seconds{4});
}

TEST_CASE("a lingering grandchild does not hang run_process", "[tools][process]") {
#ifdef _WIN32
  const std::vector<std::string> command{
      "cmd", "/c", "start /b ping -n 30 127.0.0.1 > NUL & exit 7"};
#else
  const std::vector<std::string> command{"sh", "-c", "sleep 30 & exit 7"};
#endif
  tools::ProcessOptions options;
  options.timeout = std::chrono::milliseconds{3000};
  const auto started = std::chrono::steady_clock::now();
  const auto result = tools::run_process(command, options);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  REQUIRE(elapsed < std::chrono::seconds{8});
#ifdef _WIN32
  REQUIRE(result);
  REQUIRE(result->exit_code == 7);
#else
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Timeout);
#endif
}

TEST_CASE("concurrent run_process calls stay isolated", "[tools][process]") {
  constexpr int kThreads = 4;
  std::array<core::Result<tools::ProcessResult>, kThreads> results;
  std::vector<std::thread> threads;
  const auto started = std::chrono::steady_clock::now();
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&results, t] {
      tools::ProcessOptions options;
      options.stdin_data = "payload " + std::to_string(t) + "\n";
      options.timeout = std::chrono::milliseconds{10000};
      results[t] = tools::run_process(kCat, options);
    });
  }
  for (std::thread& thread : threads) thread.join();
  REQUIRE(std::chrono::steady_clock::now() - started < std::chrono::seconds{9});
  for (int t = 0; t < kThreads; ++t) {
    REQUIRE(results[t]);
    REQUIRE(results[t]->output.find("payload " + std::to_string(t)) != std::string::npos);
  }
}

TEST_CASE("run_process rejects an empty command and missing executables",
          "[tools][process]") {
  const auto empty = tools::run_process({});
  REQUIRE(!empty);
  REQUIRE(empty.error().code == core::ErrorCode::Invalid);

  const auto missing = tools::run_process({"agents-framework-no-such-binary"});
  if (missing) {
    REQUIRE(missing->exit_code == 127);
  } else {
    REQUIRE(missing.error().code == core::ErrorCode::Io);
  }
}

TEST_CASE("a process tool runs through the registry with validated args",
          "[tools][process_tool]") {
  tools::ProcessToolOptions options;
  options.name = "shell_echo";
  options.description = "Echo text through the platform shell.";
  options.command = kShell;
  options.allow_args = true;
  tools::ToolRegistry registry;
  REQUIRE(registry.add(tools::make_process_tool(options)));

  const auto defs = registry.defs();
  REQUIRE(defs.front().name == "shell_echo");
  REQUIRE(defs.front().input_schema.at("properties").contains("args"));
  REQUIRE(!defs.front().input_schema.at("properties").contains("stdin"));

  const auto result =
      registry.invoke("shell_echo", {{"args", nlohmann::json::array({"echo from-the-agent"})}});
  REQUIRE(result);
  REQUIRE(result->find("from-the-agent") != std::string::npos);

  const auto bad = registry.invoke("shell_echo", {{"args", "not-an-array"}});
  REQUIRE(!bad);
  REQUIRE(bad.error().message == "argument 'args' must have type array");
}

TEST_CASE("a process tool reports non-zero exits in its output", "[tools][process_tool]") {
  tools::ProcessToolOptions options;
  options.name = "fail";
  options.command = kExit3;
  tools::ProcessTool tool{options};

  const auto result = tool.call(nlohmann::json::object());
  REQUIRE(result);
  REQUIRE(result->starts_with("exit code 3"));
}

TEST_CASE("model-supplied args are ignored unless enabled", "[tools][process_tool]") {
  tools::ProcessToolOptions options;
  options.name = "echo_fixed";
  options.command = kEcho;
  tools::ProcessTool tool{options};

  const auto result = tool.call({{"args", nlohmann::json::array({"ignored"})}});
  REQUIRE(result);
  REQUIRE(result->find("hello tool") != std::string::npos);
}
