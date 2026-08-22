#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/prebuilt.hpp"
#include "agents_framework/llm/mock_backend.hpp"
#include "agents_framework/tools/registry.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;
namespace tools = agents_framework::tools;

namespace {

std::shared_ptr<tools::ToolRegistry> registry_with_adder() {
  auto registry = std::make_shared<tools::ToolRegistry>();
  llm::ToolDef def;
  def.name = "add";
  def.description = "Add two numbers.";
  def.input_schema = {
      {"type", "object"},
      {"properties", {{"a", {{"type", "number"}}}, {"b", {{"type", "number"}}}}},
      {"required", nlohmann::json::array({"a", "b"})}};
  REQUIRE(registry->add(tools::make_tool(
      std::move(def), [](const nlohmann::json& args) -> core::Result<std::string> {
        return std::to_string(args.at("a").get<int>() + args.at("b").get<int>());
      })));
  return registry;
}

std::shared_ptr<llm::MockBackend> backend_that_uses_the_adder() {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_handler([](const llm::ChatRequest& request) -> core::Result<llm::ChatResponse> {
    std::string tool_output;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* result = std::get_if<llm::ToolResultBlock>(&block)) {
          tool_output = result->content;
        }
      }
    }

    llm::ChatResponse response;
    if (tool_output.empty()) {
      response.content.push_back(
          llm::ToolUseBlock{"call-1", "add", nlohmann::json{{"a", 2}, {"b", 3}}});
      response.stop_reason = llm::StopReason::ToolUse;
    } else {
      response.content.push_back(llm::TextBlock{"The sum is " + tool_output + "."});
    }
    return response;
  });
  return backend;
}

}  // namespace

TEST_CASE("prebuilt: the react agent loops through a tool to an answer", "[graph]") {
  auto backend = backend_that_uses_the_adder();
  auto compiled = graph::make_react_agent(backend, registry_with_adder(),
                                          {.system = "Use the add tool."});
  REQUIRE(compiled);

  auto state = graph::chat_state("What is 2 + 3?");
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, {.max_steps = 10});
  REQUIRE(stats);

  CHECK(graph::last_assistant_text(state.get<"messages">()) == "The sum is 5.");
  CHECK(backend->calls().size() == 2);
  CHECK(backend->calls().front().tools.size() == 1);  // defs advertised automatically
}

TEST_CASE("prebuilt: chat_state seeds one user message", "[graph]") {
  const auto state = graph::chat_state("hello");
  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 1);
  CHECK(messages.front().role == llm::Role::User);
}

TEST_CASE("prebuilt: last_assistant_text skips tool results and joins blocks", "[graph]") {
  std::vector<llm::Message> messages;
  messages.push_back(llm::Message::user_text("question"));
  messages.push_back(llm::Message{llm::Role::Assistant,
                                  {llm::TextBlock{"part one, "}, llm::TextBlock{"part two"}}});
  messages.push_back(
      llm::Message{llm::Role::User, {llm::ToolResultBlock{"call-1", "ignored", false}}});

  CHECK(graph::last_assistant_text(messages) == "part one, part two");
  CHECK(graph::last_assistant_text({}) == "");
}
