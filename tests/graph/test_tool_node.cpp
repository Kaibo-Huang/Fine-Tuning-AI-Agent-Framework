#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/tool_node.hpp"
#include "agents_framework/llm/mock_backend.hpp"
#include "agents_framework/tools/registry.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;
namespace tools = agents_framework::tools;

namespace {

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using AgentSchema = graph::Schema<Messages>;

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

const llm::ToolResultBlock& result_block(const llm::Message& message, std::size_t index = 0) {
  REQUIRE(message.content.size() > index);
  const auto* block = std::get_if<llm::ToolResultBlock>(&message.content[index]);
  REQUIRE(block != nullptr);
  return *block;
}

}

TEST_CASE("constructing a tool node without a registry throws", "[graph][tool_node]") {
  REQUIRE_THROWS_AS(graph::make_tool_node<AgentSchema>(nullptr), std::invalid_argument);
}

TEST_CASE("the tool node dispatches tool calls and appends results", "[graph][tool_node]") {
  const auto node = graph::make_tool_node<AgentSchema>(registry_with_adder());

  graph::State<AgentSchema> state;
  state.set<"messages">(
      {llm::Message::user_text("add 19 and 23"),
       llm::Message{llm::Role::Assistant,
                    {llm::TextBlock{"let me add those"},
                     llm::ToolUseBlock{"call_1", "add", {{"a", 19}, {"b", 23}}}}}});

  auto update = node->run(state.map());
  REQUIRE(update);
  state.apply(std::move(*update));

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 3);
  REQUIRE(messages.back().role == llm::Role::User);
  const auto& result = result_block(messages.back());
  REQUIRE(result.tool_use_id == "call_1");
  REQUIRE(result.content == "42");
  REQUIRE(!result.is_error);
}

TEST_CASE("several tool calls in one message all run", "[graph][tool_node]") {
  const auto node = graph::make_tool_node<AgentSchema>(registry_with_adder());

  graph::State<AgentSchema> state;
  state.set<"messages">(
      {llm::Message{llm::Role::Assistant,
                    {llm::ToolUseBlock{"c1", "add", {{"a", 1}, {"b", 2}}},
                     llm::ToolUseBlock{"c2", "add", {{"a", 20}, {"b", 22}}}}}});

  auto update = node->run(state.map());
  REQUIRE(update);
  state.apply(std::move(*update));

  const auto& last = state.get<"messages">().back();
  REQUIRE(last.content.size() == 2);
  REQUIRE(result_block(last, 0).content == "3");
  REQUIRE(result_block(last, 1).content == "42");
}

TEST_CASE("tool failures feed back as error results, not run aborts", "[graph][tool_node]") {
  const auto node = graph::make_tool_node<AgentSchema>(registry_with_adder());

  graph::State<AgentSchema> state;
  state.set<"messages">(
      {llm::Message{llm::Role::Assistant,
                    {llm::ToolUseBlock{"c1", "ghost", nlohmann::json::object()},
                     llm::ToolUseBlock{"c2", "add", {{"a", 1}, {"b", "nope"}}}}}});

  auto update = node->run(state.map());
  REQUIRE(update);
  state.apply(std::move(*update));

  const auto& last = state.get<"messages">().back();
  const auto& unknown = result_block(last, 0);
  REQUIRE(unknown.is_error);
  REQUIRE(unknown.content == "NotFound: unknown tool (ghost)");
  const auto& invalid = result_block(last, 1);
  REQUIRE(invalid.is_error);
  REQUIRE(invalid.content == "Invalid: argument 'b' must have type number");
}

TEST_CASE("running the tool node without pending calls is an error", "[graph][tool_node]") {
  const auto node = graph::make_tool_node<AgentSchema>(registry_with_adder());

  graph::State<AgentSchema> state;
  auto no_messages = node->run(state.map());
  REQUIRE(!no_messages);

  state.set<"messages">({llm::Message::assistant_text("all done")});
  auto no_calls = node->run(state.map());
  REQUIRE(!no_calls);
  REQUIRE(no_calls.error().message == "tool node ran but the last message has no tool calls");
}

TEST_CASE("the tools router picks the tools node only on pending calls", "[graph][tool_node]") {
  const auto route = graph::tools_router<AgentSchema>("tools");

  graph::State<AgentSchema> with_call;
  with_call.set<"messages">(
      {llm::Message{llm::Role::Assistant,
                    {llm::ToolUseBlock{"c1", "add", nlohmann::json::object()}}}});
  REQUIRE(route(with_call.view()) == "tools");

  graph::State<AgentSchema> without_call;
  without_call.set<"messages">({llm::Message::assistant_text("done")});
  REQUIRE(route(without_call.view()) == graph::kEnd);

  graph::State<AgentSchema> empty;
  REQUIRE(route(empty.view()) == graph::kEnd);
}

TEST_CASE("a ReAct agent loops llm -> tools -> llm to a final answer", "[graph][react]") {
  auto backend = std::make_shared<llm::MockBackend>();
  {
    llm::ChatResponse first;
    first.content = {llm::TextBlock{"I should add these."},
                     llm::ToolUseBlock{"call_1", "add", {{"a", 19}, {"b", 23}}}};
    first.stop_reason = llm::StopReason::ToolUse;
    backend->push_response(first);

    llm::ChatResponse second;
    second.content = {llm::TextBlock{"The answer is 42."}};
    backend->push_response(second);
  }

  auto registry = registry_with_adder();
  graph::LlmNodeOptions options;
  options.model = "test-model";
  options.tools = registry->defs();

  graph::GraphBuilder<AgentSchema> builder;
  builder.add_node("agent", graph::make_llm_node<AgentSchema>(backend, options))
      .add_node("tools", graph::make_tool_node<AgentSchema>(registry))
      .set_entry("agent")
      .add_conditional_edge("agent", graph::tools_router<AgentSchema>("tools"),
                            {"tools", std::string{graph::kEnd}})
      .add_edge("tools", "agent");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("what is 19 + 23?")});
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(stats);
  REQUIRE(stats->steps == 3);

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 4);
  REQUIRE(result_block(messages[2]).content == "42");
  REQUIRE(messages.back() == llm::Message::assistant_text("The answer is 42."));

  REQUIRE(backend->calls().size() == 2);
  REQUIRE(backend->calls().front().tools.size() == 1);
  REQUIRE(backend->calls().back().messages.size() == 3);
}
