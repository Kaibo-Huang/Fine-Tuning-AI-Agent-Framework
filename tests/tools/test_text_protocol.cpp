#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/tool_node.hpp"
#include "agents_framework/llm/mock_backend.hpp"
#include "agents_framework/tools/registry.hpp"
#include "agents_framework/tools/text_protocol.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;
namespace tools = agents_framework::tools;

TEST_CASE("fenced json tool calls parse", "[tools][text_protocol]") {
  const auto call = tools::parse_text_tool_call(
      "I will look that up.\n```json\n{\"tool\": \"search\", \"input\": {\"query\": \"c++\"}}\n```");
  REQUIRE(call);
  REQUIRE(call->name == "search");
  REQUIRE(call->input == nlohmann::json{{"query", "c++"}});
}

TEST_CASE("bare json objects with explicit tool keys parse", "[tools][text_protocol]") {
  const auto call = tools::parse_text_tool_call(
      "Thought: need data. {\"tool_name\": \"fetch\", \"arguments\": {\"id\": 7}} done");
  REQUIRE(call);
  REQUIRE(call->name == "fetch");
  REQUIRE(call->input == nlohmann::json{{"id", 7}});
}

TEST_CASE("openai-style nested function calls parse", "[tools][text_protocol]") {
  const auto call = tools::parse_text_tool_call(
      R"(```{"function": {"name": "add", "arguments": "{\"a\": 1, \"b\": 2}"}}```)");
  REQUIRE(call);
  REQUIRE(call->name == "add");
  REQUIRE(call->input == nlohmann::json{{"a", 1}, {"b", 2}});
}

TEST_CASE("react action lines parse with json input", "[tools][text_protocol]") {
  const auto call = tools::parse_text_tool_call(
      "Thought: I should search.\nAction: search\nAction Input: {\"query\": \"agents\"}\n"
      "Observation: pending");
  REQUIRE(call);
  REQUIRE(call->name == "search");
  REQUIRE(call->input == nlohmann::json{{"query", "agents"}});
}

TEST_CASE("react action lines fall back to a raw string input", "[tools][text_protocol]") {
  const auto call =
      tools::parse_text_tool_call("Action: lookup\nAction Input: capital of France");
  REQUIRE(call);
  REQUIRE(call->name == "lookup");
  REQUIRE(call->input == nlohmann::json{{"input", "capital of France"}});
}

TEST_CASE("plain prose and unrelated json do not parse as tool calls",
          "[tools][text_protocol]") {
  REQUIRE(!tools::parse_text_tool_call("The answer is 42."));
  REQUIRE(!tools::parse_text_tool_call("Profile: {\"name\": \"Bob\", \"age\": 3}"));
  REQUIRE(!tools::parse_text_tool_call("{\"answer\": 42}"));
  REQUIRE(!tools::parse_text_tool_call(""));
  REQUIRE(!tools::parse_text_tool_call("```json\n{\"name\": \"Bob\", \"age\": 3}\n```"));

  const auto named_with_input =
      tools::parse_text_tool_call("```json\n{\"name\": \"add\", \"input\": {\"a\": 1}}\n```");
  REQUIRE(named_with_input);
  REQUIRE(named_with_input->name == "add");
}

TEST_CASE("tool instructions include names, schemas, and the protocol",
          "[tools][text_protocol]") {
  llm::ToolDef def;
  def.name = "calculator";
  def.description = "Do arithmetic.";
  def.input_schema = {{"type", "object"}};

  const auto instructions = tools::render_tool_instructions(std::vector{def});
  REQUIRE(instructions.find("calculator") != std::string::npos);
  REQUIRE(instructions.find("Do arithmetic.") != std::string::npos);
  REQUIRE(instructions.find("{\"type\":\"object\"}") != std::string::npos);
  REQUIRE(instructions.find("{\"tool\": \"<tool name>\"") != std::string::npos);
}

TEST_CASE("rendering canonical history to text drops tool_use and rewrites results",
          "[tools][text_protocol]") {
  const std::vector<llm::Message> history = {
      llm::Message::user_text("add 1 and 2"),
      llm::Message{llm::Role::Assistant,
                   {llm::TextBlock{"{\"tool\": \"add\", \"input\": {}}"},
                    llm::ToolUseBlock{"call_1", "add", nlohmann::json::object()}}},
      llm::Message{llm::Role::User, {llm::ToolResultBlock{"call_1", "3", false}}},
      llm::Message{llm::Role::User, {llm::ToolResultBlock{"call_2", "nope", true}}}};

  const auto rendered = tools::render_text_messages(history);
  REQUIRE(rendered.size() == 4);
  REQUIRE(rendered[1].content ==
          std::vector<llm::ContentBlock>{llm::TextBlock{"{\"tool\": \"add\", \"input\": {}}"}});
  REQUIRE(rendered[2].content ==
          std::vector<llm::ContentBlock>{llm::TextBlock{"Tool result for call_1:\n3"}});
  REQUIRE(rendered[3].content ==
          std::vector<llm::ContentBlock>{llm::TextBlock{"Tool call call_2 failed:\nnope"}});
}

namespace {

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using AgentSchema = graph::Schema<Messages>;

}

TEST_CASE("a text-protocol ReAct loop works against a text-only backend",
          "[tools][text_protocol][react]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_capabilities({false, false});
  {
    llm::ChatResponse first;
    first.content = {
        llm::TextBlock{"{\"tool\": \"add\", \"input\": {\"a\": 19, \"b\": 23}}"}};
    backend->push_response(first);

    llm::ChatResponse second;
    second.content = {llm::TextBlock{"The answer is 42."}};
    backend->push_response(second);
  }

  auto registry = std::make_shared<tools::ToolRegistry>();
  llm::ToolDef def;
  def.name = "add";
  def.description = "Add two numbers.";
  def.input_schema = {{"type", "object"}};
  REQUIRE(registry->add(tools::make_tool(
      std::move(def), [](const nlohmann::json& args) -> core::Result<std::string> {
        return std::to_string(args.at("a").get<int>() + args.at("b").get<int>());
      })));

  graph::LlmNodeOptions options;
  options.system = "You are a helpful assistant.";
  options.tools = registry->defs();
  options.tool_format = graph::ToolCallFormat::TextProtocol;

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
  REQUIRE(executor.run(*compiled, state));

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 4);
  REQUIRE(messages[1].content.size() == 2);
  const auto* use = std::get_if<llm::ToolUseBlock>(&messages[1].content[1]);
  REQUIRE(use != nullptr);
  REQUIRE(use->id == "call_1");
  REQUIRE(use->name == "add");
  REQUIRE(messages.back() == llm::Message::assistant_text("The answer is 42."));

  REQUIRE(backend->calls().size() == 2);
  const auto& first_request = backend->calls().front();
  REQUIRE(first_request.tools.empty());
  REQUIRE(first_request.system.has_value());
  REQUIRE(first_request.system->find("You are a helpful assistant.") != std::string::npos);
  REQUIRE(first_request.system->find("To call a tool") != std::string::npos);

  const auto& second_request = backend->calls().back();
  REQUIRE(second_request.messages.size() == 3);
  for (const llm::Message& message : second_request.messages) {
    for (const llm::ContentBlock& block : message.content) {
      REQUIRE(std::holds_alternative<llm::TextBlock>(block));
    }
  }
  const auto* rendered_result =
      std::get_if<llm::TextBlock>(&second_request.messages.back().content.front());
  REQUIRE(rendered_result != nullptr);
  REQUIRE(rendered_result->text == "Tool result for call_1:\n42");
}
