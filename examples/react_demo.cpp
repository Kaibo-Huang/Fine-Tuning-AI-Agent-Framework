#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/tool_node.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/anthropic_backend.hpp"
#include "agents_framework/llm/openai_backend.hpp"
#include "agents_framework/tools/registry.hpp"

namespace {

using namespace agents_framework;

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using AgentSchema = graph::Schema<Messages>;

std::shared_ptr<tools::ToolRegistry> build_registry() {
  auto registry = std::make_shared<tools::ToolRegistry>();

  llm::ToolDef calculator;
  calculator.name = "calculator";
  calculator.description =
      "Evaluate a basic arithmetic operation on two numbers. Supports add, subtract, "
      "multiply, divide, and power.";
  calculator.input_schema = {
      {"type", "object"},
      {"properties",
       {{"op",
         {{"type", "string"},
          {"enum", nlohmann::json::array({"add", "subtract", "multiply", "divide", "power"})}}},
        {"a", {{"type", "number"}}},
        {"b", {{"type", "number"}}}}},
      {"required", nlohmann::json::array({"op", "a", "b"})}};

  auto result = registry->add(tools::make_tool(
      std::move(calculator), [](const nlohmann::json& args) -> core::Result<std::string> {
        const auto op = args.at("op").get<std::string>();
        const double a = args.at("a").get<double>();
        const double b = args.at("b").get<double>();
        double value = 0.0;
        if (op == "add") {
          value = a + b;
        } else if (op == "subtract") {
          value = a - b;
        } else if (op == "multiply") {
          value = a * b;
        } else if (op == "divide") {
          if (b == 0.0) return core::fail(core::ErrorCode::Tool, "division by zero");
          value = a / b;
        } else {
          value = std::pow(a, b);
        }
        return std::to_string(value);
      }));
  if (!result) std::cout << "failed to register tool: " << result.error().to_string() << "\n";
  return registry;
}

void run_agent(std::string_view label, std::shared_ptr<llm::LLMBackend> backend,
               const std::string& question) {
  std::cout << "=== " << label << " (backend: " << backend->name() << ") ===\n";
  std::cout << "[user] " << question << "\n";

  auto registry = build_registry();

  graph::LlmNodeOptions options;
  options.system =
      "You are a precise assistant. Use the calculator tool for any arithmetic instead of "
      "computing it yourself, then answer in one short sentence.";
  options.tools = registry->defs();
  options.sampling.max_tokens = 512;

  graph::GraphBuilder<AgentSchema> builder;
  builder.add_node("agent", graph::make_llm_node<AgentSchema>(std::move(backend), options))
      .add_node("tools", graph::make_tool_node<AgentSchema>(registry))
      .set_entry("agent")
      .add_conditional_edge("agent", graph::tools_router<AgentSchema>("tools"),
                            {"tools", std::string{graph::kEnd}})
      .add_edge("tools", "agent");
  auto compiled = std::move(builder).compile();
  if (!compiled) {
    std::cout << "compile error: " << compiled.error().to_string() << "\n";
    return;
  }

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text(question)});

  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.max_steps = 10});
  if (!stats) {
    std::cout << "run error: " << stats.error().to_string() << "\n";
    return;
  }

  for (const llm::Message& message : state.get<"messages">()) {
    for (const llm::ContentBlock& block : message.content) {
      if (const auto* use = std::get_if<llm::ToolUseBlock>(&block)) {
        std::cout << "[tool call] " << use->name << "(" << use->input.dump() << ")\n";
      } else if (const auto* result = std::get_if<llm::ToolResultBlock>(&block)) {
        std::cout << "[tool result] " << result->content << "\n";
      }
    }
  }
  const auto& last = state.get<"messages">().back();
  for (const llm::ContentBlock& block : last.content) {
    if (const auto* text = std::get_if<llm::TextBlock>(&block)) {
      std::cout << "[answer] " << text->text << "\n";
    }
  }
  std::cout << "(" << stats->steps << " super-steps, " << stats->node_runs << " node runs)\n\n";
}

}

int main() {
  using namespace agents_framework;
  const std::string question = "What is 987654321 times 123456789?";
  bool ran = false;

  if (auto key = http::SecretStore::from_env("ANTHROPIC_API_KEY")) {
    run_agent("ReAct", std::make_shared<llm::AnthropicBackend>(std::move(*key)), question);
    ran = true;
  }
  if (auto key = http::SecretStore::from_env("OPENAI_API_KEY")) {
    run_agent("ReAct", std::make_shared<llm::OpenAiBackend>(std::move(*key)), question);
    ran = true;
  }
  if (!ran) {
    std::cout << "Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY to run this demo.\n";
  }
  return 0;
}
