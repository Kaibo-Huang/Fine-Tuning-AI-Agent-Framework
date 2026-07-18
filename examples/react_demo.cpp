#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/dotenv.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/tool_node.hpp"
#include "agents_framework/llm/backend_factory.hpp"
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

llm::MockBackend::Handler canned_react() {
  return [](const llm::ChatRequest& request) -> core::Result<llm::ChatResponse> {
    bool has_tool_result = false;
    std::string tool_output;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* result = std::get_if<llm::ToolResultBlock>(&block)) {
          has_tool_result = true;
          tool_output = result->content;
        }
      }
    }

    llm::ChatResponse response;
    if (!has_tool_result) {
      response.content.push_back(llm::ToolUseBlock{
          "mock-call-1", "calculator",
          nlohmann::json{{"op", "multiply"}, {"a", 987654321}, {"b", 123456789}}});
      response.stop_reason = llm::StopReason::ToolUse;
    } else {
      response.content.push_back(llm::TextBlock{"The product is " + tool_output + "."});
    }
    return response;
  };
}

}

int main() {
  using namespace agents_framework;

  if (const auto env_file = core::load_dotenv()) {
    std::cout << "[env] loaded " << env_file->applied.size() << " variable(s) from "
              << env_file->path.string() << "\n";
  }

  llm::BackendOptions options;
  options.max_tokens = 512;
  options.mock_handler = canned_react();

  const auto selection = llm::select_backend(llm::system_env(), options);
  if (!selection) {
    std::cerr << "[backend] " << selection.error().to_string() << "\n";
    return 1;
  }
  auto backend = llm::make_backend(*selection, std::move(options), llm::system_env());
  if (!backend) {
    std::cerr << "[backend] " << backend.error().to_string() << "\n";
    return 1;
  }

  run_agent(selection->describe(), std::move(*backend), "What is 987654321 times 123456789?");
  return 0;
}
