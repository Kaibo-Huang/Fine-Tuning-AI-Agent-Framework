// react_demo — a minimal ReAct agent: an LLM node and a tool node in a loop.
//
// The model is told to use the calculator tool for arithmetic. A conditional
// edge hands the run to the tool node whenever the model asked for a tool and
// ends it once the model answers in plain text; the whole conversation
// accumulates on one appending "messages" channel.
//
// Offline by default (mock backend). Set AF_BACKEND=anthropic or openai in
// .env for a live run. Walkthrough: docs/examples.md.

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

using namespace agents_framework::core;
using namespace agents_framework::graph;
using namespace agents_framework::llm;
using namespace agents_framework::tools;
using json = nlohmann::json;

namespace {

// The agent's whole state: one conversation where every node appends messages.
using Messages = Channel<"messages", std::vector<Message>, Append>;
using AgentSchema = Schema<Messages>;

// A five-operation calculator: a JSON-schema definition the model sees, and a
// C++ callback the tool node runs with validated arguments.
std::shared_ptr<ToolRegistry> make_calculator_registry() {
  ToolDef calculator;
  calculator.name = "calculator";
  calculator.description =
      "Evaluate a basic arithmetic operation on two numbers. Supports add, subtract, "
      "multiply, divide, and power.";
  calculator.input_schema = {
      {"type", "object"},
      {"properties",
       {{"op",
         {{"type", "string"},
          {"enum", json::array({"add", "subtract", "multiply", "divide", "power"})}}},
        {"a", {{"type", "number"}}},
        {"b", {{"type", "number"}}}}},
      {"required", json::array({"op", "a", "b"})}};

  const auto evaluate = [](const json& args) -> Result<std::string> {
    const auto op = args.at("op").get<std::string>();
    const double a = args.at("a").get<double>();
    const double b = args.at("b").get<double>();
    if (op == "add") return std::to_string(a + b);
    if (op == "subtract") return std::to_string(a - b);
    if (op == "multiply") return std::to_string(a * b);
    if (op == "divide") {
      if (b == 0.0) return fail(ErrorCode::Tool, "division by zero");
      return std::to_string(a / b);
    }
    return std::to_string(std::pow(a, b));
  };

  auto registry = std::make_shared<ToolRegistry>();
  const auto added = registry->add(make_tool(std::move(calculator), evaluate));
  if (!added) std::cout << "failed to register tool: " << added.error().to_string() << "\n";
  return registry;
}

// Show the agent's work: every tool call and result, then the final answer.
void print_transcript(const std::vector<Message>& messages) {
  for (const auto& message : messages) {
    for (const auto& block : message.content) {
      if (const auto* use = std::get_if<ToolUseBlock>(&block)) {
        std::cout << "[tool call] " << use->name << "(" << use->input.dump() << ")\n";
      } else if (const auto* result = std::get_if<ToolResultBlock>(&block)) {
        std::cout << "[tool result] " << result->content << "\n";
      }
    }
  }
  for (const auto& block : messages.back().content) {
    if (const auto* text = std::get_if<TextBlock>(&block)) {
      std::cout << "[answer] " << text->text << "\n";
    }
  }
}

void run_agent(std::shared_ptr<LLMBackend> backend, const std::string& question) {
  std::cout << "[user] " << question << "\n";

  auto registry = make_calculator_registry();

  LlmNodeOptions options;
  options.system =
      "You are a precise assistant. Use the calculator tool for any arithmetic instead of "
      "computing it yourself, then answer in one short sentence.";
  options.tools = registry->defs();
  options.sampling.max_tokens = 512;

  // Wire the loop: "agent" goes to "tools" whenever the model asked for a
  // tool and to END otherwise; "tools" always hands control back to "agent".
  GraphBuilder<AgentSchema> builder;
  builder.add_node("agent", make_llm_node<AgentSchema>(std::move(backend), options))
      .add_node("tools", make_tool_node<AgentSchema>(registry))
      .set_entry("agent")
      .add_conditional_edge("agent", tools_router<AgentSchema>("tools"),
                            {"tools", std::string{kEnd}})
      .add_edge("tools", "agent");
  auto compiled = std::move(builder).compile();
  if (!compiled) {
    std::cout << "compile error: " << compiled.error().to_string() << "\n";
    return;
  }

  State<AgentSchema> state;
  state.set<"messages">({Message::user_text(question)});

  Executor executor;
  const auto stats = executor.run(*compiled, state, RunOptions{.max_steps = 10});
  if (!stats) {
    std::cout << "run error: " << stats.error().to_string() << "\n";
    return;
  }

  print_transcript(state.get<"messages">());
  std::cout << "(" << stats->steps << " super-steps, " << stats->node_runs << " node runs)\n";
}

// Offline stand-in for the model: ask for the calculator once, then turn the
// tool's result into a sentence — the smallest possible ReAct trajectory.
MockBackend::Handler canned_react() {
  return [](const ChatRequest& request) -> Result<ChatResponse> {
    std::string tool_output;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* result = std::get_if<ToolResultBlock>(&block)) {
          tool_output = result->content;
        }
      }
    }

    ChatResponse response;
    if (tool_output.empty()) {
      response.content.push_back(ToolUseBlock{
          "mock-call-1", "calculator",
          json{{"op", "multiply"}, {"a", 987654321}, {"b", 123456789}}});
      response.stop_reason = StopReason::ToolUse;
    } else {
      response.content.push_back(TextBlock{"The product is " + tool_output + "."});
    }
    return response;
  };
}

}  // namespace

int main() {
  if (const auto env = load_dotenv()) {
    std::cout << "[env] loaded " << env->applied.size() << " variable(s) from "
              << env->path.string() << "\n";
  }

  BackendOptions options;
  options.max_tokens = 512;
  options.mock_handler = canned_react();

  const auto selection = select_backend(system_env(), options);
  if (!selection) {
    std::cerr << "[backend] " << selection.error().to_string() << "\n";
    return 1;
  }
  auto backend = make_backend(*selection, std::move(options), system_env());
  if (!backend) {
    std::cerr << "[backend] " << backend.error().to_string() << "\n";
    return 1;
  }

  std::cout << "=== react demo — " << selection->describe() << " ===\n";
  run_agent(std::move(*backend), "What is 987654321 times 123456789?");
  return 0;
}
