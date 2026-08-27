// chat_demo: the LLM backend abstraction in three short scenes:
//
//   1. ask a one-off question and print the reply
//   2. stream a reply token by token
//   3. let the model call a tool, feed the result back, print the final answer
//
// By default this runs fully offline against the built-in mock backend. Set
// AF_BACKEND=anthropic or AF_BACKEND=openai (plus the matching API key) in the
// environment or a .env file to run the exact same code against a live model.
// Walkthrough: docs/examples.md.

#include <iostream>
#include <string>
#include <string_view>
#include <variant>

#include <nlohmann/json.hpp>

#include "agents_framework/core/dotenv.hpp"
#include "agents_framework/llm/backend_factory.hpp"

using namespace agents_framework::core;
using namespace agents_framework::llm;
using json = nlohmann::json;

namespace {

// Scene 1: a single-turn question.
void ask_a_question(LLMBackend& backend) {
  ChatRequest request;
  request.messages = {Message::user_text("In one sentence, what is a directed graph?")};
  request.sampling.max_tokens = 128;

  const auto reply = backend.generate(request);
  if (!reply) {
    std::cout << "[reply]  error: " << reply.error().to_string() << "\n";
    return;
  }
  std::cout << "[reply]  " << reply->text() << "\n";
}

// Scene 2: the same kind of call, but streamed token by token.
void stream_a_reply(LLMBackend& backend) {
  ChatRequest request;
  request.messages = {Message::user_text("Count from 1 to 5.")};
  request.sampling.max_tokens = 64;

  std::cout << "[stream] ";
  const auto reply = backend.generate_stream(
      request, on_text([](std::string_view text) { std::cout << text << std::flush; }));
  std::cout << "\n";
  if (!reply) std::cout << "[stream] error: " << reply.error().to_string() << "\n";
}

// Scene 3: a complete tool round-trip. The model asks for a tool, we run it and
// hand the result back, and the model folds it into a final answer.
void call_a_tool(LLMBackend& backend) {
  ToolDef weather;
  weather.name = "get_weather";
  weather.description = "Get the current weather for a city.";
  weather.input_schema = json{{"type", "object"},
                              {"properties", {{"city", {{"type", "string"}}}}},
                              {"required", json::array({"city"})}};

  ChatRequest request;
  request.tools = {weather};
  request.messages = {Message::user_text("Use the get_weather tool for Waterloo.")};
  request.sampling.max_tokens = 256;

  const auto first = backend.generate(request);
  if (!first) {
    std::cout << "[tool]   error: " << first.error().to_string() << "\n";
    return;
  }
  const auto calls = first->tool_uses();
  if (calls.empty()) {
    std::cout << "[tool]   model answered without a tool: " << first->text() << "\n";
    return;
  }

  const auto& call = calls.front();
  std::cout << "[tool]   model called " << call.name << "(" << call.input.dump() << ")\n";

  // Play the tool's part ourselves and append its result to the conversation.
  request.messages.push_back(Message{Role::Assistant, first->content});
  request.messages.push_back(
      Message{Role::User, {ToolResultBlock{call.id, "18C and sunny", false}}});

  const auto second = backend.generate(request);
  if (!second) {
    std::cout << "[tool]   error: " << second.error().to_string() << "\n";
    return;
  }
  std::cout << "[tool]   final: " << second->text() << "\n";
}

// Deterministic replies for the offline mock backend, so the demo runs without
// an API key and always prints the same conversation.
MockBackend::Handler canned_replies() {
  return [](const ChatRequest& request) -> Result<ChatResponse> {
    std::string prompt;
    bool answered_tool = false;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* text = std::get_if<TextBlock>(&block)) {
          prompt += text->text;
        } else if (std::get_if<ToolResultBlock>(&block) != nullptr) {
          answered_tool = true;
        }
      }
    }

    ChatResponse response;
    if (!request.tools.empty() && !answered_tool) {
      response.content.push_back(
          ToolUseBlock{"mock-tool-1", "get_weather", json{{"city", "Waterloo"}}});
      response.stop_reason = StopReason::ToolUse;
    } else if (answered_tool) {
      response.content.push_back(TextBlock{"It is 18C and sunny in Waterloo."});
    } else if (prompt.find("Count") != std::string::npos) {
      response.content.push_back(TextBlock{"1 2 3 4 5"});
    } else {
      response.content.push_back(
          TextBlock{"A directed graph is a set of nodes joined by edges that "
                    "point from one node to another."});
    }
    response.usage = {12, 24};
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
  options.max_tokens = 256;
  options.mock_handler = canned_replies();

  const auto selection = select_backend(system_env(), options);
  if (!selection) {
    std::cerr << "[backend] " << selection.error().to_string() << "\n";
    return 1;
  }
  const auto opened = make_backend(*selection, options, system_env());
  if (!opened) {
    std::cerr << "[backend] " << opened.error().to_string() << "\n";
    return 1;
  }
  LLMBackend& backend = **opened;

  std::cout << "=== chat demo: " << selection->describe() << " ===\n";
  ask_a_question(backend);
  stream_a_reply(backend);
  call_a_tool(backend);

  if (!selection->live) {
    std::cout << "\nOffline mock run. Set AF_BACKEND=openai (or anthropic) in .env for a "
                 "live run.\n";
  }
  return 0;
}
