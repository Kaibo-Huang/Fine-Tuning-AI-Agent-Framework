#include <iostream>
#include <string>
#include <string_view>
#include <variant>

#include <nlohmann/json.hpp>

#include "agents_framework/core/dotenv.hpp"
#include "agents_framework/llm/backend_factory.hpp"

namespace {

using namespace agents_framework;

llm::MockBackend::Handler canned_replies() {
  return [](const llm::ChatRequest& request) -> core::Result<llm::ChatResponse> {
    std::string prompt;
    bool answered_tool = false;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* text = std::get_if<llm::TextBlock>(&block)) {
          prompt += text->text;
        } else if (std::get_if<llm::ToolResultBlock>(&block) != nullptr) {
          answered_tool = true;
        }
      }
    }

    llm::ChatResponse response;
    if (!request.tools.empty() && !answered_tool) {
      response.content.push_back(
          llm::ToolUseBlock{"mock-tool-1", "get_weather", nlohmann::json{{"city", "Waterloo"}}});
      response.stop_reason = llm::StopReason::ToolUse;
    } else if (answered_tool) {
      response.content.push_back(llm::TextBlock{"It is 18C and sunny in Waterloo."});
    } else if (prompt.find("Count") != std::string::npos) {
      response.content.push_back(llm::TextBlock{"1 2 3 4 5"});
    } else {
      response.content.push_back(
          llm::TextBlock{"A directed graph is a set of nodes joined by edges that "
                         "point from one node to another."});
    }
    response.usage = {12, 24};
    return response;
  };
}

void run_demo(std::string_view label, llm::LLMBackend& backend) {
  std::cout << "=== " << label << " (backend: " << backend.name() << ") ===\n";

  {
    llm::ChatRequest request;
    request.messages = {llm::Message::user_text("In one sentence, what is a directed graph?")};
    request.sampling.max_tokens = 128;
    if (auto response = backend.generate(request)) {
      std::cout << "[reply]  " << response->text() << "\n";
    } else {
      std::cout << "[reply]  error: " << response.error().to_string() << "\n";
    }
  }

  {
    llm::ChatRequest request;
    request.messages = {llm::Message::user_text("Count from 1 to 5.")};
    request.sampling.max_tokens = 64;
    std::cout << "[stream] ";
    auto response = backend.generate_stream(request, [](const llm::StreamEvent& event) {
      if (const auto* delta = std::get_if<llm::TextDelta>(&event)) {
        std::cout << delta->text << std::flush;
      }
    });
    std::cout << "\n";
    if (!response) std::cout << "[stream] error: " << response.error().to_string() << "\n";
  }

  {
    llm::ToolDef weather;
    weather.name = "get_weather";
    weather.description = "Get the current weather for a city.";
    weather.input_schema = nlohmann::json{
        {"type", "object"},
        {"properties", {{"city", {{"type", "string"}}}}},
        {"required", nlohmann::json::array({"city"})}};

    llm::ChatRequest request;
    request.tools = {weather};
    request.messages = {llm::Message::user_text("Use the get_weather tool for Waterloo.")};
    request.sampling.max_tokens = 256;

    auto first = backend.generate(request);
    if (!first) {
      std::cout << "[tool]   error: " << first.error().to_string() << "\n\n";
      return;
    }
    const auto uses = first->tool_uses();
    if (uses.empty()) {
      std::cout << "[tool]   model answered without a tool: " << first->text() << "\n\n";
      return;
    }

    const auto& call = uses.front();
    std::cout << "[tool]   model called " << call.name << "(" << call.input.dump() << ")\n";

    request.messages.push_back(llm::Message{llm::Role::Assistant, first->content});
    request.messages.push_back(
        llm::Message{llm::Role::User, {llm::ToolResultBlock{call.id, "18C and sunny", false}}});

    if (auto second = backend.generate(request)) {
      std::cout << "[tool]   final: " << second->text() << "\n";
    } else {
      std::cout << "[tool]   error: " << second.error().to_string() << "\n";
    }
  }
  std::cout << "\n";
}

}  

int main() {
  using namespace agents_framework;

  if (const auto env_file = core::load_dotenv()) {
    std::cout << "[env] loaded " << env_file->applied.size() << " variable(s) from "
              << env_file->path.string() << "\n";
  }

  llm::BackendOptions options;
  options.max_tokens = 256;
  options.mock_handler = canned_replies();

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

  run_demo(selection->describe(), **backend);
  if (!selection->live) {
    std::cout << "Offline mock run. Set AF_BACKEND=openai (or anthropic) in .env for a "
                 "live run.\n";
  }
  return 0;
}
