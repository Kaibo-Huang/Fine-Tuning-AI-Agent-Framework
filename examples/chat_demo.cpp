#include <iostream>
#include <string_view>
#include <variant>

#include <nlohmann/json.hpp>

#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/anthropic_backend.hpp"
#include "agents_framework/llm/openai_backend.hpp"

namespace {

using namespace agents_framework;

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
  bool ran = false;

  if (auto key = http::SecretStore::from_env("ANTHROPIC_API_KEY")) {
    llm::AnthropicBackend backend{std::move(*key)};
    run_demo("Anthropic", backend);
    ran = true;
  }
  if (auto key = http::SecretStore::from_env("OPENAI_API_KEY")) {
    llm::OpenAiBackend backend{std::move(*key)};
    run_demo("OpenAI", backend);
    ran = true;
  }
  if (!ran) {
    std::cout << "Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY to run this demo.\n";
  }
  return 0;
}
