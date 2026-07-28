#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "../live_env.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/tool_node.hpp"
#include "agents_framework/tools/registry.hpp"

using namespace agents_framework;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("live: the configured backend answers a prompt", "[.live][llm]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured (set AF_BACKEND and an API key)");

  INFO("backend: " << live->selection.describe());

  llm::ChatRequest request;
  request.system = "Answer with a single word and no punctuation.";
  request.messages = {llm::Message::user_text("What is the capital of France?")};
  request.sampling.max_tokens = 16;
  request.sampling.temperature = 0.0;

  const auto response = live->backend->generate(request);
  REQUIRE(response.has_value());
  CHECK_THAT(response->text(), ContainsSubstring("Paris"));
  CHECK(response->usage.input_tokens > 0);
  CHECK(response->usage.output_tokens > 0);
}

TEST_CASE("live: the backend streams token deltas", "[.live][llm]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  llm::ChatRequest request;
  request.messages = {llm::Message::user_text("Count from 1 to 5, separated by spaces.")};
  request.sampling.max_tokens = 64;
  request.sampling.temperature = 0.0;

  int deltas = 0;
  std::string streamed;
  const auto response = live->backend->generate_stream(request, [&](const llm::StreamEvent& event) {
    if (const auto* delta = std::get_if<llm::TextDelta>(&event)) {
      ++deltas;
      streamed += delta->text;
    }
  });

  REQUIRE(response.has_value());
  CHECK(deltas > 1);
  CHECK(streamed == response->text());
}

TEST_CASE("live: the streaming path reports usage and lifecycle events", "[.live][llm]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  llm::ChatRequest request;
  request.system = "Reply with exactly the word: pong";
  request.messages = {llm::Message::user_text("ping")};
  request.sampling.max_tokens = 16;
  request.sampling.temperature = 0.0;

  bool saw_start = false;
  bool saw_stop = false;
  llm::Usage stop_usage;
  const auto streamed = live->backend->generate_stream(request, [&](const llm::StreamEvent& event) {
    if (std::holds_alternative<llm::MessageStart>(event)) saw_start = true;
    if (const auto* stop = std::get_if<llm::MessageStop>(&event)) {
      saw_stop = true;
      stop_usage = stop->usage;
    }
  });
  REQUIRE(streamed.has_value());

  CHECK(saw_start);
  CHECK(saw_stop);
  CHECK(streamed->usage.input_tokens > 0);
  CHECK(streamed->usage.output_tokens > 0);
  CHECK(stop_usage.input_tokens == streamed->usage.input_tokens);

  const auto blocking = live->backend->generate(request);
  REQUIRE(blocking.has_value());
  CHECK(blocking->stop_reason == streamed->stop_reason);
}

TEST_CASE("live: a truncated response maps to StopReason::MaxTokens", "[.live][llm]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  llm::ChatRequest request;
  request.messages = {
      llm::Message::user_text("Write a detailed 500-word essay about the history of the bicycle.")};
  request.sampling.max_tokens = 8;

  const auto response = live->backend->generate(request);
  REQUIRE(response.has_value());
  CHECK(response->stop_reason == llm::StopReason::MaxTokens);
}

TEST_CASE("live: an unknown model maps a 4xx onto a typed error", "[.live][llm]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  llm::ChatRequest request;
  request.model = "definitely-not-a-real-model-name";
  request.messages = {llm::Message::user_text("hello")};
  request.sampling.max_tokens = 16;

  const auto response = live->backend->generate(request);
  REQUIRE_FALSE(response.has_value());

  const auto code = response.error().code;
  INFO("error: " << response.error().to_string());
  CHECK((code == core::ErrorCode::Invalid || code == core::ErrorCode::NotFound));
  CHECK_FALSE(response.error().message.empty());
  CHECK(response.error().message != "OpenAI API error");
  CHECK_THAT(response.error().context, ContainsSubstring("status 4"));
}

TEST_CASE("live: the backend emits a native tool call", "[.live][llm][tools]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  llm::ToolDef weather;
  weather.name = "get_weather";
  weather.description = "Get the current weather for a city.";
  weather.input_schema = nlohmann::json{
      {"type", "object"},
      {"properties", {{"city", {{"type", "string"}}}}},
      {"required", nlohmann::json::array({"city"})}};

  llm::ChatRequest request;
  request.tools = {weather};
  request.messages = {llm::Message::user_text("What is the weather in Waterloo, Ontario?")};
  request.sampling.max_tokens = 256;

  const auto response = live->backend->generate(request);
  REQUIRE(response.has_value());

  const auto uses = response->tool_uses();
  REQUIRE_FALSE(uses.empty());
  CHECK(uses.front().name == "get_weather");
  CHECK(response->stop_reason == llm::StopReason::ToolUse);
  REQUIRE(uses.front().input.contains("city"));
  CHECK_THAT(uses.front().input["city"].get<std::string>(), ContainsSubstring("Waterloo"));
}

TEST_CASE("live: a tool call round-trips back into a final answer", "[.live][llm][tools]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  llm::ToolDef lookup;
  lookup.name = "get_stock_level";
  lookup.description = "Look up the current stock level for a part number.";
  lookup.input_schema = nlohmann::json{
      {"type", "object"},
      {"properties", {{"part", {{"type", "string"}}}}},
      {"required", nlohmann::json::array({"part"})}};

  llm::ChatRequest request;
  request.tools = {lookup};
  request.messages = {llm::Message::user_text("How many of part XZ-9 are in stock?")};
  request.sampling.max_tokens = 256;

  const auto first = live->backend->generate(request);
  REQUIRE(first.has_value());
  const auto uses = first->tool_uses();
  REQUIRE_FALSE(uses.empty());

  request.messages.push_back(llm::Message{llm::Role::Assistant, first->content});
  request.messages.push_back(
      llm::Message{llm::Role::User, {llm::ToolResultBlock{uses.front().id, "42", false}}});

  const auto second = live->backend->generate(request);
  REQUIRE(second.has_value());
  CHECK_THAT(second->text(), ContainsSubstring("42"));
}

TEST_CASE("live: a ReAct graph completes a tool loop end to end", "[.live][graph][tools]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
  using AgentSchema = graph::Schema<Messages>;

  auto registry = std::make_shared<tools::ToolRegistry>();
  llm::ToolDef multiply;
  multiply.name = "multiply";
  multiply.description = "Multiply two integers exactly. Always use this instead of "
                         "doing arithmetic yourself.";
  multiply.input_schema = nlohmann::json{
      {"type", "object"},
      {"properties", {{"a", {{"type", "number"}}}, {"b", {{"type", "number"}}}}},
      {"required", nlohmann::json::array({"a", "b"})}};

  int tool_calls = 0;
  REQUIRE(registry
              ->add(tools::make_tool(std::move(multiply),
                                     [&](const nlohmann::json& args) -> core::Result<std::string> {
                                       ++tool_calls;
                                       const long long a = args.at("a").get<long long>();
                                       const long long b = args.at("b").get<long long>();
                                       return std::to_string(a * b);
                                     }))
              .has_value());

  graph::LlmNodeOptions options;
  options.system = "Use the multiply tool for arithmetic, then state the result.";
  options.tools = registry->defs();
  options.sampling.max_tokens = 512;

  graph::GraphBuilder<AgentSchema> builder;
  builder.add_node("agent", graph::make_llm_node<AgentSchema>(live->backend, options))
      .add_node("tools", graph::make_tool_node<AgentSchema>(registry))
      .set_entry("agent")
      .add_conditional_edge("agent", graph::tools_router<AgentSchema>("tools"),
                            {"tools", std::string{graph::kEnd}})
      .add_edge("tools", "agent");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled.has_value());

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("What is 987654321 times 123456789?")});

  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.max_steps = 10});
  REQUIRE(stats.has_value());

  CHECK(tool_calls >= 1);

  std::string final_text;
  for (const auto& block : state.get<"messages">().back().content) {
    if (const auto* text = std::get_if<llm::TextBlock>(&block)) final_text += text->text;
  }
  INFO("final answer: " << final_text);
  CHECK_THAT(final_text, ContainsSubstring("121932631112635269"));
}
