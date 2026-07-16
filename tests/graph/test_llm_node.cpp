#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/llm/mock_backend.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;

namespace {

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using AgentSchema = graph::Schema<Messages>;

llm::ChatResponse text_response(std::string text) {
  llm::ChatResponse response;
  response.id = "resp";
  response.content = {llm::TextBlock{std::move(text)}};
  return response;
}

}

TEST_CASE("the llm node sends the conversation and appends the reply", "[graph][llm_node]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->push_response(text_response("hello there"));

  graph::LlmNodeOptions options;
  options.model = "test-model";
  options.system = "be brief";
  options.sampling.max_tokens = 64;
  const auto node = graph::make_llm_node<AgentSchema>(backend, options);

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("hi")});

  auto update = node->run(state.map());
  REQUIRE(update);
  state.apply(std::move(*update));

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 2);
  REQUIRE(messages.back().role == llm::Role::Assistant);
  REQUIRE(messages.back().content == std::vector<llm::ContentBlock>{llm::TextBlock{"hello there"}});

  REQUIRE(backend->calls().size() == 1);
  const auto& request = backend->calls().front();
  REQUIRE(request.model == "test-model");
  REQUIRE(request.system == "be brief");
  REQUIRE(request.sampling.max_tokens == 64);
  REQUIRE(request.messages == std::vector<llm::Message>{llm::Message::user_text("hi")});
}

TEST_CASE("tool definitions from the options reach the request", "[graph][llm_node]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->push_response(text_response("ok"));

  llm::ToolDef weather;
  weather.name = "get_weather";
  weather.description = "Get the weather.";
  weather.input_schema = {{"type", "object"}};

  graph::LlmNodeOptions options;
  options.tools = {weather};
  const auto node = graph::make_llm_node<AgentSchema>(backend, options);

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("weather?")});
  REQUIRE(node->run(state.map()));
  REQUIRE(backend->calls().front().tools == std::vector<llm::ToolDef>{weather});
}

TEST_CASE("constructing an llm node without a backend throws", "[graph][llm_node]") {
  REQUIRE_THROWS_AS(graph::make_llm_node<AgentSchema>(nullptr), std::invalid_argument);
}

TEST_CASE("backend errors propagate out of the llm node", "[graph][llm_node]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->push_error(core::Error{core::ErrorCode::RateLimited, "slow down", ""});

  const auto node = graph::make_llm_node<AgentSchema>(backend);
  graph::State<AgentSchema> state;

  auto update = node->run(state.map());
  REQUIRE(!update);
  REQUIRE(update.error().code == core::ErrorCode::RateLimited);
}

TEST_CASE("an llm node runs inside a graph", "[graph][llm_node]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->push_response(text_response("42"));

  graph::GraphBuilder<AgentSchema> builder;
  builder.add_node("llm", graph::make_llm_node<AgentSchema>(backend))
      .set_entry("llm")
      .set_finish("llm");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("meaning of life?")});
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));

  REQUIRE(state.get<"messages">().size() == 2);
  REQUIRE(state.get<"messages">().back().role == llm::Role::Assistant);
}
