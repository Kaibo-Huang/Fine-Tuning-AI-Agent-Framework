#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/events.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/llm/mock_backend.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;

namespace {

using Steps = graph::Channel<"steps", int>;
using Log = graph::Channel<"log", std::vector<std::string>, graph::Append>;
using DemoSchema = graph::Schema<Steps, Log>;

auto log_node(std::string label) {
  return [label = std::move(label)](graph::StateView<DemoSchema> view) {
    return graph::Update<DemoSchema>{}
        .write<"steps">(view.get<"steps">() + 1)
        .write<"log">({label});
  };
}

template <class T>
std::vector<T> events_of(const std::vector<graph::ExecEvent>& events) {
  std::vector<T> found;
  for (const auto& event : events) {
    if (const auto* value = std::get_if<T>(&event)) found.push_back(*value);
  }
  return found;
}

}

TEST_CASE("a run publishes a deterministic event stream", "[graph][events]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", log_node("plan"))
      .add_node("act", log_node("act"))
      .set_entry("plan")
      .add_edge("plan", "act")
      .set_finish("act");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  auto bus = std::make_shared<graph::EventBus>();
  std::vector<graph::ExecEvent> events;
  bus->subscribe([&events](const graph::ExecEvent& event) { events.push_back(event); });

  graph::State<DemoSchema> state;
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state, graph::RunOptions{.run_id = "run-1", .events = bus}));

  const auto started = events_of<graph::RunStarted>(events);
  REQUIRE(started.size() == 1);
  REQUIRE(started.front().run_id == "run-1");

  const auto steps = events_of<graph::StepStarted>(events);
  REQUIRE(steps.size() == 2);
  REQUIRE(steps[0].nodes == std::vector<std::string>{"plan"});
  REQUIRE(steps[1].nodes == std::vector<std::string>{"act"});

  const auto nodes = events_of<graph::NodeFinished>(events);
  REQUIRE(nodes.size() == 2);
  REQUIRE(nodes[0].node == "plan");
  REQUIRE(nodes[0].ok);
  REQUIRE(nodes[1].node == "act");

  const auto states = events_of<graph::StateUpdated>(events);
  REQUIRE(states.size() == 2);
  REQUIRE(states[1].state["log"] == nlohmann::json::array({"plan", "act"}));

  const auto finished = events_of<graph::RunFinished>(events);
  REQUIRE(finished.size() == 1);
  REQUIRE(finished.front().ok);

  REQUIRE(std::holds_alternative<graph::RunStarted>(events.front()));
  REQUIRE(std::holds_alternative<graph::RunFinished>(events.back()));
}

TEST_CASE("a failing node publishes its error before the run finishes", "[graph][events]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("explode",
                   [](graph::StateView<DemoSchema>) -> core::Result<graph::Update<DemoSchema>> {
                     return core::fail(core::ErrorCode::Tool, "boom");
                   })
      .set_entry("explode")
      .set_finish("explode");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  auto bus = std::make_shared<graph::EventBus>();
  std::vector<graph::ExecEvent> events;
  bus->subscribe([&events](const graph::ExecEvent& event) { events.push_back(event); });

  graph::State<DemoSchema> state;
  graph::Executor executor;
  REQUIRE(!executor.run(*compiled, state, graph::RunOptions{.events = bus}));

  const auto nodes = events_of<graph::NodeFinished>(events);
  REQUIRE(nodes.size() == 1);
  REQUIRE(!nodes.front().ok);
  REQUIRE(nodes.front().error.find("boom") != std::string::npos);

  const auto finished = events_of<graph::RunFinished>(events);
  REQUIRE(finished.size() == 1);
  REQUIRE(!finished.front().ok);
}

TEST_CASE("unsubscribing stops delivery", "[graph][events]") {
  graph::EventBus bus;
  int calls = 0;
  const auto id = bus.subscribe([&calls](const graph::ExecEvent&) { ++calls; });
  bus.publish(graph::RunStarted{"run", 0});
  bus.unsubscribe(id);
  bus.publish(graph::RunStarted{"run", 0});
  REQUIRE(calls == 1);
  REQUIRE(!bus.has_subscribers());
}

TEST_CASE("an llm node streams token deltas into the event bus", "[graph][events]") {
  using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
  using AgentSchema = graph::Schema<Messages>;

  auto backend = std::make_shared<llm::MockBackend>();
  llm::ChatResponse response;
  response.id = "msg-1";
  response.content.push_back(llm::TextBlock{"Hello world!"});
  backend->push_response(response);

  auto bus = std::make_shared<graph::EventBus>();
  std::vector<graph::TokenDelta> tokens;
  bus->subscribe([&tokens](const graph::ExecEvent& event) {
    if (const auto* delta = std::get_if<graph::TokenDelta>(&event)) tokens.push_back(*delta);
  });

  graph::GraphBuilder<AgentSchema> builder;
  builder
      .add_node("llm", graph::make_llm_node<AgentSchema>(
                           backend, graph::LlmNodeOptions{.events = bus, .label = "writer"}))
      .set_entry("llm")
      .set_finish("llm");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("hi")});
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state, graph::RunOptions{.events = bus}));

  REQUIRE(!tokens.empty());
  std::string streamed;
  for (const auto& token : tokens) {
    REQUIRE(token.node == "writer");
    streamed += token.text;
  }
  REQUIRE(streamed == "Hello world!");
}
