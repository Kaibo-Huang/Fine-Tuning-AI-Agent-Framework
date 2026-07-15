#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;

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

}

TEST_CASE("a linear graph runs end to end in order", "[graph][executor]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", log_node("plan"))
      .add_node("act", log_node("act"))
      .add_node("summarize", log_node("summarize"))
      .set_entry("plan")
      .add_edge("plan", "act")
      .add_edge("act", "summarize")
      .set_finish("summarize");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(stats);
  REQUIRE(stats->steps == 3);
  REQUIRE(stats->node_runs == 3);
  REQUIRE(state.get<"steps">() == 3);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"plan", "act", "summarize"});
}

TEST_CASE("fan-out branches run in one super-step and merge deterministically",
          "[graph][executor]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("split", log_node("split"))
      .add_node("left", log_node("left"))
      .add_node("right", log_node("right"))
      .add_node("join", log_node("join"))
      .set_entry("split")
      .add_edge("split", "right")
      .add_edge("split", "left")
      .add_edge("left", "join")
      .add_edge("right", "join")
      .set_finish("join");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(stats);
  REQUIRE(stats->steps == 3);
  REQUIRE(stats->node_runs == 4);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"split", "left", "right", "join"});
}

TEST_CASE("parallel branches read the same snapshot and the join runs once",
          "[graph][executor]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("split", log_node("split"))
      .add_node("left", log_node("left"))
      .add_node("right", log_node("right"))
      .add_node("join", log_node("join"))
      .set_entry("split")
      .add_edge("split", "left")
      .add_edge("split", "right")
      .add_edge("left", "join")
      .add_edge("right", "join")
      .set_finish("join");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(stats);
  REQUIRE(stats->node_runs == 4);
  REQUIRE(state.get<"steps">() == 3);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"split", "left", "right", "join"});
}

TEST_CASE("node errors abort the run with node and step context", "[graph][executor]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", log_node("plan"))
      .add_node("explode",
                [](graph::StateView<DemoSchema>) -> core::Result<graph::Update<DemoSchema>> {
                  return core::fail(core::ErrorCode::Tool, "boom");
                })
      .set_entry("plan")
      .add_edge("plan", "explode")
      .set_finish("explode");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(!stats);
  REQUIRE(stats.error().code == core::ErrorCode::Tool);
  REQUIRE(stats.error().message == "boom");
  REQUIRE(stats.error().context == "node 'explode' (step 2)");
  REQUIRE(state.get<"log">() == std::vector<std::string>{"plan"});
}

TEST_CASE("a failed super-step applies none of its buffered updates", "[graph][executor]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("split", log_node("split"))
      .add_node("ok", log_node("ok"))
      .add_node("bad",
                [](graph::StateView<DemoSchema>) -> core::Result<graph::Update<DemoSchema>> {
                  return core::fail(core::ErrorCode::Tool, "boom");
                })
      .set_entry("split")
      .add_edge("split", "ok")
      .add_edge("split", "bad")
      .set_finish("ok")
      .set_finish("bad");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(!stats);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"split"});
}

TEST_CASE("the step budget bounds a run", "[graph][executor]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("a", log_node("a"))
      .add_node("b", log_node("b"))
      .set_entry("a")
      .add_edge("a", "b")
      .add_edge("b", "a")
      .set_finish("b");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.max_steps = 5});
  REQUIRE(!stats);
  REQUIRE(stats.error().code == core::ErrorCode::Cancelled);
  REQUIRE(stats.error().message == "step budget exhausted");
  REQUIRE(state.get<"steps">() == 5);
}

TEST_CASE("repeated runs over the same compiled graph are deterministic", "[graph][executor]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("split", log_node("split"))
      .add_node("left", log_node("left"))
      .add_node("right", log_node("right"))
      .set_entry("split")
      .add_edge("split", "left")
      .add_edge("split", "right")
      .set_finish("left")
      .set_finish("right");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::Executor executor;
  graph::State<DemoSchema> first;
  graph::State<DemoSchema> second;
  REQUIRE(executor.run(*compiled, first));
  REQUIRE(executor.run(*compiled, second));
  REQUIRE(first.serialize() == second.serialize());
}
