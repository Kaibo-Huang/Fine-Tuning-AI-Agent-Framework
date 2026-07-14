#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;

namespace {

using Steps = graph::Channel<"steps", int>;
using DemoSchema = graph::Schema<Steps>;

graph::Update<DemoSchema> bump(graph::StateView<DemoSchema> view) {
  return graph::Update<DemoSchema>{}.write<"steps">(view.get<"steps">() + 1);
}

core::Error compile_error(graph::GraphBuilder<DemoSchema>&& builder) {
  auto compiled = std::move(builder).compile();
  REQUIRE(!compiled);
  return compiled.error();
}

}

TEST_CASE("a valid linear graph compiles with ordered topology", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump)
      .add_node("act", bump)
      .set_entry("plan")
      .add_edge("plan", "act")
      .set_finish("act");

  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);
  REQUIRE(compiled->node_count() == 2);
  REQUIRE(compiled->node_name(0) == "plan");
  REQUIRE(compiled->find_node("act") == 1);
  REQUIRE(!compiled->find_node("missing"));

  REQUIRE(compiled->entry_nodes().size() == 1);
  REQUIRE(compiled->entry_nodes()[0] == 0);
  REQUIRE(compiled->static_targets(0).size() == 1);
  REQUIRE(compiled->static_targets(0)[0] == 1);
  REQUIRE(compiled->static_targets(1).empty());
}

TEST_CASE("fan-out targets are deduplicated and ordered by node index", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("split", bump)
      .add_node("right", bump)
      .add_node("left", bump)
      .set_entry("split")
      .add_edge("split", "right")
      .add_edge("split", "left")
      .set_finish("left")
      .set_finish("right");

  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);
  const auto targets = compiled->static_targets(0);
  REQUIRE(targets.size() == 2);
  REQUIRE(targets[0] == 1);
  REQUIRE(targets[1] == 2);
}

TEST_CASE("an empty graph is rejected", "[graph][compile]") {
  const auto error = compile_error(graph::GraphBuilder<DemoSchema>{});
  REQUIRE(error.code == core::ErrorCode::Invalid);
  REQUIRE(error.message == "graph has no nodes");
}

TEST_CASE("reserved and duplicate node names are rejected", "[graph][compile]") {
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node(std::string{graph::kEnd}, bump);
    const auto error = compile_error(std::move(builder));
    REQUIRE(error.message == "node name is reserved");
    REQUIRE(error.context == graph::kEnd);
  }
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("plan", bump).add_node("plan", bump);
    const auto error = compile_error(std::move(builder));
    REQUIRE(error.message == "duplicate node name");
    REQUIRE(error.context == "plan");
  }
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("plan", nullptr);
    const auto error = compile_error(std::move(builder));
    REQUIRE(error.message == "node has no implementation");
  }
}

TEST_CASE("edges referencing unknown nodes are rejected", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump).set_entry("plan").add_edge("plan", "ghost");
  const auto error = compile_error(std::move(builder));
  REQUIRE(error.message == "edge references unknown node");
  REQUIRE(error.context == "ghost");
}

TEST_CASE("a graph without an entry point is rejected", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump).set_finish("plan");
  const auto error = compile_error(std::move(builder));
  REQUIRE(error.message == "graph has no entry point; add an edge from __start__");
}

TEST_CASE("unreachable nodes are rejected", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump)
      .add_node("island", bump)
      .set_entry("plan")
      .set_finish("plan")
      .set_finish("island");
  const auto error = compile_error(std::move(builder));
  REQUIRE(error.message == "node is unreachable from __start__");
  REQUIRE(error.context == "island");
}

TEST_CASE("dead ends are rejected", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump).set_entry("plan");
  const auto error = compile_error(std::move(builder));
  REQUIRE(error.message == "node is a dead end; add an edge to __end__ or another node");
  REQUIRE(error.context == "plan");
}

TEST_CASE("static cycles with no exit are rejected", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("ping", bump)
      .add_node("pong", bump)
      .set_entry("ping")
      .add_edge("ping", "pong")
      .add_edge("pong", "ping");
  const auto error = compile_error(std::move(builder));
  REQUIRE(error.message == "node has no path to __end__");
}

TEST_CASE("edges into __start__ or out of __end__ are rejected", "[graph][compile]") {
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("plan", bump).add_edge("plan", std::string{graph::kStart});
    REQUIRE(compile_error(std::move(builder)).message == "edge may not target __start__");
  }
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("plan", bump).add_edge(std::string{graph::kEnd}, "plan");
    REQUIRE(compile_error(std::move(builder)).message == "edge may not leave __end__");
  }
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("plan", bump).add_edge(std::string{graph::kStart}, std::string{graph::kEnd});
    REQUIRE(compile_error(std::move(builder)).message ==
            "edge from __start__ must target a node");
  }
}

TEST_CASE("compiled nodes stay runnable", "[graph][compile]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump).set_entry("plan").set_finish("plan");

  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  auto update = compiled->node(0).run(state.map());
  REQUIRE(update);
  state.apply(std::move(*update));
  REQUIRE(state.get<"steps">() == 1);
}
