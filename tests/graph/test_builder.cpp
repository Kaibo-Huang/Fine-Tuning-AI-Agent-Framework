#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/node.hpp"

namespace graph = agents_framework::graph;

namespace {

using Steps = graph::Channel<"steps", int>;
using DemoSchema = graph::Schema<Steps>;

graph::Update<DemoSchema> bump(graph::StateView<DemoSchema> view) {
  return graph::Update<DemoSchema>{}.write<"steps">(view.get<"steps">() + 1);
}

bool contains(std::span<const graph::Edge> edges, std::string_view from, std::string_view to) {
  return std::any_of(edges.begin(), edges.end(), [&](const graph::Edge& edge) {
    return edge.from == from && edge.to == to;
  });
}

}

TEST_CASE("a linear graph is constructed from nodes and static edges", "[graph][builder]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump)
      .add_node("act", bump)
      .add_node("summarize", bump)
      .set_entry("plan")
      .add_edge("plan", "act")
      .add_edge("act", "summarize")
      .set_finish("summarize");

  REQUIRE(builder.node_count() == 3);
  REQUIRE(builder.has_node("plan"));
  REQUIRE(builder.has_node("act"));
  REQUIRE(builder.has_node("summarize"));
  REQUIRE(!builder.has_node("missing"));

  const auto edges = builder.edges();
  REQUIRE(edges.size() == 4);
  REQUIRE(contains(edges, graph::kStart, "plan"));
  REQUIRE(contains(edges, "plan", "act"));
  REQUIRE(contains(edges, "act", "summarize"));
  REQUIRE(contains(edges, "summarize", graph::kEnd));
}

TEST_CASE("duplicate static edges are stored once", "[graph][builder]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", bump).add_edge("plan", "plan").add_edge("plan", "plan");

  REQUIRE(builder.edges().size() == 1);
}

TEST_CASE("prebuilt nodes are added through the erased overload", "[graph][builder]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", graph::make_fn_node<DemoSchema>(bump));

  REQUIRE(builder.node_count() == 1);
  REQUIRE(builder.has_node("plan"));
}

TEST_CASE("entry and finish are sugar over start and end edges", "[graph][builder]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("only", bump).set_entry("only").set_finish("only");

  const auto edges = builder.edges();
  REQUIRE(edges.size() == 2);
  REQUIRE(contains(edges, graph::kStart, "only"));
  REQUIRE(contains(edges, "only", graph::kEnd));
}
