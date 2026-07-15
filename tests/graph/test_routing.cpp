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

TEST_CASE("a conditional edge picks one branch from state", "[graph][routing]") {
  const auto build = [](int start) {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("classify", log_node("classify"))
        .add_node("small", log_node("small"))
        .add_node("large", log_node("large"))
        .set_entry("classify")
        .add_conditional_edge(
            "classify",
            [](graph::StateView<DemoSchema> view) -> std::string {
              return view.get<"steps">() > 10 ? "large" : "small";
            },
            {"small", "large"})
        .set_finish("small")
        .set_finish("large");
    auto compiled = std::move(builder).compile();
    REQUIRE(compiled);
    graph::State<DemoSchema> state;
    state.set<"steps">(start);
    graph::Executor executor;
    REQUIRE(executor.run(*compiled, state));
    return state.get<"log">();
  };

  REQUIRE(build(0) == std::vector<std::string>{"classify", "small"});
  REQUIRE(build(50) == std::vector<std::string>{"classify", "large"});
}

TEST_CASE("a router can fan out to several targets at once", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", log_node("plan"))
      .add_node("left", log_node("left"))
      .add_node("right", log_node("right"))
      .set_entry("plan")
      .add_conditional_edge(
          "plan",
          [](graph::StateView<DemoSchema>) {
            return std::vector<std::string>{"left", "right"};
          },
          {"left", "right"})
      .set_finish("left")
      .set_finish("right");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(stats);
  REQUIRE(stats->steps == 2);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"plan", "left", "right"});
}

TEST_CASE("a cycle loops until its router exits", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("work", log_node("work"))
      .set_entry("work")
      .add_conditional_edge(
          "work",
          [](graph::StateView<DemoSchema> view) -> std::string {
            return view.get<"steps">() < 3 ? "work" : std::string{graph::kEnd};
          },
          {"work", std::string{graph::kEnd}});
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(stats);
  REQUIRE(stats->steps == 3);
  REQUIRE(state.get<"steps">() == 3);
}

TEST_CASE("the step budget stops a router-driven infinite loop", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("work", log_node("work"))
      .set_entry("work")
      .add_conditional_edge("work",
                            [](graph::StateView<DemoSchema>) -> std::string { return "work"; });
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.max_steps = 7});
  REQUIRE(!stats);
  REQUIRE(stats.error().code == core::ErrorCode::Cancelled);
  REQUIRE(state.get<"steps">() == 7);
}

TEST_CASE("router errors abort the run with node and step context", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("work", log_node("work"))
      .set_entry("work")
      .add_conditional_edge(
          "work", [](graph::StateView<DemoSchema>) -> core::Result<std::string> {
            return core::fail(core::ErrorCode::Invalid, "cannot decide");
          });
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(!stats);
  REQUIRE(stats.error().message == "cannot decide");
  REQUIRE(stats.error().context == "node 'work' (step 1)");
}

TEST_CASE("routing outside declared targets fails at run time", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("work", log_node("work"))
      .add_node("other", log_node("other"))
      .set_entry("work")
      .add_edge("work", "other")
      .add_conditional_edge("work",
                            [](graph::StateView<DemoSchema>) -> std::string { return "other"; },
                            {std::string{graph::kEnd}})
      .set_finish("other");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(!stats);
  REQUIRE(stats.error().message == "conditional edge returned undeclared target");
  REQUIRE(stats.error().context == "other; node 'work' (step 1)");
}

TEST_CASE("routing to an unknown node fails at run time", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("work", log_node("work"))
      .set_entry("work")
      .add_conditional_edge("work",
                            [](graph::StateView<DemoSchema>) -> std::string { return "ghost"; });
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(!stats);
  REQUIRE(stats.error().message == "conditional edge returned unknown target");
}

TEST_CASE("conditional edges participate in compile-time validation", "[graph][routing]") {
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("work", log_node("work"))
        .set_entry("work")
        .add_conditional_edge("ghost",
                              [](graph::StateView<DemoSchema>) -> std::string { return "work"; });
    auto compiled = std::move(builder).compile();
    REQUIRE(!compiled);
    REQUIRE(compiled.error().message == "conditional edge references unknown node");
  }
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("work", log_node("work"))
        .set_entry("work")
        .add_conditional_edge("work",
                              [](graph::StateView<DemoSchema>) -> std::string { return "work"; })
        .add_conditional_edge("work", [](graph::StateView<DemoSchema>) -> std::string {
          return std::string{graph::kEnd};
        });
    auto compiled = std::move(builder).compile();
    REQUIRE(!compiled);
    REQUIRE(compiled.error().message == "node already has a conditional edge");
  }
  {
    graph::GraphBuilder<DemoSchema> builder;
    builder.add_node("work", log_node("work"))
        .set_entry("work")
        .add_conditional_edge(
            "work", [](graph::StateView<DemoSchema>) -> std::string { return "work"; },
            {"ghost"});
    auto compiled = std::move(builder).compile();
    REQUIRE(!compiled);
    REQUIRE(compiled.error().message == "conditional edge declares unknown target");
  }
}

TEST_CASE("declared router targets satisfy reachability and termination", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("decide", log_node("decide"))
      .add_node("only_via_router", log_node("only_via_router"))
      .set_entry("decide")
      .add_conditional_edge(
          "decide",
          [](graph::StateView<DemoSchema>) -> std::string { return "only_via_router"; },
          {"only_via_router"})
      .set_finish("only_via_router");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));
  REQUIRE(state.get<"log">() ==
          std::vector<std::string>{"decide", "only_via_router"});
}

TEST_CASE("an undeclared router is assumed able to reach anything", "[graph][routing]") {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("a", log_node("a"))
      .add_node("b", log_node("b"))
      .set_entry("a")
      .add_conditional_edge("a", [](graph::StateView<DemoSchema> view) -> std::string {
        return view.get<"steps">() < 2 ? "b" : std::string{graph::kEnd};
      })
      .add_edge("b", "a");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(stats);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"a", "b", "a"});
}
