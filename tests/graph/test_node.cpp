#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/llm/message.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;

namespace {

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using Steps = graph::Channel<"steps", int>;
using AgentSchema = graph::Schema<Messages, Steps>;

}

TEST_CASE("a function node reads state and its writes update it", "[graph][node]") {
  const auto node = graph::make_fn_node<AgentSchema>([](graph::StateView<AgentSchema> view) {
    graph::Update<AgentSchema> update;
    update.write<"steps">(view.get<"steps">() + 1)
        .write<"messages">({llm::Message::assistant_text("stepped")});
    return update;
  });

  graph::State<AgentSchema> state;
  state.set<"steps">(41);

  auto result = node->run(state.map());
  REQUIRE(result);
  REQUIRE(state.get<"steps">() == 41);

  state.apply(std::move(*result));
  REQUIRE(state.get<"steps">() == 42);
  REQUIRE(state.get<"messages">().size() == 1);
}

TEST_CASE("a transform node can be a one-liner returning its update", "[graph][node]") {
  const auto node = graph::make_fn_node<AgentSchema>([](graph::StateView<AgentSchema> view) {
    return graph::Update<AgentSchema>{}.write<"steps">(view.get<"steps">() * 2);
  });

  graph::State<AgentSchema> state;
  state.set<"steps">(21);

  auto result = node->run(state.map());
  REQUIRE(result);
  state.apply(std::move(*result));
  REQUIRE(state.get<"steps">() == 42);
}

TEST_CASE("a node returning Result can fail and the error propagates", "[graph][node]") {
  const auto node = graph::make_fn_node<AgentSchema>(
      [](graph::StateView<AgentSchema> view) -> core::Result<graph::Update<AgentSchema>> {
        if (view.get<"steps">() > 3) {
          return core::fail(core::ErrorCode::Invalid, "step budget exhausted");
        }
        return graph::Update<AgentSchema>{}.write<"steps">(view.get<"steps">() + 1);
      });

  graph::State<AgentSchema> state;

  auto ok = node->run(state.map());
  REQUIRE(ok);

  state.set<"steps">(4);
  auto failed = node->run(state.map());
  REQUIRE(!failed);
  REQUIRE(failed.error().code == core::ErrorCode::Invalid);
  REQUIRE(failed.error().message == "step budget exhausted");
}

TEST_CASE("nodes are driven uniformly through the erased interface", "[graph][node]") {
  std::vector<std::unique_ptr<graph::Node>> nodes;
  nodes.push_back(graph::make_fn_node<AgentSchema>([](graph::StateView<AgentSchema>) {
    return graph::Update<AgentSchema>{}.write<"messages">({llm::Message::user_text("plan")});
  }));
  nodes.push_back(graph::make_fn_node<AgentSchema>([](graph::StateView<AgentSchema>) {
    return graph::Update<AgentSchema>{}.write<"messages">({llm::Message::assistant_text("act")});
  }));

  graph::State<AgentSchema> state;
  for (const auto& node : nodes) {
    auto result = node->run(state.map());
    REQUIRE(result);
    state.apply(std::move(*result));
  }

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 2);
  REQUIRE(messages.front().role == llm::Role::User);
  REQUIRE(messages.back().role == llm::Role::Assistant);
}

TEST_CASE("an empty update leaves state untouched", "[graph][node]") {
  const auto node = graph::make_fn_node<AgentSchema>(
      [](graph::StateView<AgentSchema>) { return graph::Update<AgentSchema>{}; });

  graph::State<AgentSchema> state;
  state.set<"steps">(9);

  auto result = node->run(state.map());
  REQUIRE(result);
  REQUIRE(result->empty());
  state.apply(std::move(*result));
  REQUIRE(state.get<"steps">() == 9);
}
