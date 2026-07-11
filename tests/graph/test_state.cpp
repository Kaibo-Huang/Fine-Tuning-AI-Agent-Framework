#include <catch2/catch_test_macros.hpp>

#include <any>
#include <concepts>
#include <string>
#include <vector>

#include "agents_framework/graph/channel.hpp"
#include "agents_framework/graph/channel_map.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/llm/message.hpp"

namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;

namespace {

struct Sum {
  static void apply(int& current, int update) { current += update; }
};

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using Goal = graph::Channel<"goal", std::string>;
using Steps = graph::Channel<"steps", int>;
using Total = graph::Channel<"total", int, Sum>;

using AgentSchema = graph::Schema<Messages, Goal, Steps, Total>;

}

TEST_CASE("Schema resolves channels at compile time", "[graph][state]") {
  STATIC_REQUIRE(AgentSchema::size == 4);
  STATIC_REQUIRE(AgentSchema::has<"messages">);
  STATIC_REQUIRE(!AgentSchema::has<"missing">);
  STATIC_REQUIRE(AgentSchema::index_of<"messages">() == 0);
  STATIC_REQUIRE(AgentSchema::index_of<"total">() == 3);
  STATIC_REQUIRE(std::same_as<AgentSchema::value_of<"steps">, int>);
}

TEST_CASE("reducer concepts gate channel declarations", "[graph][state]") {
  STATIC_REQUIRE(graph::ReducerFor<graph::Append, std::vector<int>>);
  STATIC_REQUIRE(!graph::ReducerFor<graph::Append, int>);
  STATIC_REQUIRE(graph::ReducerFor<graph::LastValue, std::string>);
  STATIC_REQUIRE(graph::ReducerFor<Sum, int>);
}

TEST_CASE("State channels start default-constructed", "[graph][state]") {
  const graph::State<AgentSchema> state;
  REQUIRE(state.get<"messages">().empty());
  REQUIRE(state.get<"goal">().empty());
  REQUIRE(state.get<"steps">() == 0);
}

TEST_CASE("set and get round-trip typed values", "[graph][state]") {
  graph::State<AgentSchema> state;
  state.set<"goal">("book a flight");
  state.set<"steps">(3);
  REQUIRE(state.get<"goal">() == "book a flight");
  REQUIRE(state.get<"steps">() == 3);
}

TEST_CASE("reduce merges through the channel reducer", "[graph][state]") {
  graph::State<AgentSchema> state;

  SECTION("LastValue replaces the current value") {
    state.reduce<"steps">(1);
    state.reduce<"steps">(2);
    REQUIRE(state.get<"steps">() == 2);
  }

  SECTION("Append concatenates message updates") {
    state.reduce<"messages">({llm::Message::user_text("hi")});
    state.reduce<"messages">({llm::Message::assistant_text("hello")});
    REQUIRE(state.get<"messages">().size() == 2);
    REQUIRE(state.get<"messages">().front() == llm::Message::user_text("hi"));
  }

  SECTION("a custom reducer is applied") {
    state.reduce<"total">(2);
    state.reduce<"total">(40);
    REQUIRE(state.get<"total">() == 42);
  }
}

TEST_CASE("Update buffers writes and apply merges them in order", "[graph][state]") {
  graph::State<AgentSchema> state;
  state.set<"steps">(1);

  graph::Update<AgentSchema> update;
  update.write<"messages">({llm::Message::user_text("one")})
      .write<"messages">({llm::Message::user_text("two")})
      .write<"steps">(2)
      .write<"steps">(7);
  REQUIRE(update.size() == 4);

  state.apply(std::move(update));

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 2);
  REQUIRE(std::get<llm::TextBlock>(messages[0].content.front()).text == "one");
  REQUIRE(std::get<llm::TextBlock>(messages[1].content.front()).text == "two");
  REQUIRE(state.get<"steps">() == 7);
}

TEST_CASE("the erased ChannelMap drives the same state uniformly", "[graph][state]") {
  graph::ChannelMap map{AgentSchema::channels()};
  REQUIRE(map.size() == AgentSchema::size);
  REQUIRE(map.channels()[0].name == "messages");

  graph::StateUpdate update;
  update.write(AgentSchema::index_of<"steps">(), std::any{7});
  update.write(AgentSchema::index_of<"total">(), std::any{5});
  update.write(AgentSchema::index_of<"total">(), std::any{6});
  map.apply(std::move(update));

  const graph::StateView<AgentSchema> view{map};
  REQUIRE(view.get<"steps">() == 7);
  REQUIRE(view.get<"total">() == 11);
}

TEST_CASE("copies are independent snapshots", "[graph][state]") {
  graph::State<AgentSchema> state;
  state.set<"steps">(1);

  graph::State<AgentSchema> snapshot = state;
  state.set<"steps">(2);

  REQUIRE(snapshot.get<"steps">() == 1);
  REQUIRE(state.get<"steps">() == 2);
}
