#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/channel.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/llm/message.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;

namespace {

struct IntStringSerde {
  static nlohmann::json to_json(const int& value) { return std::to_string(value); }
  static int from_json(const nlohmann::json& j) { return std::stoi(j.get<std::string>()); }
};

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using Steps = graph::Channel<"steps", int>;
using Attempts = graph::Channel<"attempts", int, graph::LastValue, IntStringSerde>;

using AgentSchema = graph::Schema<Steps, Messages, Attempts>;

graph::State<AgentSchema> make_state() {
  graph::State<AgentSchema> state;
  state.set<"steps">(5);
  state.set<"attempts">(7);
  state.set<"messages">({
      llm::Message::user_text("find flights"),
      llm::Message{llm::Role::Assistant,
                   {llm::TextBlock{"searching"},
                    llm::ToolUseBlock{"tu_1", "search", nlohmann::json{{"q", "yyz to sfo"}}}}},
  });
  return state;
}

}

TEST_CASE("a full state round-trips through JSON", "[graph][serde]") {
  const graph::State<AgentSchema> state = make_state();

  const nlohmann::json j = state.to_json();
  REQUIRE(j["steps"] == 5);
  REQUIRE(j["messages"].is_array());

  const auto restored = graph::State<AgentSchema>::from_json(j);
  REQUIRE(restored);
  REQUIRE(restored->get<"steps">() == 5);
  REQUIRE(restored->get<"attempts">() == 7);
  REQUIRE(restored->get<"messages">() == state.get<"messages">());
}

TEST_CASE("a full state round-trips through bytes", "[graph][serde]") {
  const graph::State<AgentSchema> state = make_state();

  const std::string bytes = state.serialize();
  const auto restored = graph::State<AgentSchema>::deserialize(bytes);
  REQUIRE(restored);
  REQUIRE(restored->serialize() == bytes);
  REQUIRE(restored->get<"messages">() == state.get<"messages">());
}

TEST_CASE("a custom serde controls the wire format of its channel", "[graph][serde]") {
  const graph::State<AgentSchema> state = make_state();
  const nlohmann::json j = state.to_json();
  REQUIRE(j["attempts"] == "7");
}

TEST_CASE("deserialization rejects malformed state", "[graph][serde]") {
  const nlohmann::json valid{
      {"steps", 1}, {"messages", nlohmann::json::array()}, {"attempts", "2"}};
  REQUIRE(graph::State<AgentSchema>::from_json(valid));

  SECTION("not an object") {
    const auto result = graph::State<AgentSchema>::from_json(nlohmann::json::array());
    REQUIRE(!result);
    REQUIRE(result.error().code == core::ErrorCode::Parse);
  }

  SECTION("missing channel") {
    nlohmann::json j = valid;
    j.erase("messages");
    const auto result = graph::State<AgentSchema>::from_json(j);
    REQUIRE(!result);
    REQUIRE(result.error().code == core::ErrorCode::Parse);
    REQUIRE(result.error().context == "messages");
  }

  SECTION("unknown channel") {
    nlohmann::json j = valid;
    j["extra"] = true;
    const auto result = graph::State<AgentSchema>::from_json(j);
    REQUIRE(!result);
    REQUIRE(result.error().code == core::ErrorCode::Parse);
    REQUIRE(result.error().context == "extra");
  }

  SECTION("wrong value type") {
    nlohmann::json j = valid;
    j["steps"] = "oops";
    const auto result = graph::State<AgentSchema>::from_json(j);
    REQUIRE(!result);
    REQUIRE(result.error().code == core::ErrorCode::Parse);
    REQUIRE(result.error().context == "steps");
  }

  SECTION("invalid JSON text") {
    const auto result = graph::State<AgentSchema>::deserialize("{nope");
    REQUIRE(!result);
    REQUIRE(result.error().code == core::ErrorCode::Parse);
  }
}
