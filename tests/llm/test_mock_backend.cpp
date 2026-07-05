#include <catch2/catch_test_macros.hpp>

#include <string>
#include <variant>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/mock_backend.hpp"

using namespace agents_framework;
using namespace agents_framework::llm;

TEST_CASE("MockBackend records calls and defaults its identity", "[mock]") {
  MockBackend backend;
  REQUIRE(backend.name() == "mock");
  REQUIRE(backend.capabilities().native_tools);
  REQUIRE(backend.capabilities().streaming);

  backend.push_response(ChatResponse{});
  ChatRequest request;
  request.model = "model-a";
  REQUIRE(backend.generate(request).has_value());
  REQUIRE(backend.calls().size() == 1);
  REQUIRE(backend.calls().front().model == "model-a");
}

TEST_CASE("MockBackend drives a scripted tool round-trip", "[mock]") {
  MockBackend backend;

  ChatResponse turn1;
  turn1.id = "msg_1";
  turn1.stop_reason = StopReason::ToolUse;
  turn1.content = {ToolUseBlock{"tu_1", "add", nlohmann::json{{"a", 2}, {"b", 3}}}};

  ChatResponse turn2;
  turn2.id = "msg_2";
  turn2.stop_reason = StopReason::EndTurn;
  turn2.content = {TextBlock{"The answer is 5."}};

  backend.push_response(turn1);
  backend.push_response(turn2);

  ChatRequest request;
  request.model = "mock-model";
  request.messages = {Message::user_text("what is 2+3 via the add tool?")};

  const auto first = backend.generate(request);
  REQUIRE(first.has_value());
  REQUIRE(first->stop_reason == StopReason::ToolUse);
  const auto uses = first->tool_uses();
  REQUIRE(uses.size() == 1);
  REQUIRE(uses.front().name == "add");

  // Caller echoes the assistant turn and feeds the tool result back.
  request.messages.push_back(Message{Role::Assistant, first->content});
  request.messages.push_back(
      Message{Role::User, {ToolResultBlock{uses.front().id, "5", false}}});

  const auto second = backend.generate(request);
  REQUIRE(second.has_value());
  REQUIRE(second->stop_reason == StopReason::EndTurn);
  REQUIRE(second->text() == "The answer is 5.");

  REQUIRE(backend.calls().size() == 2);
  REQUIRE(backend.calls().back().messages.size() == 3);
}

TEST_CASE("MockBackend streams events that assemble into the final response", "[mock]") {
  MockBackend backend;
  ChatResponse response;
  response.id = "msg_s";
  response.stop_reason = StopReason::EndTurn;
  response.usage = Usage{3, 7};
  response.content = {TextBlock{"streaming works"}};
  backend.push_response(response);

  std::string started_id;
  std::string assembled;
  bool stopped = false;
  StopReason stop_reason = StopReason::Error;

  const auto result = backend.generate_stream(ChatRequest{}, [&](const StreamEvent& event) {
    if (const auto* start = std::get_if<MessageStart>(&event)) {
      started_id = start->id;
    } else if (const auto* delta = std::get_if<TextDelta>(&event)) {
      assembled += delta->text;
    } else if (const auto* stop = std::get_if<MessageStop>(&event)) {
      stopped = true;
      stop_reason = stop->reason;
    }
  });

  REQUIRE(result.has_value());
  REQUIRE(started_id == "msg_s");
  REQUIRE(assembled == "streaming works");
  REQUIRE(assembled == result->text());  // streamed assembly equals non-streaming text
  REQUIRE(stopped);
  REQUIRE(stop_reason == StopReason::EndTurn);
}

TEST_CASE("MockBackend streams tool_use as a start plus input delta", "[mock]") {
  MockBackend backend;
  ChatResponse response;
  response.stop_reason = StopReason::ToolUse;
  response.content = {ToolUseBlock{"tu_1", "search", nlohmann::json{{"q", "x"}}}};
  backend.push_response(response);

  std::string tool_name;
  std::string input_json;
  const auto result = backend.generate_stream(ChatRequest{}, [&](const StreamEvent& event) {
    if (const auto* start = std::get_if<ToolUseStart>(&event)) {
      tool_name = start->name;
    } else if (const auto* delta = std::get_if<ToolUseInputDelta>(&event)) {
      input_json += delta->partial_json;
    }
  });
  REQUIRE(result.has_value());

  REQUIRE(tool_name == "search");
  REQUIRE(nlohmann::json::parse(input_json) == nlohmann::json{{"q", "x"}});
}

TEST_CASE("MockBackend handler mode computes responses from the request", "[mock]") {
  MockBackend backend;
  backend.set_handler([](const ChatRequest& request) -> core::Result<ChatResponse> {
    ChatResponse response;
    response.content = {TextBlock{"model=" + request.model}};
    return response;
  });

  ChatRequest request;
  request.model = "haiku";
  const auto result = backend.generate(request);
  REQUIRE(result.has_value());
  REQUIRE(result->text() == "model=haiku");
}

TEST_CASE("MockBackend returns scripted errors and fails when empty", "[mock]") {
  MockBackend backend;
  backend.push_error(core::Error{core::ErrorCode::RateLimited, "slow down"});

  const auto err = backend.generate(ChatRequest{});
  REQUIRE_FALSE(err.has_value());
  REQUIRE(err.error().code == core::ErrorCode::RateLimited);

  const auto empty = backend.generate(ChatRequest{});
  REQUIRE_FALSE(empty.has_value());
  REQUIRE(empty.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("MockBackend surfaces stream errors via StreamError", "[mock]") {
  MockBackend backend;
  backend.push_error(core::Error{core::ErrorCode::Network, "boom"});

  bool saw_error = false;
  const auto result = backend.generate_stream(ChatRequest{}, [&](const StreamEvent& event) {
    if (std::get_if<StreamError>(&event)) saw_error = true;
  });

  REQUIRE_FALSE(result.has_value());
  REQUIRE(saw_error);
}
