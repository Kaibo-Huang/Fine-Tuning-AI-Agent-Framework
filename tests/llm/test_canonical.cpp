#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/llm/message.hpp"
#include "agents_framework/llm/response.hpp"
#include "agents_framework/llm/tool.hpp"

using namespace agents_framework::llm;

TEST_CASE("Message helpers build single-text-block messages", "[canonical]") {
  const Message message = Message::user_text("hello");
  REQUIRE(message.role == Role::User);
  REQUIRE(message.content.size() == 1);
  REQUIRE(std::get<TextBlock>(message.content.front()).text == "hello");
}

TEST_CASE("A Message with mixed content round-trips through JSON", "[canonical]") {
  Message message;
  message.role = Role::Assistant;
  message.content = {
      TextBlock{"let me check"},
      ToolUseBlock{"tu_1", "get_weather", nlohmann::json{{"city", "Waterloo"}}},
  };

  const nlohmann::json encoded = message;
  const Message decoded = encoded.get<Message>();
  REQUIRE(decoded == message);
}

TEST_CASE("A tool_result block round-trips through JSON", "[canonical]") {
  const ContentBlock block = ToolResultBlock{"tu_1", "22C and sunny", false};
  const nlohmann::json encoded = block;
  REQUIRE(encoded["type"] == "tool_result");
  const ContentBlock decoded = encoded.get<ContentBlock>();
  REQUIRE(decoded == block);
}

TEST_CASE("ToolDef round-trips through JSON", "[canonical]") {
  ToolDef tool;
  tool.name = "get_weather";
  tool.description = "Look up the weather";
  tool.input_schema = nlohmann::json{{"type", "object"},
                                     {"properties", {{"city", {{"type", "string"}}}}}};

  const nlohmann::json encoded = tool;
  const ToolDef decoded = encoded.get<ToolDef>();
  REQUIRE(decoded == tool);
}

TEST_CASE("ChatResponse round-trips and exposes text/tool_uses", "[canonical]") {
  ChatResponse response;
  response.id = "msg_123";
  response.stop_reason = StopReason::ToolUse;
  response.usage = Usage{12, 34};
  response.content = {
      TextBlock{"Thinking. "},
      TextBlock{"Calling a tool."},
      ToolUseBlock{"tu_9", "search", nlohmann::json{{"q", "cats"}}},
  };

  SECTION("accessors aggregate the right blocks") {
    REQUIRE(response.text() == "Thinking. Calling a tool.");
    const auto uses = response.tool_uses();
    REQUIRE(uses.size() == 1);
    REQUIRE(uses.front().name == "search");
    REQUIRE(uses.front().input.at("q") == "cats");
  }

  SECTION("JSON round-trip preserves the response") {
    const nlohmann::json encoded = response;
    REQUIRE(encoded["stop_reason"] == "tool_use");
    REQUIRE(encoded["usage"]["input_tokens"] == 12);
    const ChatResponse decoded = encoded.get<ChatResponse>();
    REQUIRE(decoded == response);
  }
}
