#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/http/http_client.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/anthropic_backend.hpp"

using namespace agents_framework;
using namespace agents_framework::llm;
using agents_framework::http::HttpClient;
using agents_framework::http::Request;
using agents_framework::http::Response;
using agents_framework::http::Secret;
using agents_framework::http::StreamChunkCallback;
using agents_framework::http::Transport;

namespace {

class ScriptedTransport : public Transport {
 public:
  core::Result<Response> next = Response{200, {}, {}};
  std::string stream_body;
  long stream_status = 200;
  std::vector<Request> calls;

  core::Result<Response> send(const Request& request, std::chrono::milliseconds) override {
    calls.push_back(request);
    return next;
  }

  core::Result<Response> send_stream(const Request& request, std::chrono::milliseconds,
                                     const StreamChunkCallback& on_chunk) override {
    calls.push_back(request);
    for (std::size_t i = 0; i < stream_body.size(); i += 7) {
      if (!on_chunk(std::string_view{stream_body}.substr(i, 7))) break;
    }
    return Response{stream_status, {}, {}};
  }
};

std::shared_ptr<HttpClient> client_with(std::shared_ptr<ScriptedTransport> transport) {
  return std::make_shared<HttpClient>(agents_framework::http::ClientOptions{}, std::move(transport));
}

constexpr std::string_view kTextResponse = R"({
  "id": "msg_01", "type": "message", "role": "assistant",
  "content": [{"type": "text", "text": "Hello there!"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 10, "output_tokens": 5}
})";

constexpr std::string_view kToolResponse = R"({
  "id": "msg_02", "role": "assistant",
  "content": [
    {"type": "text", "text": "Let me check."},
    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Waterloo"}}
  ],
  "stop_reason": "tool_use",
  "usage": {"input_tokens": 20, "output_tokens": 15}
})";

constexpr std::string_view kErrorResponse =
    R"({"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}})";

constexpr std::string_view kTextStream = R"(event: message_start
data: {"type":"message_start","message":{"id":"msg_stream","usage":{"input_tokens":8,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

)";

constexpr std::string_view kToolStream = R"(event: message_start
data: {"type":"message_start","message":{"id":"msg_t","usage":{"input_tokens":5}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"search","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"cats\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}

event: message_stop
data: {"type":"message_stop"}

)";

}

TEST_CASE("to_anthropic_json maps model, system, messages, and sampling", "[anthropic]") {
  ChatRequest request;
  request.model = "claude-haiku-4-5";
  request.system = "You are helpful.";
  request.messages = {Message::user_text("Hi")};
  request.sampling.max_tokens = 256;
  request.sampling.temperature = 0.5;

  const nlohmann::json body = AnthropicBackend::to_anthropic_json(request, AnthropicOptions{});
  REQUIRE(body["model"] == "claude-haiku-4-5");
  REQUIRE(body["max_tokens"] == 256);
  REQUIRE(body["system"] == "You are helpful.");
  REQUIRE(body["messages"][0]["role"] == "user");
  REQUIRE(body["messages"][0]["content"][0]["type"] == "text");
  REQUIRE(body["messages"][0]["content"][0]["text"] == "Hi");
  REQUIRE(body["temperature"] == 0.5);
  REQUIRE_FALSE(body.contains("stream"));
}

TEST_CASE("to_anthropic_json maps tools and tool_result blocks", "[anthropic]") {
  ChatRequest request;
  ToolDef tool;
  tool.name = "get_weather";
  tool.description = "Look up the weather";
  tool.input_schema = nlohmann::json{{"type", "object"}};
  request.tools = {tool};
  request.messages = {Message{Role::User, {ToolResultBlock{"toolu_1", "22C", false}}}};

  const nlohmann::json body = AnthropicBackend::to_anthropic_json(request, AnthropicOptions{});
  REQUIRE(body["tools"][0]["name"] == "get_weather");
  REQUIRE(body["tools"][0]["input_schema"]["type"] == "object");

  const auto& block = body["messages"][0]["content"][0];
  REQUIRE(block["type"] == "tool_result");
  REQUIRE(block["tool_use_id"] == "toolu_1");
  REQUIRE(block["content"] == "22C");
}

TEST_CASE("from_anthropic_json parses a text response", "[anthropic]") {
  const auto result = AnthropicBackend::from_anthropic_json(nlohmann::json::parse(kTextResponse));
  REQUIRE(result.has_value());
  REQUIRE(result->id == "msg_01");
  REQUIRE(result->text() == "Hello there!");
  REQUIRE(result->stop_reason == StopReason::EndTurn);
  REQUIRE(result->usage.input_tokens == 10);
  REQUIRE(result->usage.output_tokens == 5);
}

TEST_CASE("from_anthropic_json parses tool_use blocks", "[anthropic]") {
  const auto result = AnthropicBackend::from_anthropic_json(nlohmann::json::parse(kToolResponse));
  REQUIRE(result.has_value());
  REQUIRE(result->stop_reason == StopReason::ToolUse);
  const auto uses = result->tool_uses();
  REQUIRE(uses.size() == 1);
  REQUIRE(uses.front().name == "get_weather");
  REQUIRE(uses.front().input.at("city") == "Waterloo");
}

TEST_CASE("AnthropicBackend::generate posts to /v1/messages and parses the reply", "[anthropic]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->next = Response{200, {}, std::string{kTextResponse}};
  AnthropicBackend backend{Secret{"test-key"}, AnthropicOptions{}, client_with(transport)};

  ChatRequest request;
  request.messages = {Message::user_text("Hi")};
  const auto result = backend.generate(request);
  REQUIRE(result.has_value());
  REQUIRE(result->text() == "Hello there!");

  REQUIRE(transport->calls.size() == 1);
  REQUIRE(transport->calls.front().url.ends_with("/v1/messages"));
  const auto& headers = transport->calls.front().headers;
  const bool has_key = std::any_of(headers.begin(), headers.end(), [](const auto& h) {
    return h.first == "x-api-key" && h.second == "test-key";
  });
  REQUIRE(has_key);
}

TEST_CASE("AnthropicBackend::generate maps an error status to an Error", "[anthropic]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->next = Response{401, {}, std::string{kErrorResponse}};
  AnthropicBackend backend{Secret{"bad-key"}, AnthropicOptions{}, client_with(transport)};

  ChatRequest request;
  request.messages = {Message::user_text("Hi")};
  const auto result = backend.generate(request);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error().code == core::ErrorCode::Auth);
  REQUIRE(result.error().message.find("invalid x-api-key") != std::string::npos);
}

TEST_CASE("AnthropicBackend::generate_stream assembles streamed text", "[anthropic]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->stream_body = std::string{kTextStream};
  AnthropicBackend backend{Secret{"test-key"}, AnthropicOptions{}, client_with(transport)};

  std::string streamed;
  std::string start_id;
  bool stopped = false;
  const auto result = backend.generate_stream(ChatRequest{}, [&](const StreamEvent& event) {
    if (const auto* start = std::get_if<MessageStart>(&event)) start_id = start->id;
    else if (const auto* delta = std::get_if<TextDelta>(&event)) streamed += delta->text;
    else if (std::get_if<MessageStop>(&event)) stopped = true;
  });

  REQUIRE(result.has_value());
  REQUIRE(start_id == "msg_stream");
  REQUIRE(streamed == "Hello world");
  REQUIRE(result->text() == "Hello world");
  REQUIRE(result->id == "msg_stream");
  REQUIRE(result->stop_reason == StopReason::EndTurn);
  REQUIRE(result->usage.input_tokens == 8);
  REQUIRE(result->usage.output_tokens == 3);
  REQUIRE(stopped);
}

TEST_CASE("AnthropicBackend::generate_stream assembles a streamed tool call", "[anthropic]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->stream_body = std::string{kToolStream};
  AnthropicBackend backend{Secret{"test-key"}, AnthropicOptions{}, client_with(transport)};

  std::string tool_name;
  const auto result = backend.generate_stream(ChatRequest{}, [&](const StreamEvent& event) {
    if (const auto* start = std::get_if<ToolUseStart>(&event)) tool_name = start->name;
  });

  REQUIRE(result.has_value());
  REQUIRE(tool_name == "search");
  REQUIRE(result->stop_reason == StopReason::ToolUse);
  const auto uses = result->tool_uses();
  REQUIRE(uses.size() == 1);
  REQUIRE(uses.front().name == "search");
  REQUIRE(uses.front().input.at("q") == "cats");
}

TEST_CASE("AnthropicBackend live round-trip", "[anthropic][live]") {
  auto key = agents_framework::http::SecretStore::from_env("ANTHROPIC_API_KEY");
  if (!key) {
    SKIP("set ANTHROPIC_API_KEY to run live Anthropic tests");
  }
  AnthropicBackend backend{std::move(*key)};
  ChatRequest request;
  request.messages = {Message::user_text("Reply with the single word: pong")};
  request.sampling.max_tokens = 16;
  const auto result = backend.generate(request);
  REQUIRE(result.has_value());
  REQUIRE_FALSE(result->text().empty());
}
