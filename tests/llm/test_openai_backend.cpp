#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "../live_env.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/openai_backend.hpp"

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
  "id": "chatcmpl-1", "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5}
})";

constexpr std::string_view kToolResponse = R"({
  "id": "chatcmpl-2",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": null,
    "tool_calls": [{"id": "call_1", "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\":\"Waterloo\"}"}}]},
    "finish_reason": "tool_calls"}],
  "usage": {"prompt_tokens": 20, "completion_tokens": 15}
})";

constexpr std::string_view kErrorResponse =
    R"({"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}})";

constexpr std::string_view kTextStream = R"(data: {"id":"chatcmpl-s","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-s","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-s","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-s","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-s","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2}}

data: [DONE]

)";

constexpr std::string_view kToolStream = R"(data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"cats\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-t","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

)";

}

TEST_CASE("to_openai_json maps system, messages, and sampling", "[openai]") {
  ChatRequest request;
  request.system = "Be brief.";
  request.messages = {Message::user_text("Hi")};
  request.sampling.max_tokens = 128;
  request.sampling.temperature = 0.3;

  const nlohmann::json body = OpenAiBackend::to_openai_json(request, OpenAiOptions{});
  REQUIRE(body["model"] == "gpt-4o-mini");
  REQUIRE(body["messages"][0]["role"] == "system");
  REQUIRE(body["messages"][0]["content"] == "Be brief.");
  REQUIRE(body["messages"][1]["role"] == "user");
  REQUIRE(body["messages"][1]["content"] == "Hi");
  REQUIRE(body["max_tokens"] == 128);
  REQUIRE(body["temperature"] == 0.3);
}

TEST_CASE("to_openai_json expands tool_use and tool_result across messages", "[openai]") {
  ChatRequest request;
  request.messages = {
      Message{Role::Assistant, {ToolUseBlock{"call_1", "search", nlohmann::json{{"q", "cats"}}}}},
      Message{Role::User, {ToolResultBlock{"call_1", "found cats", false}}},
  };

  const nlohmann::json body = OpenAiBackend::to_openai_json(request, OpenAiOptions{});
  const auto& messages = body["messages"];
  REQUIRE(messages.size() == 2);
  REQUIRE(messages[0]["role"] == "assistant");
  REQUIRE(messages[0]["content"].is_null());
  REQUIRE(messages[0]["tool_calls"][0]["id"] == "call_1");
  REQUIRE(messages[0]["tool_calls"][0]["function"]["name"] == "search");
  REQUIRE(messages[0]["tool_calls"][0]["function"]["arguments"] == R"({"q":"cats"})");
  REQUIRE(messages[1]["role"] == "tool");
  REQUIRE(messages[1]["tool_call_id"] == "call_1");
  REQUIRE(messages[1]["content"] == "found cats");
}

TEST_CASE("to_openai_json maps tools to function definitions", "[openai]") {
  ChatRequest request;
  ToolDef tool;
  tool.name = "search";
  tool.description = "Search the web";
  tool.input_schema = nlohmann::json{{"type", "object"}};
  request.tools = {tool};

  const nlohmann::json body = OpenAiBackend::to_openai_json(request, OpenAiOptions{});
  REQUIRE(body["tools"][0]["type"] == "function");
  REQUIRE(body["tools"][0]["function"]["name"] == "search");
  REQUIRE(body["tools"][0]["function"]["parameters"]["type"] == "object");
}

TEST_CASE("from_openai_json parses a text response", "[openai]") {
  const auto result = OpenAiBackend::from_openai_json(nlohmann::json::parse(kTextResponse));
  REQUIRE(result.has_value());
  REQUIRE(result->id == "chatcmpl-1");
  REQUIRE(result->text() == "Hello there!");
  REQUIRE(result->stop_reason == StopReason::EndTurn);
  REQUIRE(result->usage.input_tokens == 10);
  REQUIRE(result->usage.output_tokens == 5);
}

TEST_CASE("from_openai_json parses tool_calls", "[openai]") {
  const auto result = OpenAiBackend::from_openai_json(nlohmann::json::parse(kToolResponse));
  REQUIRE(result.has_value());
  REQUIRE(result->stop_reason == StopReason::ToolUse);
  const auto uses = result->tool_uses();
  REQUIRE(uses.size() == 1);
  REQUIRE(uses.front().id == "call_1");
  REQUIRE(uses.front().name == "get_weather");
  REQUIRE(uses.front().input.at("city") == "Waterloo");
}

TEST_CASE("OpenAiBackend::generate posts to chat/completions with a Bearer token", "[openai]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->next = Response{200, {}, std::string{kTextResponse}};
  OpenAiBackend backend{Secret{"sk-test"}, OpenAiOptions{}, client_with(transport)};

  ChatRequest request;
  request.messages = {Message::user_text("Hi")};
  const auto result = backend.generate(request);
  REQUIRE(result.has_value());
  REQUIRE(result->text() == "Hello there!");

  REQUIRE(transport->calls.size() == 1);
  REQUIRE(transport->calls.front().url.ends_with("/v1/chat/completions"));
  const auto& headers = transport->calls.front().headers;
  const bool has_bearer = std::any_of(headers.begin(), headers.end(), [](const auto& h) {
    return h.first == "authorization" && h.second == "Bearer sk-test";
  });
  REQUIRE(has_bearer);
}

TEST_CASE("OpenAiBackend::generate maps an error status to an Error", "[openai]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->next = Response{401, {}, std::string{kErrorResponse}};
  OpenAiBackend backend{Secret{"sk-bad"}, OpenAiOptions{}, client_with(transport)};

  ChatRequest request;
  request.messages = {Message::user_text("Hi")};
  const auto result = backend.generate(request);
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error().code == core::ErrorCode::Auth);
  REQUIRE(result.error().message.find("Incorrect API key") != std::string::npos);
}

TEST_CASE("OpenAiBackend::generate_stream assembles streamed text", "[openai]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->stream_body = std::string{kTextStream};
  OpenAiBackend backend{Secret{"sk-test"}, OpenAiOptions{}, client_with(transport)};

  std::string streamed;
  std::string start_id;
  const auto result = backend.generate_stream(ChatRequest{}, [&](const StreamEvent& event) {
    if (const auto* start = std::get_if<MessageStart>(&event)) start_id = start->id;
    else if (const auto* delta = std::get_if<TextDelta>(&event)) streamed += delta->text;
  });

  REQUIRE(result.has_value());
  REQUIRE(start_id == "chatcmpl-s");
  REQUIRE(streamed == "Hello world");
  REQUIRE(result->text() == "Hello world");
  REQUIRE(result->stop_reason == StopReason::EndTurn);
  REQUIRE(result->usage.input_tokens == 8);
  REQUIRE(result->usage.output_tokens == 2);
}

TEST_CASE("OpenAiBackend::generate_stream assembles a streamed tool call", "[openai]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->stream_body = std::string{kToolStream};
  OpenAiBackend backend{Secret{"sk-test"}, OpenAiOptions{}, client_with(transport)};

  std::string tool_name;
  const auto result = backend.generate_stream(ChatRequest{}, [&](const StreamEvent& event) {
    if (const auto* start = std::get_if<ToolUseStart>(&event)) tool_name = start->name;
  });

  REQUIRE(result.has_value());
  REQUIRE(tool_name == "search");
  REQUIRE(result->stop_reason == StopReason::ToolUse);
  const auto uses = result->tool_uses();
  REQUIRE(uses.size() == 1);
  REQUIRE(uses.front().id == "call_9");
  REQUIRE(uses.front().name == "search");
  REQUIRE(uses.front().input.at("q") == "cats");
}

TEST_CASE("OpenAiBackend live round-trip", "[.live][openai]") {
  test_support::load_env_once();
  auto key = agents_framework::http::SecretStore::from_env("OPENAI_API_KEY");
  if (!key) {
    SKIP("set OPENAI_API_KEY to run live OpenAI tests");
  }
  OpenAiBackend backend{std::move(*key)};
  ChatRequest request;
  request.messages = {Message::user_text("Reply with the single word: pong")};
  request.sampling.max_tokens = 16;
  const auto result = backend.generate(request);
  REQUIRE(result.has_value());

  REQUIRE_FALSE(result->text().empty());
  CHECK(result->role == Role::Assistant);
  CHECK(result->stop_reason == StopReason::EndTurn);
  CHECK(result->id.starts_with("chatcmpl"));
  CHECK(result->usage.input_tokens > 0);
  CHECK(result->usage.output_tokens > 0);
}
