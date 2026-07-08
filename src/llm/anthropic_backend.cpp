#include "agents_framework/llm/anthropic_backend.hpp"

#include <optional>
#include <string>
#include <utility>

#include "agents_framework/http/sse.hpp"

namespace agents_framework::llm {
namespace {

core::Error anthropic_error(long status, const nlohmann::json* body) {
  std::string message = "Anthropic API error";
  if (body != nullptr && body->contains("error") && (*body)["error"].contains("message")) {
    message = (*body)["error"]["message"].get<std::string>();
  }
  return core::Error{http::error_code_for_status(status), std::move(message),
                     "status " + std::to_string(status)};
}

nlohmann::json parse_relaxed(std::string_view text) {
  return nlohmann::json::parse(text, nullptr, false);
}

}

AnthropicBackend::AnthropicBackend(http::Secret api_key, AnthropicOptions options,
                                   std::shared_ptr<http::HttpClient> client)
    : api_key_{std::move(api_key)}, options_{std::move(options)}, client_{std::move(client)} {}

nlohmann::json AnthropicBackend::to_anthropic_json(const ChatRequest& request,
                                                   const AnthropicOptions& options) {
  nlohmann::json body = nlohmann::json::object();
  body["model"] = request.model.empty() ? options.model : request.model;
  body["max_tokens"] =
      request.sampling.max_tokens > 0 ? request.sampling.max_tokens : options.max_tokens;
  if (request.system && !request.system->empty()) {
    body["system"] = *request.system;
  }

  nlohmann::json messages = nlohmann::json::array();
  for (const auto& message : request.messages) {
    nlohmann::json entry = nlohmann::json::object();
    entry["role"] = message.role == Role::Assistant ? "assistant" : "user";
    entry["content"] = message.content;
    messages.push_back(std::move(entry));
  }
  body["messages"] = std::move(messages);

  body["temperature"] = request.sampling.temperature;
  if (request.sampling.top_p) body["top_p"] = *request.sampling.top_p;
  if (request.sampling.top_k) body["top_k"] = *request.sampling.top_k;
  if (!request.sampling.stop.empty()) body["stop_sequences"] = request.sampling.stop;
  if (!request.tools.empty()) body["tools"] = request.tools;
  if (request.stream) body["stream"] = true;
  return body;
}

core::Result<ChatResponse> AnthropicBackend::from_anthropic_json(const nlohmann::json& body) {
  try {
    ChatResponse response;
    response.id = body.value("id", std::string{});
    response.role = Role::Assistant;

    if (body.contains("content") && body["content"].is_array()) {
      for (const auto& block : body["content"]) {
        const std::string type = block.value("type", std::string{});
        if (type == "text") {
          response.content.push_back(TextBlock{block.value("text", std::string{})});
        } else if (type == "tool_use") {
          response.content.push_back(ToolUseBlock{block.value("id", std::string{}),
                                                  block.value("name", std::string{}),
                                                  block.value("input", nlohmann::json::object())});
        }
      }
    }

    response.stop_reason = stop_reason_from_string(body.value("stop_reason", std::string{}));
    if (body.contains("usage")) {
      response.usage.input_tokens = body["usage"].value("input_tokens", 0);
      response.usage.output_tokens = body["usage"].value("output_tokens", 0);
    }
    return response;
  } catch (const nlohmann::json::exception& e) {
    return core::fail(core::ErrorCode::Parse, "failed to parse Anthropic response", e.what());
  }
}

http::Request AnthropicBackend::build_http_request(const ChatRequest& request, bool stream) const {
  ChatRequest prepared = request;
  prepared.stream = stream;

  http::Request http_request;
  http_request.method = "POST";
  http_request.url = options_.base_url + "/v1/messages";
  http_request.headers = {
      {"x-api-key", api_key_.reveal()},
      {"anthropic-version", options_.version},
      {"content-type", "application/json"},
  };
  http_request.body = to_anthropic_json(prepared, options_).dump();
  return http_request;
}

core::Result<ChatResponse> AnthropicBackend::generate(const ChatRequest& request) {
  AF_TRY(auto response, client_->send(build_http_request(request, false)));

  const nlohmann::json body = parse_relaxed(response.body);
  if (!response.ok()) {
    return std::unexpected(anthropic_error(response.status, body.is_discarded() ? nullptr : &body));
  }
  if (body.is_discarded()) {
    return core::fail(core::ErrorCode::Parse, "Anthropic response was not valid JSON");
  }
  return from_anthropic_json(body);
}

core::Result<ChatResponse> AnthropicBackend::generate_stream(const ChatRequest& request,
                                                             const StreamCallback& on_event) {
  struct BlockState {
    std::string type;
    TextBlock text;
    ToolUseBlock tool;
    std::string tool_json;
  };

  ChatResponse assembled;
  assembled.role = Role::Assistant;
  std::optional<BlockState> current;
  std::optional<core::Error> stream_error;

  const auto handle_event = [&](const http::SseEvent& sse) {
    const nlohmann::json data = parse_relaxed(sse.data);
    if (data.is_discarded()) return;
    const std::string type = data.value("type", std::string{});

    if (type == "message_start") {
      const auto& message = data["message"];
      assembled.id = message.value("id", std::string{});
      if (message.contains("usage")) {
        assembled.usage.input_tokens = message["usage"].value("input_tokens", 0);
      }
      if (on_event) on_event(MessageStart{assembled.id});
    } else if (type == "content_block_start") {
      const auto& block = data["content_block"];
      BlockState state;
      state.type = block.value("type", std::string{});
      if (state.type == "text") {
        state.text.text = block.value("text", std::string{});
      } else if (state.type == "tool_use") {
        state.tool.id = block.value("id", std::string{});
        state.tool.name = block.value("name", std::string{});
        if (on_event) on_event(ToolUseStart{state.tool.id, state.tool.name});
      }
      current = std::move(state);
    } else if (type == "content_block_delta") {
      const auto& delta = data["delta"];
      const std::string delta_type = delta.value("type", std::string{});
      if (delta_type == "text_delta") {
        const std::string text = delta.value("text", std::string{});
        if (current) current->text.text += text;
        if (on_event) on_event(TextDelta{text});
      } else if (delta_type == "input_json_delta") {
        const std::string partial = delta.value("partial_json", std::string{});
        if (current) current->tool_json += partial;
        if (on_event) on_event(ToolUseInputDelta{partial});
      }
    } else if (type == "content_block_stop") {
      if (current) {
        if (current->type == "text") {
          assembled.content.push_back(std::move(current->text));
        } else if (current->type == "tool_use") {
          nlohmann::json input = parse_relaxed(current->tool_json);
          current->tool.input = input.is_discarded() ? nlohmann::json::object() : std::move(input);
          assembled.content.push_back(std::move(current->tool));
        }
        current.reset();
      }
    } else if (type == "message_delta") {
      if (data.contains("delta") && data["delta"].contains("stop_reason") &&
          data["delta"]["stop_reason"].is_string()) {
        assembled.stop_reason =
            stop_reason_from_string(data["delta"]["stop_reason"].get<std::string>());
      }
      if (data.contains("usage")) {
        assembled.usage.output_tokens =
            data["usage"].value("output_tokens", assembled.usage.output_tokens);
      }
    } else if (type == "message_stop") {
      if (on_event) on_event(MessageStop{assembled.stop_reason, assembled.usage});
    } else if (type == "error") {
      std::string message = "Anthropic stream error";
      if (data.contains("error") && data["error"].contains("message")) {
        message = data["error"]["message"].get<std::string>();
      }
      stream_error = core::Error{core::ErrorCode::Protocol, std::move(message)};
      if (on_event) on_event(StreamError{*stream_error});
    }
  };

  http::SseParser parser;
  std::string raw;
  const auto on_chunk = [&](std::string_view bytes) {
    raw.append(bytes);
    parser.feed(bytes, handle_event);
    return true;
  };

  AF_TRY(auto response, client_->send_stream(build_http_request(request, true), on_chunk));
  if (!response.ok()) {
    const nlohmann::json body = parse_relaxed(raw);
    return std::unexpected(anthropic_error(response.status, body.is_discarded() ? nullptr : &body));
  }
  if (stream_error) {
    return std::unexpected(*stream_error);
  }
  return assembled;
}

}
