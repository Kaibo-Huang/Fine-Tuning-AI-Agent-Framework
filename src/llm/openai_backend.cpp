#include "agents_framework/llm/openai_backend.hpp"

#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/http/sse.hpp"

namespace agents_framework::llm {
namespace {

StopReason openai_stop_reason(std::string_view reason) {
  if (reason == "length") return StopReason::MaxTokens;
  if (reason == "tool_calls") return StopReason::ToolUse;
  if (reason == "content_filter") return StopReason::Error;
  return StopReason::EndTurn;
}

core::Error openai_error(long status, const nlohmann::json* body) {
  std::string message = "OpenAI API error";
  if (body != nullptr && body->contains("error")) {
    const auto& error = (*body)["error"];
    if (error.is_object() && error.contains("message") && error["message"].is_string()) {
      message = error["message"].get<std::string>();
    } else if (error.is_string()) {
      message = error.get<std::string>();
    }
  }
  return core::Error{http::error_code_for_status(status), std::move(message),
                     "status " + std::to_string(status)};
}

nlohmann::json parse_relaxed(std::string_view text) {
  return nlohmann::json::parse(text, nullptr, false);
}

void append_messages(nlohmann::json& messages, const Message& message) {
  std::string text;
  nlohmann::json tool_calls = nlohmann::json::array();

  for (const auto& block : message.content) {
    if (const auto* piece = std::get_if<TextBlock>(&block)) {
      text += piece->text;
    } else if (const auto* use = std::get_if<ToolUseBlock>(&block)) {
      nlohmann::json function = nlohmann::json::object();
      function["name"] = use->name;
      function["arguments"] = use->input.dump();
      nlohmann::json call = nlohmann::json::object();
      call["id"] = use->id;
      call["type"] = "function";
      call["function"] = std::move(function);
      tool_calls.push_back(std::move(call));
    } else if (const auto* result = std::get_if<ToolResultBlock>(&block)) {
      nlohmann::json entry = nlohmann::json::object();
      entry["role"] = "tool";
      entry["tool_call_id"] = result->tool_use_id;
      entry["content"] = result->content;
      messages.push_back(std::move(entry));
    }
  }

  if (text.empty() && tool_calls.empty()) {
    return;
  }

  const char* role = message.role == Role::Assistant ? "assistant"
                     : message.role == Role::System  ? "system"
                                                     : "user";
  nlohmann::json entry = nlohmann::json::object();
  entry["role"] = role;
  if (!tool_calls.empty()) {
    entry["content"] = text.empty() ? nlohmann::json(nullptr) : nlohmann::json(text);
    entry["tool_calls"] = std::move(tool_calls);
  } else {
    entry["content"] = text;
  }
  messages.push_back(std::move(entry));
}

}

OpenAiBackend::OpenAiBackend(http::Secret api_key, OpenAiOptions options,
                             std::shared_ptr<http::HttpClient> client)
    : api_key_{std::move(api_key)}, options_{std::move(options)}, client_{std::move(client)} {}

nlohmann::json OpenAiBackend::to_openai_json(const ChatRequest& request,
                                             const OpenAiOptions& options) {
  nlohmann::json body = nlohmann::json::object();
  body["model"] = request.model.empty() ? options.model : request.model;

  nlohmann::json messages = nlohmann::json::array();
  if (request.system && !request.system->empty()) {
    nlohmann::json system = nlohmann::json::object();
    system["role"] = "system";
    system["content"] = *request.system;
    messages.push_back(std::move(system));
  }
  for (const auto& message : request.messages) {
    append_messages(messages, message);
  }
  body["messages"] = std::move(messages);

  body["max_tokens"] =
      request.sampling.max_tokens > 0 ? request.sampling.max_tokens : options.max_tokens;
  body["temperature"] = request.sampling.temperature;
  if (request.sampling.top_p) body["top_p"] = *request.sampling.top_p;
  if (!request.sampling.stop.empty()) body["stop"] = request.sampling.stop;
  if (request.sampling.seed) body["seed"] = *request.sampling.seed;

  if (!request.tools.empty()) {
    nlohmann::json tools = nlohmann::json::array();
    for (const auto& tool : request.tools) {
      nlohmann::json function = nlohmann::json::object();
      function["name"] = tool.name;
      function["description"] = tool.description;
      function["parameters"] = tool.input_schema;
      nlohmann::json entry = nlohmann::json::object();
      entry["type"] = "function";
      entry["function"] = std::move(function);
      tools.push_back(std::move(entry));
    }
    body["tools"] = std::move(tools);
  }

  if (request.stream) {
    body["stream"] = true;
    nlohmann::json stream_options = nlohmann::json::object();
    stream_options["include_usage"] = true;
    body["stream_options"] = std::move(stream_options);
  }
  return body;
}

core::Result<ChatResponse> OpenAiBackend::from_openai_json(const nlohmann::json& body) {
  try {
    ChatResponse response;
    response.id = body.value("id", std::string{});
    response.role = Role::Assistant;

    if (body.contains("choices") && body["choices"].is_array() && !body["choices"].empty()) {
      const auto& choice = body["choices"][0];
      const auto message = choice.value("message", nlohmann::json::object());

      if (message.contains("content") && message["content"].is_string()) {
        std::string text = message["content"].get<std::string>();
        if (!text.empty()) response.content.push_back(TextBlock{std::move(text)});
      }
      if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
        for (const auto& call : message["tool_calls"]) {
          const auto function = call.value("function", nlohmann::json::object());
          ToolUseBlock use;
          use.id = call.value("id", std::string{});
          use.name = function.value("name", std::string{});
          const std::string arguments = function.value("arguments", std::string{});
          nlohmann::json input = arguments.empty() ? nlohmann::json::object() : parse_relaxed(arguments);
          use.input = input.is_discarded() ? nlohmann::json::object() : std::move(input);
          response.content.push_back(std::move(use));
        }
      }
      response.stop_reason = openai_stop_reason(choice.value("finish_reason", std::string{}));
    }

    if (body.contains("usage") && body["usage"].is_object()) {
      response.usage.input_tokens = body["usage"].value("prompt_tokens", 0);
      response.usage.output_tokens = body["usage"].value("completion_tokens", 0);
    }
    return response;
  } catch (const nlohmann::json::exception& e) {
    return core::fail(core::ErrorCode::Parse, "failed to parse OpenAI response", e.what());
  }
}

http::Request OpenAiBackend::build_http_request(const ChatRequest& request, bool stream) const {
  ChatRequest prepared = request;
  prepared.stream = stream;

  http::Request http_request;
  http_request.method = "POST";
  http_request.url = options_.base_url + "/v1/chat/completions";
  http_request.headers = {
      {"authorization", "Bearer " + api_key_.reveal()},
      {"content-type", "application/json"},
  };
  http_request.body = to_openai_json(prepared, options_).dump();
  return http_request;
}

core::Result<ChatResponse> OpenAiBackend::generate(const ChatRequest& request) {
  AF_TRY(auto response, client_->send(build_http_request(request, false)));

  const nlohmann::json body = parse_relaxed(response.body);
  if (!response.ok()) {
    return std::unexpected(openai_error(response.status, body.is_discarded() ? nullptr : &body));
  }
  if (body.is_discarded()) {
    return core::fail(core::ErrorCode::Parse, "OpenAI response was not valid JSON");
  }
  return from_openai_json(body);
}

core::Result<ChatResponse> OpenAiBackend::generate_stream(const ChatRequest& request,
                                                          const StreamCallback& on_event) {
  struct ToolAccum {
    std::string id;
    std::string name;
    std::string arguments;
  };

  ChatResponse assembled;
  assembled.role = Role::Assistant;
  std::string text;
  std::map<int, ToolAccum> tools;
  bool started = false;

  const auto handle_event = [&](const http::SseEvent& sse) {
    if (sse.data == "[DONE]") return;
    const nlohmann::json data = parse_relaxed(sse.data);
    if (data.is_discarded()) return;

    if (!started) {
      assembled.id = data.value("id", std::string{});
      if (on_event) on_event(MessageStart{assembled.id});
      started = true;
    }
    if (data.contains("usage") && data["usage"].is_object()) {
      assembled.usage.input_tokens =
          data["usage"].value("prompt_tokens", assembled.usage.input_tokens);
      assembled.usage.output_tokens =
          data["usage"].value("completion_tokens", assembled.usage.output_tokens);
    }
    if (!data.contains("choices") || !data["choices"].is_array() || data["choices"].empty()) {
      return;
    }
    const auto& choice = data["choices"][0];
    const auto delta = choice.value("delta", nlohmann::json::object());

    if (delta.contains("content") && delta["content"].is_string()) {
      const std::string piece = delta["content"].get<std::string>();
      if (!piece.empty()) {
        text += piece;
        if (on_event) on_event(TextDelta{piece});
      }
    }
    if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
      for (const auto& call : delta["tool_calls"]) {
        ToolAccum& accum = tools[call.value("index", 0)];
        if (call.contains("id") && call["id"].is_string()) accum.id = call["id"].get<std::string>();
        if (call.contains("function") && call["function"].is_object()) {
          const auto& function = call["function"];
          if (function.contains("name") && function["name"].is_string()) {
            accum.name = function["name"].get<std::string>();
            if (on_event) on_event(ToolUseStart{accum.id, accum.name});
          }
          if (function.contains("arguments") && function["arguments"].is_string()) {
            const std::string fragment = function["arguments"].get<std::string>();
            accum.arguments += fragment;
            if (!fragment.empty() && on_event) on_event(ToolUseInputDelta{fragment});
          }
        }
      }
    }
    if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
      assembled.stop_reason = openai_stop_reason(choice["finish_reason"].get<std::string>());
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
    return std::unexpected(openai_error(response.status, body.is_discarded() ? nullptr : &body));
  }

  if (!text.empty()) {
    assembled.content.push_back(TextBlock{std::move(text)});
  }
  for (auto& [index, accum] : tools) {
    ToolUseBlock use;
    use.id = std::move(accum.id);
    use.name = std::move(accum.name);
    nlohmann::json input =
        accum.arguments.empty() ? nlohmann::json::object() : parse_relaxed(accum.arguments);
    use.input = input.is_discarded() ? nlohmann::json::object() : std::move(input);
    assembled.content.push_back(std::move(use));
  }
  if (on_event) on_event(MessageStop{assembled.stop_reason, assembled.usage});
  return assembled;
}

}
