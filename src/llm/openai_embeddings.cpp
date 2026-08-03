#include "agents_framework/llm/openai_embeddings.hpp"

#include <utility>

namespace agents_framework::llm {

namespace {

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

}

OpenAiEmbeddingBackend::OpenAiEmbeddingBackend(http::Secret api_key,
                                               OpenAiEmbeddingOptions options,
                                               std::shared_ptr<http::HttpClient> client)
    : api_key_{std::move(api_key)}, options_{std::move(options)}, client_{std::move(client)} {}

nlohmann::json OpenAiEmbeddingBackend::to_openai_json(const EmbeddingRequest& request,
                                                      const OpenAiEmbeddingOptions& options) {
  nlohmann::json body = nlohmann::json::object();
  body["model"] = request.model.empty() ? options.model : request.model;
  body["input"] = request.input;
  return body;
}

core::Result<EmbeddingResponse> OpenAiEmbeddingBackend::from_openai_json(
    const nlohmann::json& body) {
  try {
    if (!body.contains("data") || !body["data"].is_array()) {
      return core::fail(core::ErrorCode::Parse, "embeddings response has no data array");
    }
    EmbeddingResponse response;
    response.embeddings.resize(body["data"].size());
    for (const auto& entry : body["data"]) {
      const auto index = entry.value("index", 0);
      if (index < 0 || static_cast<std::size_t>(index) >= response.embeddings.size()) {
        return core::fail(core::ErrorCode::Parse, "embeddings response index out of range",
                          std::to_string(index));
      }
      response.embeddings[static_cast<std::size_t>(index)] =
          entry.at("embedding").get<std::vector<float>>();
    }
    if (body.contains("usage") && body["usage"].is_object()) {
      response.usage.input_tokens = body["usage"].value("prompt_tokens", 0);
    }
    return response;
  } catch (const nlohmann::json::exception& e) {
    return core::fail(core::ErrorCode::Parse, "failed to parse embeddings response", e.what());
  }
}

core::Result<EmbeddingResponse> OpenAiEmbeddingBackend::embed(const EmbeddingRequest& request) {
  if (request.input.empty()) {
    return core::fail(core::ErrorCode::Invalid, "embedding request needs at least one input");
  }

  http::Request http_request;
  http_request.method = "POST";
  http_request.url = options_.base_url + "/v1/embeddings";
  http_request.headers = {
      {"authorization", "Bearer " + api_key_.reveal()},
      {"content-type", "application/json"},
  };
  http_request.body = to_openai_json(request, options_).dump();

  AF_TRY(auto response, client_->send(http_request));
  const nlohmann::json body = nlohmann::json::parse(response.body, nullptr, false);
  if (!response.ok()) {
    return std::unexpected(openai_error(response.status, body.is_discarded() ? nullptr : &body));
  }
  if (body.is_discarded()) {
    return core::fail(core::ErrorCode::Parse, "embeddings response was not valid JSON");
  }
  AF_TRY(auto parsed, from_openai_json(body));
  if (parsed.embeddings.size() != request.input.size()) {
    return core::fail(core::ErrorCode::Protocol,
                      "embeddings response count does not match input count",
                      std::to_string(parsed.embeddings.size()) + " != " +
                          std::to_string(request.input.size()));
  }
  return parsed;
}

}
