#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/backend.hpp"

namespace agents_framework::llm {

struct OpenAiOptions {
  std::string model{"gpt-4o-mini"};
  std::string base_url{"https://api.openai.com"};
  int max_tokens{1024};
};

class OpenAiBackend : public LLMBackend {
 public:
  explicit OpenAiBackend(
      http::Secret api_key, OpenAiOptions options = {},
      std::shared_ptr<http::HttpClient> client = std::make_shared<http::HttpClient>());

  core::Result<ChatResponse> generate(const ChatRequest& request) override;
  core::Result<ChatResponse> generate_stream(const ChatRequest& request,
                                             const StreamCallback& on_event) override;
  [[nodiscard]] BackendCapabilities capabilities() const override { return {true, true}; }
  [[nodiscard]] std::string name() const override { return "openai"; }

  static nlohmann::json to_openai_json(const ChatRequest& request, const OpenAiOptions& options);
  static core::Result<ChatResponse> from_openai_json(const nlohmann::json& body);

 private:
  [[nodiscard]] http::Request build_http_request(const ChatRequest& request, bool stream) const;

  http::Secret api_key_;
  OpenAiOptions options_;
  std::shared_ptr<http::HttpClient> client_;
};

}
