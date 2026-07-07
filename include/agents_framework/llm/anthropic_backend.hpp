#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/backend.hpp"

namespace agents_framework::llm {

struct AnthropicOptions {
  std::string model{"claude-haiku-4-5"};
  std::string base_url{"https://api.anthropic.com"};
  std::string version{"2023-06-01"};
  int max_tokens{1024};
};

class AnthropicBackend : public LLMBackend {
 public:
  explicit AnthropicBackend(
      http::Secret api_key, AnthropicOptions options = {},
      std::shared_ptr<http::HttpClient> client = std::make_shared<http::HttpClient>());

  core::Result<ChatResponse> generate(const ChatRequest& request) override;
  core::Result<ChatResponse> generate_stream(const ChatRequest& request,
                                             const StreamCallback& on_event) override;
  [[nodiscard]] BackendCapabilities capabilities() const override { return {true, true}; }
  [[nodiscard]] std::string name() const override { return "anthropic"; }

  static nlohmann::json to_anthropic_json(const ChatRequest& request,
                                          const AnthropicOptions& options);
  static core::Result<ChatResponse> from_anthropic_json(const nlohmann::json& body);

 private:
  [[nodiscard]] http::Request build_http_request(const ChatRequest& request, bool stream) const;

  http::Secret api_key_;
  AnthropicOptions options_;
  std::shared_ptr<http::HttpClient> client_;
};

}
