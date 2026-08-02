#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/embedding.hpp"

namespace agents_framework::llm {

struct OpenAiEmbeddingOptions {
  std::string model{"text-embedding-3-small"};
  std::string base_url{"https://api.openai.com"};
};

class OpenAiEmbeddingBackend : public EmbeddingBackend {
 public:
  explicit OpenAiEmbeddingBackend(
      http::Secret api_key, OpenAiEmbeddingOptions options = {},
      std::shared_ptr<http::HttpClient> client = std::make_shared<http::HttpClient>());

  core::Result<EmbeddingResponse> embed(const EmbeddingRequest& request) override;
  [[nodiscard]] std::string name() const override { return "openai-embeddings"; }

  static nlohmann::json to_openai_json(const EmbeddingRequest& request,
                                       const OpenAiEmbeddingOptions& options);
  static core::Result<EmbeddingResponse> from_openai_json(const nlohmann::json& body);

 private:
  http::Secret api_key_;
  OpenAiEmbeddingOptions options_;
  std::shared_ptr<http::HttpClient> client_;
};

}
