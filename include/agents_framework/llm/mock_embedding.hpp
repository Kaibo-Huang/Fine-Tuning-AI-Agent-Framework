#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/embedding.hpp"

namespace agents_framework::llm {

class MockEmbeddingBackend : public EmbeddingBackend {
 public:
  using Handler = std::function<core::Result<EmbeddingResponse>(const EmbeddingRequest&)>;

  explicit MockEmbeddingBackend(std::size_t dimensions = 64) : dimensions_(dimensions) {}

  void set_handler(Handler handler) { handler_ = std::move(handler); }

  core::Result<EmbeddingResponse> embed(const EmbeddingRequest& request) override;
  [[nodiscard]] std::string name() const override { return "mock-embeddings"; }

  [[nodiscard]] std::size_t dimensions() const noexcept { return dimensions_; }
  [[nodiscard]] const std::vector<EmbeddingRequest>& calls() const noexcept { return calls_; }

  static std::vector<float> embed_text(std::string_view text, std::size_t dimensions);

 private:
  std::size_t dimensions_;
  Handler handler_;
  std::vector<EmbeddingRequest> calls_;
};

}
