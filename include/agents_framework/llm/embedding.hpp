#pragma once

#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/response.hpp"

namespace agents_framework::llm {

struct EmbeddingRequest {
  std::string model;
  std::vector<std::string> input;
  bool operator==(const EmbeddingRequest&) const = default;
};

struct EmbeddingResponse {
  std::vector<std::vector<float>> embeddings;
  Usage usage;
  bool operator==(const EmbeddingResponse&) const = default;
};

class EmbeddingBackend {
 public:
  virtual ~EmbeddingBackend() = default;

  virtual core::Result<EmbeddingResponse> embed(const EmbeddingRequest& request) = 0;

  [[nodiscard]] virtual std::string name() const = 0;
};

}
