#include "agents_framework/llm/mock_embedding.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>

namespace agents_framework::llm {

namespace {

std::uint64_t fnv1a(std::string_view text) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const char c : text) {
    hash ^= static_cast<unsigned char>(std::tolower(static_cast<unsigned char>(c)));
    hash *= 1099511628211ULL;
  }
  return hash;
}

}

std::vector<float> MockEmbeddingBackend::embed_text(std::string_view text,
                                                    std::size_t dimensions) {
  std::vector<float> vector(dimensions, 0.0F);
  std::size_t start = 0;
  const auto flush = [&](std::size_t end) {
    if (end <= start) return;
    const std::uint64_t hash = fnv1a(text.substr(start, end - start));
    const float sign = (hash >> 63) != 0 ? 1.0F : -1.0F;
    vector[hash % dimensions] += sign;
  };
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (!std::isalnum(static_cast<unsigned char>(text[i]))) {
      flush(i);
      start = i + 1;
    }
  }
  flush(text.size());

  float norm = 0.0F;
  for (const float value : vector) norm += value * value;
  if (norm > 0.0F) {
    const float scale = 1.0F / std::sqrt(norm);
    for (float& value : vector) value *= scale;
  }
  return vector;
}

core::Result<EmbeddingResponse> MockEmbeddingBackend::embed(const EmbeddingRequest& request) {
  calls_.push_back(request);
  if (handler_) {
    return handler_(request);
  }
  if (request.input.empty()) {
    return core::fail(core::ErrorCode::Invalid, "embedding request needs at least one input");
  }
  EmbeddingResponse response;
  response.embeddings.reserve(request.input.size());
  for (const std::string& text : request.input) {
    response.embeddings.push_back(embed_text(text, dimensions_));
  }
  return response;
}

}
