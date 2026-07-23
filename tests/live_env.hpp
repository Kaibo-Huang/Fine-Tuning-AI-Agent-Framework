#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "agents_framework/core/dotenv.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/backend_factory.hpp"
#include "agents_framework/llm/embedding.hpp"
#include "agents_framework/llm/openai_embeddings.hpp"

namespace test_support {

inline void load_env_once() {
  static const bool loaded = [] {
    (void)agents_framework::core::load_dotenv();
    return true;
  }();
  (void)loaded;
}

struct LiveBackend {
  agents_framework::llm::BackendSelection selection;
  std::shared_ptr<agents_framework::llm::LLMBackend> backend;
};

inline std::optional<LiveBackend> live_backend(int max_tokens = 256) {
  load_env_once();
  const auto selection = agents_framework::llm::select_backend();
  if (!selection || !selection->live) return std::nullopt;

  agents_framework::llm::BackendOptions options;
  options.max_tokens = max_tokens;
  auto backend = agents_framework::llm::make_backend(*selection, options,
                                                     agents_framework::llm::system_env());
  if (!backend) return std::nullopt;
  return LiveBackend{*selection, std::move(*backend)};
}

inline std::shared_ptr<agents_framework::llm::EmbeddingBackend> live_embedder() {
  load_env_once();
  auto key = agents_framework::http::SecretStore::from_env("OPENAI_API_KEY");
  if (!key) return nullptr;
  return std::make_shared<agents_framework::llm::OpenAiEmbeddingBackend>(std::move(*key));
}

constexpr std::size_t kOpenAiEmbeddingDims = 1536;

}
