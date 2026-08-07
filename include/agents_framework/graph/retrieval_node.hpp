#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/fixed_string.hpp"
#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/llm/embedding.hpp"
#include "agents_framework/store/vector_store.hpp"

namespace agents_framework::graph {

struct Document {
  std::string id;
  std::string content;
  double score{0.0};
  nlohmann::json metadata = nlohmann::json::object();

  bool operator==(const Document&) const = default;
};

inline void to_json(nlohmann::json& j, const Document& document) {
  j = nlohmann::json::object();
  j["id"] = document.id;
  j["content"] = document.content;
  j["score"] = document.score;
  j["metadata"] = document.metadata;
}

inline void from_json(const nlohmann::json& j, Document& document) {
  document.id = j.at("id").get<std::string>();
  document.content = j.value("content", std::string{});
  document.score = j.value("score", 0.0);
  document.metadata = j.value("metadata", nlohmann::json::object());
}

[[nodiscard]] std::string render_documents(std::span<const Document> documents);

struct RetrievalOptions {
  std::string model;
  std::size_t k{4};
  std::optional<double> min_score;
};

template <class S, core::FixedString QueryChannel = "query",
          core::FixedString DocsChannel = "documents">
class RetrievalNode final : public Node {
  static_assert(std::same_as<typename S::template value_of<QueryChannel>, std::string>,
                "RetrievalNode requires a std::string query channel");
  static_assert(std::same_as<typename S::template value_of<DocsChannel>, std::vector<Document>>,
                "RetrievalNode requires a std::vector<Document> documents channel");

 public:
  RetrievalNode(std::shared_ptr<llm::EmbeddingBackend> backend,
                std::shared_ptr<store::VectorStore> vectors, RetrievalOptions options = {})
      : backend_(std::move(backend)), vectors_(std::move(vectors)),
        options_(std::move(options)) {
    if (!backend_) throw std::invalid_argument("RetrievalNode requires a non-null backend");
    if (!vectors_) throw std::invalid_argument("RetrievalNode requires a non-null vector store");
  }

  core::Result<StateUpdate> run(const ChannelMap& state) override {
    StateView<S> view(state);
    const std::string& query = view.template get<QueryChannel>();
    if (query.empty()) {
      return core::fail(core::ErrorCode::Invalid, "retrieval node ran with an empty query");
    }

    AF_TRY(auto embedded, backend_->embed({options_.model, {query}}));
    if (embedded.embeddings.size() != 1) {
      return core::fail(core::ErrorCode::Protocol,
                        "embedding backend returned an unexpected embedding count",
                        std::to_string(embedded.embeddings.size()));
    }

    AF_TRY(auto matches, vectors_->query(embedded.embeddings.front(), options_.k));
    std::vector<Document> documents;
    documents.reserve(matches.size());
    for (auto& match : matches) {
      if (options_.min_score && match.score < *options_.min_score) continue;
      documents.push_back(Document{std::move(match.record.id), std::move(match.record.content),
                                   match.score, std::move(match.record.metadata)});
    }

    Update<S> update;
    update.template write<DocsChannel>(std::move(documents));
    return std::move(update).take();
  }

 private:
  std::shared_ptr<llm::EmbeddingBackend> backend_;
  std::shared_ptr<store::VectorStore> vectors_;
  RetrievalOptions options_;
};

template <class S, core::FixedString QueryChannel = "query",
          core::FixedString DocsChannel = "documents">
[[nodiscard]] std::unique_ptr<Node> make_retrieval_node(
    std::shared_ptr<llm::EmbeddingBackend> backend,
    std::shared_ptr<store::VectorStore> vectors, RetrievalOptions options = {}) {
  return std::make_unique<RetrievalNode<S, QueryChannel, DocsChannel>>(
      std::move(backend), std::move(vectors), std::move(options));
}

}
