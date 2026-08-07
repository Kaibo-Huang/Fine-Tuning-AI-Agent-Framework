#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/memory.hpp"
#include "agents_framework/graph/retrieval_node.hpp"
#include "agents_framework/llm/mock_backend.hpp"
#include "agents_framework/llm/mock_embedding.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/store/vector_store.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;
namespace store = agents_framework::store;

namespace {

constexpr std::size_t kDims = 64;

using Query = graph::Channel<"query", std::string>;
using Docs = graph::Channel<"documents", std::vector<graph::Document>>;
using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using RagSchema = graph::Schema<Query, Docs, Messages>;

std::shared_ptr<store::VectorStore> seeded_store(std::shared_ptr<llm::MockEmbeddingBackend> embedder) {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  auto vectors = store::VectorStore::open(std::make_shared<store::Db>(std::move(*opened)),
                                          "docs", kDims);
  REQUIRE(vectors);

  const std::vector<std::pair<std::string, std::string>> corpus{
      {"paris", "The capital of France is Paris."},
      {"berlin", "The capital of Germany is Berlin."},
      {"pasta", "Fresh pasta cooks in two minutes."},
  };
  for (const auto& [id, content] : corpus) {
    const auto embedded = embedder->embed({"", {content}});
    REQUIRE(embedded);
    REQUIRE(vectors->upsert({id, embedded->embeddings.front(), content, {}}));
  }
  return std::make_shared<store::VectorStore>(std::move(*vectors));
}

}

TEST_CASE("the retrieval node writes the most relevant documents", "[graph][retrieval]") {
  auto embedder = std::make_shared<llm::MockEmbeddingBackend>(kDims);
  auto vectors = seeded_store(embedder);

  graph::GraphBuilder<RagSchema> builder;
  builder
      .add_node("retrieve", graph::make_retrieval_node<RagSchema>(embedder, vectors,
                                                                  graph::RetrievalOptions{.k = 2}))
      .set_entry("retrieve")
      .set_finish("retrieve");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<RagSchema> state;
  state.set<"query">("What is the capital of France?");
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));

  const auto& documents = state.get<"documents">();
  REQUIRE(documents.size() == 2);
  REQUIRE(documents.front().id == "paris");
  REQUIRE(documents.front().score > documents.back().score);
}

TEST_CASE("min_score filters weak matches", "[graph][retrieval]") {
  auto embedder = std::make_shared<llm::MockEmbeddingBackend>(kDims);
  auto vectors = seeded_store(embedder);

  graph::GraphBuilder<RagSchema> builder;
  builder
      .add_node("retrieve",
                graph::make_retrieval_node<RagSchema>(
                    embedder, vectors, graph::RetrievalOptions{.k = 3, .min_score = 0.9}))
      .set_entry("retrieve")
      .set_finish("retrieve");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<RagSchema> state;
  state.set<"query">("The capital of France is Paris.");
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));

  const auto& documents = state.get<"documents">();
  REQUIRE(documents.size() == 1);
  REQUIRE(documents.front().id == "paris");
}

TEST_CASE("an empty query fails with a clear error", "[graph][retrieval]") {
  auto embedder = std::make_shared<llm::MockEmbeddingBackend>(kDims);
  auto vectors = seeded_store(embedder);

  graph::GraphBuilder<RagSchema> builder;
  builder.add_node("retrieve", graph::make_retrieval_node<RagSchema>(embedder, vectors))
      .set_entry("retrieve")
      .set_finish("retrieve");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<RagSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(!stats);
  REQUIRE(stats.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("a RAG graph retrieves context and the llm answers with it", "[graph][retrieval]") {
  auto embedder = std::make_shared<llm::MockEmbeddingBackend>(kDims);
  auto vectors = seeded_store(embedder);

  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_handler([](const llm::ChatRequest& request) -> core::Result<llm::ChatResponse> {
    llm::ChatResponse response;
    std::string prompt;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* text = std::get_if<llm::TextBlock>(&block)) prompt += text->text;
      }
    }
    const bool grounded = prompt.find("Paris") != std::string::npos;
    response.content.push_back(
        llm::TextBlock{grounded ? "Paris, according to the context." : "I don't know."});
    return response;
  });

  graph::GraphBuilder<RagSchema> builder;
  builder
      .add_node("retrieve", graph::make_retrieval_node<RagSchema>(embedder, vectors,
                                                                  graph::RetrievalOptions{.k = 2}))
      .add_node("prompt",
                [](graph::StateView<RagSchema> view) {
                  const std::string context = graph::render_documents(view.get<"documents">());
                  return graph::Update<RagSchema>{}.write<"messages">(
                      {llm::Message::user_text(context + "\nQuestion: " + view.get<"query">())});
                })
      .add_node("answer", graph::make_llm_node<RagSchema>(backend))
      .set_entry("retrieve")
      .add_edge("retrieve", "prompt")
      .add_edge("prompt", "answer")
      .set_finish("answer");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<RagSchema> state;
  state.set<"query">("What is the capital of France?");
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 2);
  REQUIRE(messages.back().role == llm::Role::Assistant);
  llm::ChatResponse assembled;
  assembled.content = messages.back().content;
  REQUIRE(assembled.text() == "Paris, according to the context.");
}

TEST_CASE("render_documents numbers each document", "[graph][retrieval]") {
  const std::vector<graph::Document> documents{
      {"a", "first fact", 0.9, {}},
      {"b", "second fact", 0.5, {}},
  };
  const std::string rendered = graph::render_documents(documents);
  REQUIRE(rendered.find("[1] first fact") != std::string::npos);
  REQUIRE(rendered.find("[2] second fact") != std::string::npos);
  REQUIRE(graph::render_documents({}).empty());
}

TEST_CASE("documents serialize through the state channel", "[graph][retrieval]") {
  graph::State<RagSchema> state;
  state.set<"documents">({{"a", "first", 0.9, {{"tag", "x"}}}});
  const auto restored = graph::State<RagSchema>::deserialize(state.serialize());
  REQUIRE(restored);
  REQUIRE(restored->get<"documents">() == state.get<"documents">());
}

TEST_CASE("window_messages keeps the most recent messages", "[graph][memory]") {
  std::vector<llm::Message> messages;
  for (int i = 0; i < 6; ++i) {
    messages.push_back(llm::Message::user_text("m" + std::to_string(i)));
  }
  const auto windowed = graph::window_messages(messages, 2);
  REQUIRE(windowed.size() == 2);
  REQUIRE(windowed.front() == messages[4]);
  REQUIRE(windowed.back() == messages[5]);
  REQUIRE(graph::window_messages(messages, 10).size() == 6);
}

TEST_CASE("window_messages drops orphaned tool results at the window edge",
          "[graph][memory]") {
  std::vector<llm::Message> messages;
  messages.push_back(llm::Message::user_text("question"));
  messages.push_back(
      llm::Message{llm::Role::Assistant, {llm::ToolUseBlock{"call_1", "calc", {}}}});
  messages.push_back(
      llm::Message{llm::Role::User, {llm::ToolResultBlock{"call_1", "42", false}}});
  messages.push_back(llm::Message::assistant_text("the answer is 42"));

  const auto windowed = graph::window_messages(messages, 2);
  REQUIRE(windowed.size() == 1);
  REQUIRE(windowed.front() == messages.back());
}

TEST_CASE("the llm node applies its message window", "[graph][memory]") {
  using Messages2 = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
  using AgentSchema = graph::Schema<Messages2>;

  auto backend = std::make_shared<llm::MockBackend>();
  llm::ChatResponse response;
  response.content.push_back(llm::TextBlock{"ok"});
  backend->push_response(response);

  graph::GraphBuilder<AgentSchema> builder;
  builder
      .add_node("llm", graph::make_llm_node<AgentSchema>(
                           backend, graph::LlmNodeOptions{.max_messages = 2}))
      .set_entry("llm")
      .set_finish("llm");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("one"), llm::Message::user_text("two"),
                         llm::Message::user_text("three")});
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));

  REQUIRE(backend->calls().size() == 1);
  const auto& sent = backend->calls().front().messages;
  REQUIRE(sent.size() == 2);
  REQUIRE(sent.front() == llm::Message::user_text("two"));
  REQUIRE(sent.back() == llm::Message::user_text("three"));
}
