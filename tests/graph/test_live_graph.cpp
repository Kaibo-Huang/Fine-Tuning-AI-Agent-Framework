#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../live_env.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/events.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/retrieval_node.hpp"
#include "agents_framework/graph/subgraph_node.hpp"
#include "agents_framework/store/checkpoint_store.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/store/vector_store.hpp"

using namespace agents_framework;
using Catch::Matchers::ContainsSubstring;

namespace {

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using Query = graph::Channel<"query", std::string>;
using Docs = graph::Channel<"documents", std::vector<graph::Document>>;
using Results = graph::Channel<"results", std::vector<std::string>, graph::Append>;

using AgentSchema = graph::Schema<Messages>;
using RagSchema = graph::Schema<Query, Docs, Messages>;
using SupervisorSchema = graph::Schema<Query, Results>;

std::string text_of(const llm::Message& message) {
  std::string out;
  for (const auto& block : message.content) {
    if (const auto* text = std::get_if<llm::TextBlock>(&block)) out += text->text;
  }
  return out;
}

std::string lowered(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

std::shared_ptr<store::VectorStore> seeded_store(
    const std::shared_ptr<llm::EmbeddingBackend>& embedder, const std::shared_ptr<store::Db>& db) {
  const std::vector<std::pair<std::string, std::string>> corpus{
      {"protocol",
       "The Zorblatt handshake protocol requires exactly seven acknowledgement rounds "
       "before a session is considered open."},
      {"pasta", "Fresh egg pasta cooks in about two minutes in salted boiling water."},
      {"tides", "Tides are caused by the gravitational pull of the moon and the sun."},
  };

  std::vector<std::string> contents;
  for (const auto& [id, content] : corpus) contents.push_back(content);

  const auto embedded = embedder->embed({"", contents});
  REQUIRE(embedded.has_value());
  REQUIRE(embedded->embeddings.size() == corpus.size());

  auto opened = store::VectorStore::open(db, "live_knowledge",
                                         embedded->embeddings.front().size());
  REQUIRE(opened.has_value());
  auto vectors = std::make_shared<store::VectorStore>(std::move(*opened));

  for (std::size_t i = 0; i < corpus.size(); ++i) {
    const auto upserted = vectors->upsert(
        {corpus[i].first, embedded->embeddings[i], corpus[i].second, {}});
    REQUIRE(upserted.has_value());
  }
  return vectors;
}

}

TEST_CASE("live: real embeddings rank the semantically closest document first",
          "[.live][embedding][store]") {
  auto embedder = test_support::live_embedder();
  if (!embedder) SKIP("set OPENAI_API_KEY to run live embedding tests");

  auto opened = store::Db::open_memory();
  REQUIRE(opened.has_value());
  auto db = std::make_shared<store::Db>(std::move(*opened));
  auto vectors = seeded_store(embedder, db);

  CHECK(vectors->dimensions() == test_support::kOpenAiEmbeddingDims);

  const auto query = embedder->embed({"", {"How many rounds does the Zorblatt handshake need?"}});
  REQUIRE(query.has_value());

  const auto matches = vectors->query(query->embeddings.front(), 3);
  REQUIRE(matches.has_value());
  REQUIRE(matches->size() == 3);
  CHECK(matches->front().record.id == "protocol");
  CHECK(matches->front().score > (*matches)[1].score);
}

TEST_CASE("live: a RAG graph answers from retrieved context", "[.live][graph][rag]") {
  const auto live = test_support::live_backend();
  auto embedder = test_support::live_embedder();
  if (!live || !embedder) SKIP("no live backend/embedder configured");

  auto opened = store::Db::open_memory();
  REQUIRE(opened.has_value());
  auto db = std::make_shared<store::Db>(std::move(*opened));
  auto vectors = seeded_store(embedder, db);

  graph::GraphBuilder<RagSchema> builder;
  builder
      .add_node("retrieve", graph::make_retrieval_node<RagSchema>(embedder, vectors,
                                                                  graph::RetrievalOptions{.k = 2}))
      .add_node("compose",
                [](graph::StateView<RagSchema> view) {
                  const std::string context = graph::render_documents(view.get<"documents">());
                  return graph::Update<RagSchema>{}.write<"messages">(
                      {llm::Message::user_text("Use only this context:\n" + context +
                                               "\nQuestion: " + view.get<"query">())});
                })
      .add_node("answer", graph::make_llm_node<RagSchema>(
                              live->backend,
                              graph::LlmNodeOptions{.system = "Answer in one short sentence "
                                                              "using only the given context.",
                                                    .sampling = {.max_tokens = 128}}))
      .set_entry("retrieve")
      .add_edge("retrieve", "compose")
      .add_edge("compose", "answer")
      .set_finish("answer");

  auto compiled = std::move(builder).compile();
  REQUIRE(compiled.has_value());

  graph::State<RagSchema> state;
  state.set<"query">("How many acknowledgement rounds does the Zorblatt handshake protocol need?");

  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.max_steps = 10});
  REQUIRE(stats.has_value());

  REQUIRE_FALSE(state.get<"documents">().empty());
  CHECK(state.get<"documents">().front().id == "protocol");

  const std::string answer = text_of(state.get<"messages">().back());
  INFO("answer: " << answer);
  const std::string normalized = lowered(answer);
  const bool grounded = normalized.find("seven") != std::string::npos ||
                        normalized.find("7") != std::string::npos;
  CHECK(grounded);
}

TEST_CASE("live: concurrent branches merge at the barrier in declaration order",
          "[.live][graph][concurrency]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  const auto answer_node = [&](std::string word) {
    graph::LlmNodeOptions options;
    options.system = "Reply with exactly one word: " + word;
    options.sampling.max_tokens = 16;
    options.sampling.temperature = 0.0;
    return graph::make_llm_node<AgentSchema>(live->backend, options);
  };

  graph::GraphBuilder<AgentSchema> builder;
  builder
      .add_node("split",
                [](graph::StateView<AgentSchema>) { return graph::Update<AgentSchema>{}; })
      .add_node("alpha", answer_node("alpha"))
      .add_node("bravo", answer_node("bravo"))
      .set_entry("split")
      .add_edge("split", "alpha")
      .add_edge("split", "bravo")
      .set_finish("alpha")
      .set_finish("bravo");

  auto compiled = std::move(builder).compile();
  REQUIRE(compiled.has_value());

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("go")});

  const auto started = std::chrono::steady_clock::now();
  graph::Executor executor{graph::ExecutorOptions{.workers = 2}};
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.max_steps = 5});
  const auto elapsed = std::chrono::steady_clock::now() - started;
  REQUIRE(stats.has_value());

  CHECK(stats->steps == 2);
  CHECK(stats->node_runs == 3);

  const auto& messages = state.get<"messages">();
  REQUIRE(messages.size() == 3);

  const std::string first = lowered(text_of(messages[1]));
  const std::string second = lowered(text_of(messages[2]));
  CHECK_THAT(first, ContainsSubstring("alpha"));
  CHECK_THAT(second, ContainsSubstring("bravo"));

  const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  INFO("two live calls in " << wall_ms << " ms");
  CHECK(wall_ms > 0);
}

TEST_CASE("live: a run streams token deltas through the event bus", "[.live][graph][events]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  auto events = std::make_shared<graph::EventBus>();
  std::mutex mutex;
  int token_events = 0;
  std::string streamed;
  std::vector<std::string> node_labels;

  events->subscribe([&](const graph::ExecEvent& event) {
    if (const auto* delta = std::get_if<graph::TokenDelta>(&event)) {
      const std::lock_guard lock{mutex};
      ++token_events;
      streamed += delta->text;
      if (node_labels.empty() || node_labels.back() != delta->node) {
        node_labels.push_back(delta->node);
      }
    }
  });

  graph::LlmNodeOptions options;
  options.sampling.max_tokens = 64;
  options.sampling.temperature = 0.0;
  options.events = events;
  options.label = "storyteller";

  graph::GraphBuilder<AgentSchema> builder;
  builder.add_node("answer", graph::make_llm_node<AgentSchema>(live->backend, options))
      .set_entry("answer")
      .set_finish("answer");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled.has_value());

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("Count from 1 to 5, separated by spaces.")});

  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.events = events});
  REQUIRE(stats.has_value());

  CHECK(token_events > 1);
  REQUIRE(node_labels.size() == 1);
  CHECK(node_labels.front() == "storyteller");
  CHECK(streamed == text_of(state.get<"messages">().back()));
}

TEST_CASE("live: a run checkpoints, pauses, and resumes from SQLite",
          "[.live][graph][checkpoint]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  auto opened = store::Db::open_memory();
  REQUIRE(opened.has_value());
  auto db = std::make_shared<store::Db>(std::move(*opened));
  auto store_opened = store::CheckpointStore::open(db);
  REQUIRE(store_opened.has_value());
  auto checkpoints = std::move(*store_opened);

  graph::LlmNodeOptions options;
  options.system = "Reply with exactly one word: resumed";
  options.sampling.max_tokens = 16;
  options.sampling.temperature = 0.0;

  graph::GraphBuilder<AgentSchema> builder;
  builder
      .add_node("prepare",
                [](graph::StateView<AgentSchema>) { return graph::Update<AgentSchema>{}; })
      .add_node("answer", graph::make_llm_node<AgentSchema>(live->backend, options))
      .set_entry("prepare")
      .add_edge("prepare", "answer")
      .set_finish("answer");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled.has_value());

  graph::State<AgentSchema> state;
  state.set<"messages">({llm::Message::user_text("Are you there?")});

  const graph::RunOptions run_options{.run_id = "live-resume",
                                      .checkpointer = &checkpoints,
                                      .interrupt_before = {"answer"}};

  graph::Executor executor;
  const auto paused = executor.run(*compiled, state, run_options);
  REQUIRE(paused.has_value());
  REQUIRE(paused->status == graph::RunStatus::Interrupted);
  CHECK(state.get<"messages">().size() == 1);

  const auto checkpoint = checkpoints.latest("live-resume");
  REQUIRE(checkpoint.has_value());

  auto restored = graph::State<AgentSchema>::deserialize(checkpoint->state);
  REQUIRE(restored.has_value());

  const auto finished = executor.resume(*compiled, *restored, *checkpoint, run_options);
  REQUIRE(finished.has_value());
  CHECK(finished->status == graph::RunStatus::Completed);

  REQUIRE(restored->get<"messages">().size() == 2);
  CHECK_THAT(lowered(text_of(restored->get<"messages">().back())),
             ContainsSubstring("resumed"));
}

TEST_CASE("live: a supervisor routes work into a live worker subgraph",
          "[.live][graph][multi-agent]") {
  const auto live = test_support::live_backend();
  if (!live) SKIP("no live backend configured");

  graph::LlmNodeOptions worker_options;
  worker_options.system = "Reply with exactly one word naming the capital of the country asked.";
  worker_options.sampling.max_tokens = 16;
  worker_options.sampling.temperature = 0.0;

  graph::GraphBuilder<AgentSchema> worker_builder;
  worker_builder
      .add_node("answer", graph::make_llm_node<AgentSchema>(live->backend, worker_options))
      .set_entry("answer")
      .set_finish("answer");
  auto worker = std::move(worker_builder).compile();
  REQUIRE(worker.has_value());

  graph::GraphBuilder<SupervisorSchema> builder;
  builder
      .add_node("worker",
                graph::make_subgraph_node<SupervisorSchema, AgentSchema>(
                    std::move(*worker),
                    [](graph::StateView<SupervisorSchema> parent)
                        -> core::Result<graph::State<AgentSchema>> {
                      graph::State<AgentSchema> child;
                      child.set<"messages">({llm::Message::user_text(parent.get<"query">())});
                      return child;
                    },
                    [](const graph::State<AgentSchema>& child)
                        -> core::Result<graph::Update<SupervisorSchema>> {
                      return graph::Update<SupervisorSchema>{}.write<"results">(
                          {text_of(child.get<"messages">().back())});
                    }))
      .set_entry("worker")
      .set_finish("worker");

  auto compiled = std::move(builder).compile();
  REQUIRE(compiled.has_value());

  graph::State<SupervisorSchema> state;
  state.set<"query">("What is the capital of Japan?");

  graph::Executor executor;
  const auto stats = executor.run(*compiled, state, graph::RunOptions{.max_steps = 5});
  REQUIRE(stats.has_value());

  REQUIRE(state.get<"results">().size() == 1);
  CHECK_THAT(state.get<"results">().front(), ContainsSubstring("Tokyo"));
}
