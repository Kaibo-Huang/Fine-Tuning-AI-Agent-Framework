#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/events.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/retrieval_node.hpp"
#include "agents_framework/graph/subgraph_node.hpp"
#include "agents_framework/core/dotenv.hpp"
#include "agents_framework/llm/backend_factory.hpp"
#include "agents_framework/llm/mock_backend.hpp"
#include "agents_framework/llm/mock_embedding.hpp"
#include "agents_framework/store/checkpoint_store.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/store/vector_store.hpp"

namespace {

using namespace agents_framework;

constexpr std::size_t kDims = 64;

using Task = graph::Channel<"task", std::string>;
using Query = graph::Channel<"query", std::string>;
using Docs = graph::Channel<"documents", std::vector<graph::Document>>;
using Results = graph::Channel<"results", std::vector<std::string>, graph::Append>;
using SupervisorSchema = graph::Schema<Task, Query, Docs, Results>;

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using AgentSchema = graph::Schema<Messages>;

llm::MockBackend::Handler canned_for(const std::string& role) {
  return [role](const llm::ChatRequest& request) -> core::Result<llm::ChatResponse> {
    std::string prompt;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* text = std::get_if<llm::TextBlock>(&block)) prompt += text->text;
      }
    }
    llm::ChatResponse response;
    if (role == "researcher") {
      const bool grounded = prompt.find("Pregel") != std::string::npos;
      response.content.push_back(llm::TextBlock{
          grounded ? "Based on the retrieved context, the executor follows the Pregel "
                     "super-step model: nodes run in parallel and merge at a barrier."
                   : "I could not find relevant context."});
    } else {
      response.content.push_back(llm::TextBlock{"12 x 12 = 144."});
    }
    return response;
  };
}

std::shared_ptr<llm::LLMBackend> make_backend(const std::string& role) {
  llm::BackendOptions options;
  options.max_tokens = 512;
  options.mock_handler = canned_for(role);

  auto backend = llm::backend_from_env(std::move(options));
  if (!backend) throw std::runtime_error(backend.error().to_string());
  return std::move(*backend);
}

graph::CompiledGraph build_research_agent(std::shared_ptr<llm::EmbeddingBackend> embedder,
                                          std::shared_ptr<store::VectorStore> vectors,
                                          std::shared_ptr<graph::EventBus> events) {
  using ResearchSchema = graph::Schema<Query, Docs, Messages>;
  graph::GraphBuilder<ResearchSchema> builder;
  builder
      .add_node("retrieve", graph::make_retrieval_node<ResearchSchema>(
                                std::move(embedder), std::move(vectors),
                                graph::RetrievalOptions{.k = 2}))
      .add_node("compose",
                [](graph::StateView<ResearchSchema> view) {
                  const std::string context = graph::render_documents(view.get<"documents">());
                  return graph::Update<ResearchSchema>{}.write<"messages">(
                      {llm::Message::user_text(context + "\nQuestion: " + view.get<"query">())});
                })
      .add_node("answer", graph::make_llm_node<ResearchSchema>(
                              make_backend("researcher"),
                              graph::LlmNodeOptions{.model = "claude-haiku-4-5",
                                                    .events = std::move(events),
                                                    .label = "researcher"}))
      .set_entry("retrieve")
      .add_edge("retrieve", "compose")
      .add_edge("compose", "answer")
      .set_finish("answer");
  auto compiled = std::move(builder).compile();
  if (!compiled) throw std::runtime_error(compiled.error().to_string());
  return std::move(*compiled);
}

graph::CompiledGraph build_math_agent(std::shared_ptr<graph::EventBus> events) {
  graph::GraphBuilder<AgentSchema> builder;
  builder
      .add_node("answer", graph::make_llm_node<AgentSchema>(
                              make_backend("mathematician"),
                              graph::LlmNodeOptions{.model = "claude-haiku-4-5",
                                                    .events = std::move(events),
                                                    .label = "mathematician"}))
      .set_entry("answer")
      .set_finish("answer");
  auto compiled = std::move(builder).compile();
  if (!compiled) throw std::runtime_error(compiled.error().to_string());
  return std::move(*compiled);
}

std::string last_assistant_text(const std::vector<llm::Message>& messages) {
  std::string text;
  for (const auto& block : messages.back().content) {
    if (const auto* piece = std::get_if<llm::TextBlock>(&block)) text += piece->text;
  }
  return text;
}

}

int main() {
  if (const auto env_file = agents_framework::core::load_dotenv()) {
    std::cout << "[env] loaded " << env_file->applied.size() << " variable(s) from "
              << env_file->path.string() << "\n";
  }
  if (const auto selection = agents_framework::llm::select_backend()) {
    std::cout << "[backend] " << selection->describe() << "\n";
  } else {
    std::cerr << "[backend] " << selection.error().to_string() << "\n";
    return 1;
  }

  auto opened = store::Db::open_memory();
  if (!opened) {
    std::cerr << opened.error().to_string() << "\n";
    return 1;
  }
  auto db = std::make_shared<store::Db>(std::move(*opened));

  auto embedder = std::make_shared<llm::MockEmbeddingBackend>(kDims);
  auto vectors_opened = store::VectorStore::open(db, "knowledge", kDims);
  if (!vectors_opened) {
    std::cerr << vectors_opened.error().to_string() << "\n";
    return 1;
  }
  auto vectors = std::make_shared<store::VectorStore>(std::move(*vectors_opened));
  const std::vector<std::pair<std::string, std::string>> corpus{
      {"executor", "The graph executor follows the Pregel super-step model: active nodes run "
                   "in parallel and their outputs merge at a deterministic barrier."},
      {"tools", "Tools are registered with JSON-schema definitions and validated arguments."},
      {"pasta", "Fresh pasta cooks in about two minutes."},
  };
  for (const auto& [id, content] : corpus) {
    const auto embedded = embedder->embed({"", {content}});
    if (embedded) (void)vectors->upsert({id, embedded->embeddings.front(), content, {}});
  }

  auto events = std::make_shared<graph::EventBus>();
  events->subscribe([](const graph::ExecEvent& event) {
    if (const auto* step = std::get_if<graph::StepStarted>(&event)) {
      std::cout << "[step " << step->step << "]";
      for (const auto& node : step->nodes) std::cout << " " << node;
      std::cout << "\n";
    } else if (const auto* token = std::get_if<graph::TokenDelta>(&event)) {
      std::cout << token->text << std::flush;
    } else if (std::get_if<graph::RunInterrupted>(&event)) {
      std::cout << "[run paused for approval]\n";
    }
  });

  auto checkpoints_opened = store::CheckpointStore::open(db);
  if (!checkpoints_opened) {
    std::cerr << checkpoints_opened.error().to_string() << "\n";
    return 1;
  }
  auto checkpoints = std::move(*checkpoints_opened);

  graph::GraphBuilder<SupervisorSchema> builder;
  builder
      .add_node("supervisor",
                [](graph::StateView<SupervisorSchema> view) {
                  return graph::Update<SupervisorSchema>{}.write<"query">(view.get<"task">());
                })
      .add_node("research_agent",
                graph::make_subgraph_node<SupervisorSchema, graph::Schema<Query, Docs, Messages>>(
                    build_research_agent(embedder, vectors, events),
                    [](graph::StateView<SupervisorSchema> parent)
                        -> core::Result<graph::State<graph::Schema<Query, Docs, Messages>>> {
                      graph::State<graph::Schema<Query, Docs, Messages>> child;
                      child.set<"query">(parent.get<"query">());
                      return child;
                    },
                    [](const graph::State<graph::Schema<Query, Docs, Messages>>& child)
                        -> core::Result<graph::Update<SupervisorSchema>> {
                      return graph::Update<SupervisorSchema>{}.write<"results">(
                          {"research: " + last_assistant_text(child.get<"messages">())});
                    }))
      .add_node("math_agent",
                graph::make_subgraph_node<SupervisorSchema, AgentSchema>(
                    build_math_agent(events),
                    [](graph::StateView<SupervisorSchema> parent)
                        -> core::Result<graph::State<AgentSchema>> {
                      graph::State<AgentSchema> child;
                      child.set<"messages">({llm::Message::user_text(parent.get<"query">())});
                      return child;
                    },
                    [](const graph::State<AgentSchema>& child)
                        -> core::Result<graph::Update<SupervisorSchema>> {
                      return graph::Update<SupervisorSchema>{}.write<"results">(
                          {"math: " + last_assistant_text(child.get<"messages">())});
                    }))
      .add_node("report",
                [](graph::StateView<SupervisorSchema> view) {
                  std::string report = "FINAL REPORT\n";
                  for (const auto& result : view.get<"results">()) {
                    report += "  - " + result + "\n";
                  }
                  return graph::Update<SupervisorSchema>{}.write<"results">({std::move(report)});
                })
      .set_entry("supervisor")
      .add_conditional_edge("supervisor",
                            [](graph::StateView<SupervisorSchema> view) -> std::string {
                              const auto& task = view.get<"task">();
                              return task.find("times") != std::string::npos ? "math_agent"
                                                                             : "research_agent";
                            })
      .add_edge("research_agent", "report")
      .add_edge("math_agent", "report")
      .set_finish("report");
  auto compiled = std::move(builder).compile();
  if (!compiled) {
    std::cerr << compiled.error().to_string() << "\n";
    return 1;
  }

  graph::State<SupervisorSchema> state;
  state.set<"task">("How does the graph executor schedule nodes?");

  const graph::RunOptions options{.run_id = "demo-run",
                                  .checkpointer = &checkpoints,
                                  .events = events,
                                  .interrupt_before = {"report"}};

  graph::Executor executor{graph::ExecutorOptions{.workers = 2}};
  const auto paused = executor.run(*compiled, state, options);
  if (!paused) {
    std::cerr << paused.error().to_string() << "\n";
    return 1;
  }

  if (paused->status == graph::RunStatus::Interrupted) {
    std::cout << "\npending: ";
    for (const auto& node : paused->pending_nodes) std::cout << node << " ";
    std::cout << "\napproving and resuming from the SQLite checkpoint...\n\n";

    const auto checkpoint = checkpoints.latest("demo-run");
    if (!checkpoint) {
      std::cerr << checkpoint.error().to_string() << "\n";
      return 1;
    }
    auto restored = graph::State<SupervisorSchema>::deserialize(checkpoint->state);
    if (!restored) {
      std::cerr << restored.error().to_string() << "\n";
      return 1;
    }
    const auto finished = executor.resume(*compiled, *restored, *checkpoint, options);
    if (!finished) {
      std::cerr << finished.error().to_string() << "\n";
      return 1;
    }
    state = std::move(*restored);
  }

  std::cout << "\n" << state.get<"results">().back() << "\n";

  const auto history = checkpoints.list("demo-run");
  if (history) {
    std::cout << "checkpoints saved: " << history->size()
              << " (replay any step with CheckpointStore::fork)\n";
  }
  return 0;
}
