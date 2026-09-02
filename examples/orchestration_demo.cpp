// orchestration_demo: a supervisor that delegates to two sub-agents, with
// retrieval, streaming events, SQLite checkpoints, and a human-in-the-loop
// pause, all in one run.
//
// The supervisor routes each task to a research agent (retrieve documents,
// compose a prompt, answer) or a math agent, then merges their results into a
// final report. Each sub-agent is a complete graph of its own, mounted as a
// single node. The run checkpoints to SQLite after every super-step and is
// configured to interrupt before "report"; the demo then reloads the latest
// checkpoint and resumes, exactly as a human-approval flow would.
//
// Offline by default (mock backend + mock embeddings). Set AF_BACKEND in .env
// for a live run. Walkthrough: docs/examples.md.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/prelude.hpp"

using std::pair;
using std::string;
using std::vector;

namespace {

constexpr std::size_t kDims = 64;

// The supervisor's state, declared as typed channels. "results" appends so
// both sub-agents can report without overwriting each other. The conversation
// channel (Messages) and the math agent's schema (ChatSchema) come prebuilt.
using Task = Channel<"task", string>;
using Query = Channel<"query", string>;
using Docs = Channel<"documents", vector<Document>>;
using Results = Channel<"results", vector<string>, Append>;

using SupervisorSchema = Schema<Task, Query, Docs, Results>;
using ResearchSchema = Schema<Query, Docs, Messages>;

// Unwrap a Result or abort the demo with its error.
template <typename T>
T need(Result<T> result) {
  if (!result) {
    std::cerr << result.error().to_string() << "\n";
    std::exit(1);
  }
  return std::move(*result);
}

// Offline stand-ins for the two models. The researcher only answers well when
// the retrieved context actually reached its prompt.
MockBackend::Handler canned_reply_for(const string& role) {
  return [role](const ChatRequest& request) -> Result<ChatResponse> {
    string prompt;
    for (const auto& message : request.messages) {
      for (const auto& block : message.content) {
        if (const auto* text = std::get_if<TextBlock>(&block)) prompt += text->text;
      }
    }
    ChatResponse response;
    if (role == "researcher") {
      const bool grounded = prompt.find("Pregel") != string::npos;
      response.content.push_back(TextBlock{
          grounded ? "Based on the retrieved context, the executor follows the Pregel "
                     "super-step model: nodes run in parallel and merge at a barrier."
                   : "I could not find relevant context."});
    } else {
      response.content.push_back(TextBlock{"12 x 12 = 144."});
    }
    return response;
  };
}

std::shared_ptr<LLMBackend> backend_for(const string& role) {
  BackendOptions options;
  options.max_tokens = 512;
  options.mock_handler = canned_reply_for(role);
  return need(backend_from_env(std::move(options)));
}

// The research sub-agent: retrieve documents, compose a prompt, answer.

Update<ResearchSchema> compose_prompt(StateView<ResearchSchema> view) {
  const string context = render_documents(view.get<"documents">());
  return Update<ResearchSchema>{}.write<"messages">(
      {Message::user_text(context + "\nQuestion: " + view.get<"query">())});
}

CompiledGraph build_research_agent(std::shared_ptr<EmbeddingBackend> embedder,
                                   std::shared_ptr<VectorStore> vectors,
                                   std::shared_ptr<EventBus> events) {
  GraphBuilder<ResearchSchema> builder;
  builder
      .add_node("retrieve",
                make_retrieval_node<ResearchSchema>(std::move(embedder), std::move(vectors),
                                                    RetrievalOptions{.k = 2}))
      .add_node("compose", compose_prompt)
      .add_node("answer", make_llm_node<ResearchSchema>(
                              backend_for("researcher"),
                              LlmNodeOptions{.model = "claude-haiku-4-5",
                                             .events = std::move(events),
                                             .label = "researcher"}))
      .set_entry("retrieve")
      .add_edge("retrieve", "compose")
      .add_edge("compose", "answer")
      .set_finish("answer");
  return need(std::move(builder).compile());
}

// The math sub-agent: a single LLM node over the prebuilt chat schema.

CompiledGraph build_math_agent(std::shared_ptr<EventBus> events) {
  GraphBuilder<ChatSchema> builder;
  builder
      .add_node("answer", make_llm_node<ChatSchema>(
                              backend_for("mathematician"),
                              LlmNodeOptions{.model = "claude-haiku-4-5",
                                             .events = std::move(events),
                                             .label = "mathematician"}))
      .set_entry("answer")
      .set_finish("answer");
  return need(std::move(builder).compile());
}

// The supervisor and its sub-agents have different schemas; each sub-agent
// gets an enter function (parent state -> child state) and a report function
// (finished child state -> update for the parent).

Result<State<ResearchSchema>> enter_research(StateView<SupervisorSchema> parent) {
  State<ResearchSchema> child;
  child.set<"query">(parent.get<"query">());
  return child;
}

Result<Update<SupervisorSchema>> report_research(const State<ResearchSchema>& child) {
  return Update<SupervisorSchema>{}.write<"results">(
      {"research: " + last_assistant_text(child.get<"messages">())});
}

Result<State<ChatSchema>> enter_math(StateView<SupervisorSchema> parent) {
  return chat_state(parent.get<"query">());
}

Result<Update<SupervisorSchema>> report_math(const State<ChatSchema>& child) {
  return Update<SupervisorSchema>{}.write<"results">(
      {"math: " + last_assistant_text(child.get<"messages">())});
}

Update<SupervisorSchema> start_task(StateView<SupervisorSchema> view) {
  return Update<SupervisorSchema>{}.write<"query">(view.get<"task">());
}

// Route arithmetic to the math agent and everything else to research.
string pick_agent(StateView<SupervisorSchema> view) {
  const auto& task = view.get<"task">();
  return task.find("times") != string::npos ? "math_agent" : "research_agent";
}

Update<SupervisorSchema> write_report(StateView<SupervisorSchema> view) {
  string report = "FINAL REPORT\n";
  for (const auto& result : view.get<"results">()) report += "  - " + result + "\n";
  return Update<SupervisorSchema>{}.write<"results">({std::move(report)});
}

CompiledGraph build_supervisor(std::shared_ptr<EmbeddingBackend> embedder,
                               std::shared_ptr<VectorStore> vectors,
                               std::shared_ptr<EventBus> events) {
  GraphBuilder<SupervisorSchema> builder;
  builder
      .add_node("supervisor", start_task)
      .add_node("research_agent",
                make_subgraph_node<SupervisorSchema, ResearchSchema>(
                    build_research_agent(std::move(embedder), std::move(vectors), events),
                    enter_research, report_research))
      .add_node("math_agent", make_subgraph_node<SupervisorSchema, ChatSchema>(
                                  build_math_agent(events), enter_math, report_math))
      .add_node("report", write_report)
      .set_entry("supervisor")
      .add_conditional_edge("supervisor", pick_agent)
      .add_edge("research_agent", "report")
      .add_edge("math_agent", "report")
      .set_finish("report");
  return need(std::move(builder).compile());
}

// Embed a tiny corpus into the vector store. The third entry is a decoy the
// retrieval step should rank below the two relevant ones.
void seed_knowledge_base(EmbeddingBackend& embedder, VectorStore& vectors) {
  const vector<pair<string, string>> corpus{
      {"executor", "The graph executor follows the Pregel super-step model: active nodes run "
                   "in parallel and their outputs merge at a deterministic barrier."},
      {"tools", "Tools are registered with JSON-schema definitions and validated arguments."},
      {"pasta", "Fresh pasta cooks in about two minutes."},
  };
  for (const auto& [id, content] : corpus) {
    if (const auto embedded = embedder.embed({"", {content}})) {
      (void)vectors.upsert({id, embedded->embeddings.front(), content, {}});
    }
  }
}

// Print each super-step as it starts, stream tokens, and announce pauses.
void watch_events(EventBus& events) {
  events.subscribe([](const ExecEvent& event) {
    if (const auto* step = std::get_if<StepStarted>(&event)) {
      std::cout << "[step " << step->step << "]";
      for (const auto& node : step->nodes) std::cout << " " << node;
      std::cout << "\n";
    } else if (const auto* token = std::get_if<TokenDelta>(&event)) {
      std::cout << token->text << std::flush;
    } else if (std::get_if<RunInterrupted>(&event)) {
      std::cout << "[run paused for approval]\n";
    }
  });
}

}  // namespace

int main() {
  if (const auto env = load_dotenv()) {
    std::cout << "[env] loaded " << env->applied.size() << " variable(s) from "
              << env->path.string() << "\n";
  }
  std::cout << "[backend] " << need(select_backend()).describe() << "\n";

  // Everything persistent (vectors and checkpoints) shares one SQLite db.
  auto db = need(Db::open_memory_shared());
  auto embedder = std::make_shared<MockEmbeddingBackend>(kDims);
  auto vectors = std::make_shared<VectorStore>(need(VectorStore::open(db, "knowledge", kDims)));
  auto checkpoints = need(CheckpointStore::open(db));

  seed_knowledge_base(*embedder, *vectors);

  auto events = std::make_shared<EventBus>();
  watch_events(*events);

  auto graph = build_supervisor(embedder, vectors, events);

  State<SupervisorSchema> state;
  state.set<"task">("How does the graph executor schedule nodes?");

  const RunOptions options{.run_id = "demo-run",
                           .checkpointer = &checkpoints,
                           .events = events,
                           .interrupt_before = {"report"}};

  // Run on two workers until the executor interrupts before "report".
  Executor executor{ExecutorOptions{.workers = 2}};
  const auto paused = need(executor.run(graph, state, options));

  if (paused.status == RunStatus::Interrupted) {
    std::cout << "\npending: ";
    for (const auto& node : paused.pending_nodes) std::cout << node << " ";
    std::cout << "\napproving and resuming from the SQLite checkpoint...\n\n";

    // A real application would ask a human here. We approve unconditionally:
    // load the latest checkpoint, restore the state, and resume the run.
    const auto checkpoint = need(checkpoints.latest("demo-run"));
    auto restored = need(State<SupervisorSchema>::deserialize(checkpoint.state));
    need(executor.resume(graph, restored, checkpoint, options));
    state = std::move(restored);
  }

  std::cout << "\n" << state.get<"results">().back() << "\n";

  if (const auto history = checkpoints.list("demo-run")) {
    std::cout << "checkpoints saved: " << history->size()
              << " (replay any step with CheckpointStore::fork)\n";
  }
  return 0;
}
