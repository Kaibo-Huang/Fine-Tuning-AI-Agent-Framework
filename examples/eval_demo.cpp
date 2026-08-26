// eval_demo: the measurement stack end to end. Run an agent over the built-in
// text-to-SQL task suite, score it with the eval harness, record the run as the
// suite's baseline, and export the verified traces as a training dataset for
// distilling a smaller student model.
//
// Offline, a scripted agent plays the model, with two deliberate mistakes so
// the report shows real failures (one wrong comparison operator that only a
// result-set verifier catches, and one refusal). Set AF_BACKEND in .env to
// score a live model instead. Walkthrough: docs/examples.md.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "agents_framework/core/dotenv.hpp"
#include "agents_framework/eval/eval_store.hpp"
#include "agents_framework/eval/harness.hpp"
#include "agents_framework/eval/text_to_sql.hpp"
#include "agents_framework/llm/backend_factory.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/trace/dataset.hpp"
#include "agents_framework/trace/trace_store.hpp"

using namespace agents_framework::core;
using namespace agents_framework::eval;
using namespace agents_framework::llm;
using namespace agents_framework::store;
using namespace agents_framework::trace;
using std::string;

namespace {

// Unwrap a Result or abort the demo with its error.
template <typename T>
T need(Result<T> result) {
  if (!result) {
    std::printf("error: %s\n", result.error().to_string().c_str());
    std::exit(1);
  }
  return std::move(*result);
}

void need(Result<void> result) {
  if (!result) {
    std::printf("error: %s\n", result.error().to_string().c_str());
    std::exit(1);
  }
}

// An offline stand-in for a model. It answers with the gold SQL except for two
// deliberate mistakes: "older-than-30" writes >= where the question says >
// (the classic near-miss only a result-set verifier catches), and
// "big-spenders" refuses to answer at all.
AgentFn scripted_agent() {
  return [](const TaskInstance& instance) -> Result<AgentOutput> {
    string sql = instance.expected.at("gold_sql").get<string>();
    if (instance.id == "older-than-30") sql = "SELECT COUNT(*) FROM customers WHERE age >= 30";
    if (instance.id == "big-spenders") sql = "I am not sure how to write that query.";

    AgentOutput output;
    output.output = sql;

    Trace trace;
    trace.run_id = "demo-" + instance.id;
    trace.transcript = {
        Message::user_text(instance.input.value("question", string{})),
        Message::assistant_text(sql),
    };
    output.trace = std::move(trace);
    return output;
  };
}

// The same agent backed by a real model: one system prompt describing the
// schema, temperature 0, exactly one SQL query back.
AgentFn llm_agent(std::shared_ptr<LLMBackend> backend, string model) {
  return [backend = std::move(backend),
          model = std::move(model)](const TaskInstance& instance) -> Result<AgentOutput> {
    ChatRequest request;
    request.model = model;
    request.system =
        "You translate questions into SQLite SQL for this schema:\n"
        "customers(id, name, city, age)\n"
        "orders(id, customer_id, total, status)  -- status: completed|pending|refunded\n"
        "Reply with exactly one SQL query and nothing else.";
    request.messages.push_back(
        Message::user_text(instance.input.value("question", string{})));
    request.sampling.temperature = 0.0;
    AF_TRY(const auto response, backend->generate(request));

    AgentOutput output;
    output.output = response.text();

    Trace trace;
    trace.transcript = {request.messages.front(), Message{Role::Assistant, response.content}};
    output.trace = std::move(trace);
    return output;
  };
}

void print_report(const EvalReport& report) {
  std::printf("suite      %s\n", report.suite.c_str());
  std::printf("model      %s\n", report.model.empty() ? "(scripted)" : report.model.c_str());
  std::printf("split      %s\n", report.split.empty() ? "all" : report.split.c_str());
  std::printf("instances  %zu  (failures: %zu)\n", report.total, report.failures);
  std::printf("accuracy   %.3f  [%.3f, %.3f] 95%% CI\n", report.mean_score, report.ci.low,
              report.ci.high);
  std::printf("pinned     seed=%llu version=%s commit=%s eval_id=%s\n",
              static_cast<unsigned long long>(report.seed), report.framework_version.c_str(),
              report.framework_commit.c_str(), report.eval_id.c_str());
  for (const auto& result : report.results) {
    std::printf("  %-20s %s%.1f%s\n", result.task_id.c_str(), result.ok ? "" : "FAILED ",
                result.score, result.verified ? "" : " (unverified)");
  }
}

}  // namespace

int main() {
  (void)load_dotenv();

  // The REF-A suite: natural-language questions over a small retail schema,
  // verified by executing the predicted and gold SQL and comparing result sets.
  auto suite = need(TextToSqlSuite::create(default_text_to_sql_spec()));

  // Traces and eval reports persist side by side in one SQLite database.
  auto db = std::make_shared<Db>(need(Db::open("eval_demo.sqlite")));
  auto traces = need(TraceStore::open(db));
  auto evals = need(EvalStore::open(db));

  EvalOptions options;
  options.traces = &traces;
  options.store = &evals;
  options.split = Split::Train;

  // Score a live model when one is configured, the scripted agent otherwise.
  AgentFn agent;
  const auto selection = select_backend();
  if (selection && selection->live) {
    auto backend = need(make_backend(*selection, {}, system_env()));
    std::printf("agent: %s\n\n", selection->describe().c_str());
    options.model = selection->model;
    options.workers = 1;
    agent = llm_agent(std::move(backend), selection->model);
  } else {
    std::printf("agent: scripted offline agent (set AF_BACKEND for a live model)\n\n");
    options.workers = 4;
    agent = scripted_agent();
  }

  // Fan the agent across the split and aggregate into accuracy ± 95% CI.
  const auto report = need(run_eval(suite, agent, options));
  print_report(report);

  // Record this run as the baseline every later run is compared against.
  need(evals.set_baseline(report.eval_id));
  std::printf("\nrecorded as the baseline for '%s'\n", report.suite.c_str());

  // The traces whose SQL actually verified become the training data a smaller
  // student model would be distilled on.
  TraceQuery query;
  query.suite = report.suite;
  query.verified = true;
  const auto verified = need(traces.select(query));
  const auto dataset = need(build_dataset(verified, {}));
  need(write_jsonl(dataset, "eval_demo_dataset.jsonl"));
  std::printf("wrote %zu verified training examples to eval_demo_dataset.jsonl\n",
              dataset.size());
  return 0;
}
