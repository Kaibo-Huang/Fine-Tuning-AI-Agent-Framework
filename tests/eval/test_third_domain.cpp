#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/eval_store.hpp"
#include "agents_framework/eval/harness.hpp"
#include "agents_framework/eval/task_suite.hpp"
#include "agents_framework/eval/verifiers.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/llm/mock_backend.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/trace/dataset.hpp"
#include "agents_framework/trace/trace_store.hpp"

namespace core = agents_framework::core;
namespace eval = agents_framework::eval;
namespace llm = agents_framework::llm;
namespace store = agents_framework::store;
namespace trace = agents_framework::trace;

namespace {

struct ConversionCase {
  std::string id;
  eval::Split split;
  double kilometers;
};

constexpr double kKmToMiles = 0.621371;

eval::ListTaskSuite conversion_suite() {
  const std::vector<ConversionCase> cases = {
      {"km-3", eval::Split::Train, 3.0},        {"km-10", eval::Split::Train, 10.0},
      {"km-42", eval::Split::Train, 42.0},      {"km-7", eval::Split::HeldOut, 7.0},
      {"km-55", eval::Split::HeldOut, 55.0},    {"km-120", eval::Split::HeldOut, 120.0},
      {"km-88", eval::Split::HeldOut, 88.0},    {"km-1", eval::Split::Retention, 1.0},
      {"km-200", eval::Split::Retention, 200.0},
  };
  std::vector<eval::TaskInstance> instances;
  for (const ConversionCase& c : cases) {
    eval::TaskInstance instance;
    instance.id = c.id;
    instance.split = c.split;
    instance.input["question"] =
        "Convert " + std::to_string(c.kilometers) + " kilometers to miles.";
    instance.input["kilometers"] = c.kilometers;
    instance.expected["answer"] = c.kilometers * kKmToMiles;
    instances.push_back(std::move(instance));
  }
  eval::NumericOptions tolerance;
  tolerance.abs_tolerance = 0.0;
  tolerance.rel_tolerance = 0.005;
  auto suite = eval::ListTaskSuite::create("unit-conversion", std::move(instances),
                                           eval::make_numeric_verifier(tolerance));
  REQUIRE(suite);
  return std::move(*suite);
}

eval::AgentFn conversion_agent(double wrong_factor_above) {
  return [wrong_factor_above](const eval::TaskInstance& instance)
             -> core::Result<eval::AgentOutput> {
    const double km = instance.input.at("kilometers").get<double>();
    const double factor = km > wrong_factor_above ? 0.6 : kKmToMiles;
    const double miles = km * factor;

    eval::AgentOutput output;
    output.output = "That is about " + std::to_string(miles) + " miles.";
    trace::Trace t;
    t.run_id = "conv-" + instance.id;
    t.transcript = {
        llm::Message::user_text(instance.input.at("question").get<std::string>()),
        llm::Message::assistant_text(output.output),
    };
    output.trace = std::move(t);
    return output;
  };
}

class TableScorer final : public eval::SequenceScorer {
 public:
  core::Result<std::vector<eval::ContinuationScore>> score(
      std::string_view, std::span<const std::string> continuations) override {
    std::vector<eval::ContinuationScore> scores;
    for (const std::string& continuation : continuations) {
      const bool good = continuation.find("correct") != std::string::npos;
      scores.push_back({good ? -1.0 * static_cast<double>(continuation.size()) / 8.0
                             : -1.0 * static_cast<double>(continuation.size()) / 2.0,
                        continuation.size() / 4 + 1});
    }
    return scores;
  }
};

std::shared_ptr<llm::MockBackend> grading_backend() {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_handler([](const llm::ChatRequest& request) -> core::Result<llm::ChatResponse> {
    const auto& prompt = std::get<llm::TextBlock>(request.messages.front().content.front()).text;
    double points = 2.0;
    if (prompt.find("tides") != std::string::npos) points += 3.0;
    if (prompt.find("gravity") != std::string::npos) points += 5.0;
    llm::ChatResponse response;
    response.content.push_back(
        llm::TextBlock{"Graded.\nSCORE: " + std::to_string(points)});
    return response;
  });
  return backend;
}

}

TEST_CASE("a brand-new domain runs the full fine-tuning measurement loop",
          "[eval][third-domain]") {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  auto db = std::make_shared<store::Db>(std::move(*opened));
  auto traces = trace::TraceStore::open(db);
  auto evals = eval::EvalStore::open(db);
  REQUIRE(traces);
  REQUIRE(evals);

  auto suite = conversion_suite();

  eval::EvalOptions held_out;
  held_out.split = eval::Split::HeldOut;
  held_out.model = "conv-v1";
  held_out.workers = 4;
  held_out.traces = &*traces;
  held_out.store = &*evals;
  const auto baseline = eval::run_eval(suite, conversion_agent(50.0), held_out);
  REQUIRE(baseline);
  REQUIRE(baseline->total == 4);
  REQUIRE(baseline->mean_score == 0.25);
  REQUIRE(evals->set_baseline(baseline->eval_id));

  trace::TraceQuery verified;
  verified.suite = "unit-conversion";
  verified.verified = true;
  const auto harvest = traces->select(verified);
  REQUIRE(harvest);
  REQUIRE(harvest->size() == 1);
  const auto dataset = trace::build_dataset(*harvest, {});
  REQUIRE(dataset);
  REQUIRE(dataset->size() == 1);
  REQUIRE(trace::validate_example(dataset->front()));

  eval::EvalOptions candidate = held_out;
  candidate.model = "conv-v1";
  candidate.adapter = "adapter-001";
  const auto adapted = eval::run_eval(suite, conversion_agent(1e9), candidate);
  REQUIRE(adapted);
  REQUIRE(adapted->mean_score == 1.0);

  const auto recorded = evals->baseline("unit-conversion");
  REQUIRE(recorded);
  const auto delta = eval::compare(*recorded, *adapted);
  REQUIRE(delta);
  REQUIRE(delta->delta == 0.75);
  REQUIRE(delta->significant);
  REQUIRE(delta->improved == 3);
  REQUIRE(delta->regressed == 0);

  eval::EvalOptions retention;
  retention.split = eval::Split::Retention;
  const auto retention_before = eval::run_eval(suite, conversion_agent(50.0), retention);
  const auto retention_after = eval::run_eval(suite, conversion_agent(1e9), retention);
  REQUIRE(retention_before);
  REQUIRE(retention_after);
  const auto retention_delta = eval::compare(*retention_before, *retention_after);
  REQUIRE(retention_delta);
  REQUIRE(retention_delta->regressed == 0);
}

TEST_CASE("a multiple-choice domain scores with no agent and no generation",
          "[eval][third-domain]") {
  std::vector<eval::TaskInstance> instances;
  for (int i = 0; i < 4; ++i) {
    eval::TaskInstance instance;
    instance.id = "mc-" + std::to_string(i);
    instance.split = eval::Split::HeldOut;
    instance.input["prompt"] = "Question " + std::to_string(i) + ":";
    instance.expected["choices"] = {" the correct answer", " a wrong but plausible answer",
                                    " another distractor entirely"};
    instance.expected["answer_index"] = i % 2 == 0 ? 0 : 1;
    instances.push_back(std::move(instance));
  }
  auto suite = eval::ListTaskSuite::create("mc-battery", std::move(instances),
                                           eval::make_exact_match_verifier());
  REQUIRE(suite);

  TableScorer scorer;
  eval::AccNormEvaluator evaluator{scorer};
  eval::EvalOptions options;
  options.split = eval::Split::HeldOut;
  const auto report = eval::run_eval(*suite, evaluator, options);
  REQUIRE(report);
  REQUIRE(report->total == 4);
  REQUIRE(report->mean_score == 0.5);
  for (const auto& result : report->results) {
    REQUIRE(result.verified);
    REQUIRE(result.detail == "picked choice 0");
  }
}

TEST_CASE("an open-ended domain gets scalar judge scores and a bootstrap CI",
          "[eval][third-domain]") {
  std::vector<eval::TaskInstance> instances;
  eval::TaskInstance a;
  a.id = "essay-tides";
  a.input = "Explain what causes ocean tides.";
  eval::TaskInstance b;
  b.id = "essay-seasons";
  b.input = "Explain what causes the seasons.";
  eval::TaskInstance c;
  c.id = "essay-eclipse";
  c.input = "Explain what causes a solar eclipse.";
  instances = {a, b, c};

  auto suite = eval::ListTaskSuite::create(
      "explanations", std::move(instances),
      eval::make_llm_judge_verifier(grading_backend()));
  REQUIRE(suite);
  REQUIRE(!suite->verifier().trusted());

  const eval::AgentFn essayist = [](const eval::TaskInstance& instance)
      -> core::Result<eval::AgentOutput> {
    std::string essay = "It happens because of gravity.";
    if (instance.id == "essay-tides") essay += " The tides follow the moon.";
    if (instance.id == "essay-eclipse") essay = "The moon blocks the sun.";
    return eval::AgentOutput{std::move(essay), std::nullopt};
  };

  const auto report = eval::run_eval(*suite, essayist);
  REQUIRE(report);
  REQUIRE(report->total == 3);
  REQUIRE(report->mean_score > 0.6);
  REQUIRE(report->mean_score < 0.7);
  const std::vector<double> scores{1.0, 0.7, 0.2};
  REQUIRE(report->ci == eval::bootstrap_interval(scores, report->seed));
  REQUIRE(report->ci.low <= report->mean_score);
  REQUIRE(report->ci.high >= report->mean_score);
  for (const auto& result : report->results) {
    REQUIRE(!result.verified);
  }

  const auto again = eval::run_eval(*suite, essayist);
  REQUIRE(again);
  REQUIRE(again->mean_score == report->mean_score);
  REQUIRE(again->ci == report->ci);
}
