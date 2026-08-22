#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/eval_store.hpp"
#include "agents_framework/eval/harness.hpp"
#include "agents_framework/eval/task_suite.hpp"
#include "agents_framework/eval/text_to_sql.hpp"
#include "agents_framework/eval/verifiers.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/trace/dataset.hpp"
#include "agents_framework/trace/trace_store.hpp"

namespace core = agents_framework::core;
namespace eval = agents_framework::eval;
namespace llm = agents_framework::llm;
namespace store = agents_framework::store;
namespace trace = agents_framework::trace;

namespace {

eval::ListTaskSuite counting_suite(std::size_t n, eval::Split split = eval::Split::Train) {
  std::vector<eval::TaskInstance> instances;
  for (std::size_t i = 0; i < n; ++i) {
    eval::TaskInstance instance;
    instance.id = "case-" + std::to_string(i);
    instance.split = split;
    instance.input["question"] = "echo " + instance.id;
    instance.expected["answer"] = instance.id;
    instances.push_back(std::move(instance));
  }
  auto suite = eval::ListTaskSuite::create("counting", std::move(instances),
                                           eval::make_exact_match_verifier());
  REQUIRE(suite);
  return std::move(*suite);
}

eval::AgentFn modular_agent(std::size_t correct_below) {
  return [correct_below](const eval::TaskInstance& instance)
             -> core::Result<eval::AgentOutput> {
    const std::size_t index = std::stoull(instance.id.substr(5));
    eval::AgentOutput output;
    output.output = index < correct_below ? instance.id : "wrong";

    trace::Trace t;
    t.transcript = {
        llm::Message::user_text(instance.input.value("question", std::string{})),
        llm::Message::assistant_text(output.output),
    };
    output.trace = std::move(t);
    return output;
  };
}

}

TEST_CASE("the harness aggregates to mean and a binomial CI", "[eval][harness]") {
  auto suite = counting_suite(20);
  const auto report = eval::run_eval(suite, modular_agent(15));
  REQUIRE(report);
  REQUIRE(report->total == 20);
  REQUIRE(report->failures == 0);
  REQUIRE(report->mean_score == 0.75);
  REQUIRE(report->ci.low > 0.50);
  REQUIRE(report->ci.low < 0.60);
  REQUIRE(report->ci.high > 0.85);
  REQUIRE(report->ci.high < 0.92);
  REQUIRE(report->results.size() == 20);
  REQUIRE(!report->eval_id.empty());
  REQUIRE(!report->framework_version.empty());
  REQUIRE(!report->framework_commit.empty());
}

TEST_CASE("two runs of the same config agree", "[eval][harness]") {
  auto suite = counting_suite(20);
  eval::EvalOptions options;
  options.seed = 7;
  options.limit = 10;
  const auto first = eval::run_eval(suite, modular_agent(12), options);
  const auto second = eval::run_eval(suite, modular_agent(12), options);
  REQUIRE(first);
  REQUIRE(second);
  REQUIRE(first->mean_score == second->mean_score);
  REQUIRE(first->ci == second->ci);
  REQUIRE(first->results.size() == second->results.size());
  for (std::size_t i = 0; i < first->results.size(); ++i) {
    REQUIRE(first->results[i].task_id == second->results[i].task_id);
    REQUIRE(first->results[i].score == second->results[i].score);
  }
  eval::EvalOptions reseeded = options;
  reseeded.seed = 8;
  const auto third = eval::run_eval(suite, modular_agent(12), reseeded);
  REQUIRE(third);
  bool any_difference = third->results.size() != first->results.size();
  for (std::size_t i = 0; !any_difference && i < first->results.size(); ++i) {
    any_difference = first->results[i].task_id != third->results[i].task_id;
  }
  REQUIRE(any_difference);
}

TEST_CASE("concurrent and sequential execution produce identical reports",
          "[eval][harness][concurrency]") {
  auto suite = counting_suite(24);
  eval::EvalOptions sequential;
  sequential.workers = 1;
  eval::EvalOptions parallel;
  parallel.workers = 8;
  const auto a = eval::run_eval(suite, modular_agent(13), sequential);
  const auto b = eval::run_eval(suite, modular_agent(13), parallel);
  REQUIRE(a);
  REQUIRE(b);
  REQUIRE(a->mean_score == b->mean_score);
  REQUIRE(a->ci == b->ci);
  for (std::size_t i = 0; i < a->results.size(); ++i) {
    REQUIRE(a->results[i].task_id == b->results[i].task_id);
    REQUIRE(a->results[i].score == b->results[i].score);
  }
}

TEST_CASE("split slicing selects only the requested split", "[eval][harness]") {
  std::vector<eval::TaskInstance> instances;
  for (std::size_t i = 0; i < 6; ++i) {
    eval::TaskInstance instance;
    instance.id = "case-" + std::to_string(i);
    instance.split = i < 4 ? eval::Split::Train : eval::Split::HeldOut;
    instance.expected["answer"] = instance.id;
    instances.push_back(std::move(instance));
  }
  auto suite = eval::ListTaskSuite::create("split-suite", std::move(instances),
                                           eval::make_exact_match_verifier());
  REQUIRE(suite);

  eval::EvalOptions held_out;
  held_out.split = eval::Split::HeldOut;
  const auto report = eval::run_eval(*suite, modular_agent(6), held_out);
  REQUIRE(report);
  REQUIRE(report->total == 2);
  REQUIRE(report->split == "held_out");
  for (const auto& result : report->results) {
    REQUIRE(result.split == eval::Split::HeldOut);
  }

  eval::EvalOptions retention;
  retention.split = eval::Split::Retention;
  const auto empty = eval::run_eval(*suite, modular_agent(6), retention);
  REQUIRE(!empty);
  REQUIRE(empty.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("mechanical failures score zero and are counted separately", "[eval][harness]") {
  auto suite = counting_suite(4);
  const eval::AgentFn flaky = [](const eval::TaskInstance& instance)
      -> core::Result<eval::AgentOutput> {
    if (instance.id == "case-2") {
      return core::fail(core::ErrorCode::Network, "backend unreachable");
    }
    return eval::AgentOutput{instance.id, std::nullopt};
  };
  const auto report = eval::run_eval(suite, flaky);
  REQUIRE(report);
  REQUIRE(report->total == 4);
  REQUIRE(report->failures == 1);
  REQUIRE(report->mean_score == 0.75);
  for (const auto& result : report->results) {
    if (result.task_id == "case-2") {
      REQUIRE(!result.ok);
      REQUIRE(result.error.find("backend unreachable") != std::string::npos);
      REQUIRE(result.score == 0.0);
    } else {
      REQUIRE(result.ok);
    }
  }
}

TEST_CASE("compare pairs by task id and reports a signed delta with a CI",
          "[eval][harness]") {
  auto suite = counting_suite(30);
  const auto before = eval::run_eval(suite, modular_agent(10));
  const auto after = eval::run_eval(suite, modular_agent(25));
  REQUIRE(before);
  REQUIRE(after);

  const auto improvement = eval::compare(*before, *after);
  REQUIRE(improvement);
  REQUIRE(improvement->common == 30);
  REQUIRE(std::abs(improvement->delta - 0.5) < 1e-12);
  REQUIRE(improvement->improved == 15);
  REQUIRE(improvement->regressed == 0);
  REQUIRE(improvement->unchanged == 15);
  REQUIRE(improvement->significant);
  REQUIRE(improvement->ci.low > 0.0);

  const auto small_before = eval::run_eval(suite, modular_agent(14));
  const auto small_after = eval::run_eval(suite, modular_agent(15));
  REQUIRE(small_before);
  REQUIRE(small_after);
  const auto noise = eval::compare(*small_before, *small_after);
  REQUIRE(noise);
  REQUIRE(std::abs(noise->delta - (1.0 / 30.0)) < 1e-12);
  REQUIRE(!noise->significant);

  const auto same = eval::compare(*before, *before);
  REQUIRE(same);
  REQUIRE(same->delta == 0.0);
  REQUIRE(!same->significant);
}

TEST_CASE("compare with no overlapping ids is an error", "[eval][harness]") {
  auto a = counting_suite(3);
  auto b_suite = counting_suite(3);
  const auto ra = eval::run_eval(a, modular_agent(3));
  REQUIRE(ra);
  eval::EvalReport disjoint = *ra;
  for (auto& result : disjoint.results) result.task_id = "other-" + result.task_id;
  REQUIRE(!eval::compare(*ra, disjoint));
}

TEST_CASE("interval helpers behave at the edges", "[eval][harness]") {
  const auto all_correct = eval::wilson_interval(1.0, 50);
  REQUIRE(all_correct.low > 0.9);
  REQUIRE(all_correct.high > 1.0 - 1e-9);
  const auto all_wrong = eval::wilson_interval(0.0, 50);
  REQUIRE(all_wrong.low < 1e-9);
  REQUIRE(all_wrong.high < 0.1);

  const std::vector<double> scalar{0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7};
  const auto boot = eval::bootstrap_interval(scalar, 42);
  REQUIRE(boot.low < 0.5);
  REQUIRE(boot.high > 0.5);
  REQUIRE(boot == eval::bootstrap_interval(scalar, 42));

  const std::vector<double> binary{1.0, 0.0, 1.0, 1.0};
  REQUIRE(eval::confidence_interval(binary, 1) == eval::wilson_interval(0.75, 4));
  REQUIRE(eval::confidence_interval(scalar, 1) == eval::bootstrap_interval(scalar, 1));
}

TEST_CASE("pass@k separates what one try gets from what any try gets", "[eval][harness]") {
  auto suite = counting_suite(10);
  const auto shifted_agent = [](std::size_t low, std::size_t high) -> eval::AgentFn {
    return [low, high](const eval::TaskInstance& instance) -> core::Result<eval::AgentOutput> {
      const std::size_t index = std::stoull(instance.id.substr(5));
      return eval::AgentOutput{index >= low && index <= high ? instance.id : "wrong",
                               std::nullopt};
    };
  };
  std::vector<eval::EvalReport> runs;
  for (const auto [low, high] :
       {std::pair<std::size_t, std::size_t>{0, 4}, {3, 7}, {5, 9}}) {
    auto report = eval::run_eval(suite, shifted_agent(low, high));
    REQUIRE(report);
    runs.push_back(std::move(*report));
  }

  const auto result = eval::pass_at_k(runs);
  REQUIRE(result);
  REQUIRE(result->k == 3);
  REQUIRE(result->instances == 10);
  REQUIRE(result->pass_at_1 == 0.5);
  REQUIRE(result->pass_at_k == 1.0);
  REQUIRE(result->ci.low > 0.5);

  const auto single = eval::pass_at_k({runs.data(), 1});
  REQUIRE(single);
  REQUIRE(single->pass_at_1 == single->pass_at_k);

  eval::EvalReport truncated = runs.front();
  truncated.results.pop_back();
  const auto bad = eval::pass_at_k(std::vector<eval::EvalReport>{runs.front(), truncated});
  REQUIRE(!bad);
}

TEST_CASE("scoring mode runs without an agent", "[eval][harness]") {
  class ConstantEvaluator final : public eval::Evaluator {
   public:
    core::Result<eval::Evaluation> evaluate(const eval::TaskInstance& instance,
                                            const trace::Trace&) override {
      return eval::Evaluation{instance.id == "case-0" ? 1.0 : 0.0, true, {}};
    }
    [[nodiscard]] bool needs_agent() const override { return false; }
  };

  auto suite = counting_suite(4);
  ConstantEvaluator evaluator;
  const auto report = eval::run_eval(suite, evaluator);
  REQUIRE(report);
  REQUIRE(report->total == 4);
  REQUIRE(report->mean_score == 0.25);
}

TEST_CASE("an agent-requiring evaluator without an agent is rejected", "[eval][harness]") {
  auto suite = counting_suite(2);
  auto verifier = eval::make_exact_match_verifier();
  eval::TaskMetricEvaluator evaluator{*verifier};
  const auto report = eval::run_eval(suite, evaluator);
  REQUIRE(!report);
  REQUIRE(report.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("the whole Phase 3 loop: REF-A -> harness -> traces -> dataset",
          "[eval][harness][e2e]") {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  auto db = std::make_shared<store::Db>(std::move(*opened));
  auto traces = trace::TraceStore::open(db);
  auto evals = eval::EvalStore::open(db);
  REQUIRE(traces);
  REQUIRE(evals);

  auto suite = eval::TextToSqlSuite::create(eval::default_text_to_sql_spec());
  REQUIRE(suite);

  const eval::AgentFn agent = [&suite](const eval::TaskInstance& instance)
      -> core::Result<eval::AgentOutput> {
    std::string sql = instance.expected.at("gold_sql").get<std::string>();
    if (instance.id == "older-than-30") {
      sql = "SELECT COUNT(*) FROM customers WHERE age >= 30";
    } else if (instance.id == "cities") {
      sql = "not sql at all";
    }
    eval::AgentOutput output;
    output.output = sql;
    trace::Trace t;
    t.run_id = "e2e-" + instance.id;
    t.transcript = {
        llm::Message::user_text(instance.input.value("question", std::string{})),
        llm::Message::assistant_text(sql),
    };
    output.trace = std::move(t);
    return output;
  };

  eval::EvalOptions options;
  options.split = eval::Split::Train;
  options.model = "scripted";
  options.workers = 4;
  options.traces = &*traces;
  options.store = &*evals;
  const auto report = eval::run_eval(*suite, agent, options);
  REQUIRE(report);
  REQUIRE(report->total == 6);
  REQUIRE(report->mean_score == 4.0 / 6.0);

  REQUIRE(evals->set_baseline(report->eval_id));
  const auto baseline = evals->baseline("ref-a-text-to-sql");
  REQUIRE(baseline);
  REQUIRE(baseline->mean_score == report->mean_score);
  REQUIRE(baseline->results.size() == 6);

  trace::TraceQuery query;
  query.suite = "ref-a-text-to-sql";
  const auto stored = traces->select(query);
  REQUIRE(stored);
  REQUIRE(stored->size() == 6);

  const auto dataset = trace::build_dataset(*stored, {});
  REQUIRE(dataset);
  REQUIRE(dataset->size() == 4);
  for (const auto& example : *dataset) {
    REQUIRE(trace::validate_example(example));
  }
}
