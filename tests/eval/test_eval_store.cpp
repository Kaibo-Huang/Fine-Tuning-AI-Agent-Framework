#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/eval_store.hpp"
#include "agents_framework/store/db.hpp"

namespace core = agents_framework::core;
namespace eval = agents_framework::eval;
namespace store = agents_framework::store;

namespace {

eval::EvalStore make_store() {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  auto evals = eval::EvalStore::open(std::make_shared<store::Db>(std::move(*opened)));
  REQUIRE(evals);
  return std::move(*evals);
}

eval::EvalReport make_report(std::string eval_id, std::string suite, double mean) {
  eval::EvalReport report;
  report.eval_id = std::move(eval_id);
  report.suite = std::move(suite);
  report.model = "mock";
  report.adapter = "adapter-v1";
  report.label = "test";
  report.split = "held_out";
  report.seed = 7;
  report.framework_version = "0.0.1";
  report.framework_commit = "abc123";
  report.total = 2;
  report.failures = 1;
  report.mean_score = mean;
  report.ci = {mean - 0.1, mean + 0.1};
  report.results = {
      {"case-a", eval::Split::HeldOut, mean, true, true, "", "detail-a", "trace-1"},
      {"case-b", eval::Split::HeldOut, mean, false, false, "agent exploded", "", ""},
  };
  return report;
}

}

TEST_CASE("eval reports round-trip through the store", "[eval][store]") {
  auto evals = make_store();
  const auto saved = make_report("e1", "ref-a", 0.5);
  REQUIRE(evals.save(saved));

  const auto loaded = evals.load("e1");
  REQUIRE(loaded);
  REQUIRE(loaded->eval_id == saved.eval_id);
  REQUIRE(loaded->suite == saved.suite);
  REQUIRE(loaded->model == saved.model);
  REQUIRE(loaded->adapter == saved.adapter);
  REQUIRE(loaded->split == saved.split);
  REQUIRE(loaded->seed == saved.seed);
  REQUIRE(loaded->framework_commit == saved.framework_commit);
  REQUIRE(loaded->total == saved.total);
  REQUIRE(loaded->failures == saved.failures);
  REQUIRE(loaded->mean_score == saved.mean_score);
  REQUIRE(loaded->ci == saved.ci);
  REQUIRE(loaded->results == saved.results);
}

TEST_CASE("duplicate eval ids are rejected", "[eval][store]") {
  auto evals = make_store();
  REQUIRE(evals.save(make_report("e1", "ref-a", 0.5)));
  REQUIRE(!evals.save(make_report("e1", "ref-a", 0.6)));

  const auto loaded = evals.load("e1");
  REQUIRE(loaded);
  REQUIRE(loaded->mean_score == 0.5);
}

TEST_CASE("reports without an id or suite are rejected", "[eval][store]") {
  auto evals = make_store();
  auto no_id = make_report("", "ref-a", 0.5);
  REQUIRE(!evals.save(no_id));
  auto no_suite = make_report("e1", "", 0.5);
  REQUIRE(!evals.save(no_suite));
}

TEST_CASE("list filters by suite", "[eval][store]") {
  auto evals = make_store();
  REQUIRE(evals.save(make_report("e1", "ref-a", 0.4)));
  REQUIRE(evals.save(make_report("e2", "ref-a", 0.6)));
  REQUIRE(evals.save(make_report("e3", "ref-b", 0.9)));

  const auto all = evals.list();
  REQUIRE(all);
  REQUIRE(all->size() == 3);

  const auto ref_a = evals.list("ref-a");
  REQUIRE(ref_a);
  REQUIRE(ref_a->size() == 2);
  REQUIRE(!ref_a->front().created_at.empty());
}

TEST_CASE("the baseline is one per suite and replaceable", "[eval][store]") {
  auto evals = make_store();
  REQUIRE(evals.save(make_report("e1", "ref-a", 0.4)));
  REQUIRE(evals.save(make_report("e2", "ref-a", 0.6)));
  REQUIRE(evals.save(make_report("e3", "ref-b", 0.9)));

  const auto none = evals.baseline("ref-a");
  REQUIRE(!none);
  REQUIRE(none.error().code == core::ErrorCode::NotFound);

  REQUIRE(evals.set_baseline("e1"));
  auto baseline = evals.baseline("ref-a");
  REQUIRE(baseline);
  REQUIRE(baseline->eval_id == "e1");

  REQUIRE(evals.set_baseline("e2"));
  baseline = evals.baseline("ref-a");
  REQUIRE(baseline);
  REQUIRE(baseline->eval_id == "e2");
  const auto listed = evals.list("ref-a");
  REQUIRE(listed);
  std::size_t baselines = 0;
  for (const auto& info : *listed) {
    if (info.baseline) ++baselines;
  }
  REQUIRE(baselines == 1);
  REQUIRE(!evals.baseline("ref-b"));

  const auto missing = evals.set_baseline("ghost");
  REQUIRE(!missing);
  REQUIRE(missing.error().code == core::ErrorCode::NotFound);
}
