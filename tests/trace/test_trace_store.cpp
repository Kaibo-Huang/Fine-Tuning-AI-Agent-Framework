#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/trace/trace_store.hpp"

namespace core = agents_framework::core;
namespace llm = agents_framework::llm;
namespace store = agents_framework::store;
namespace trace = agents_framework::trace;

namespace {

trace::TraceStore make_store() {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  auto traces = trace::TraceStore::open(std::make_shared<store::Db>(std::move(*opened)));
  REQUIRE(traces);
  return std::move(*traces);
}

trace::Trace make_trace(std::string trace_id, std::string run_id, double score, bool verified) {
  trace::Trace t;
  t.trace_id = std::move(trace_id);
  t.run_id = std::move(run_id);
  t.suite = "ref-a";
  t.task_id = "case-" + t.trace_id;
  t.model = "mock";
  t.transcript = {llm::Message::user_text("q"), llm::Message::assistant_text("a")};
  t.node_runs = {{1, "llm", true, ""}};
  t.final_output = "a";
  t.score = score;
  t.verified = verified;
  return t;
}

}

TEST_CASE("traces round-trip through the store", "[trace][store]") {
  auto traces = make_store();
  const trace::Trace saved = make_trace("t1", "run-1", 1.0, true);
  REQUIRE(traces.save(saved));

  const auto loaded = traces.load("t1");
  REQUIRE(loaded);
  REQUIRE(*loaded == saved);
}

TEST_CASE("saving the same trace id twice overwrites it", "[trace][store]") {
  auto traces = make_store();
  REQUIRE(traces.save(make_trace("t1", "run-1", 0.0, false)));
  trace::Trace updated = make_trace("t1", "run-1", 1.0, true);
  updated.final_output = "corrected";
  REQUIRE(traces.save(updated));

  const auto loaded = traces.load("t1");
  REQUIRE(loaded);
  REQUIRE(loaded->final_output == "corrected");
  REQUIRE(loaded->verified);

  const auto listed = traces.list({});
  REQUIRE(listed);
  REQUIRE(listed->size() == 1);
}

TEST_CASE("an empty trace id is rejected and missing ids report NotFound", "[trace][store]") {
  auto traces = make_store();
  const auto saved = traces.save(trace::Trace{});
  REQUIRE(!saved);
  REQUIRE(saved.error().code == core::ErrorCode::Invalid);

  const auto missing = traces.load("ghost");
  REQUIRE(!missing);
  REQUIRE(missing.error().code == core::ErrorCode::NotFound);
}

TEST_CASE("queries filter by run, suite, verification, and score", "[trace][store]") {
  auto traces = make_store();
  REQUIRE(traces.save(make_trace("t1", "run-1", 1.0, true)));
  REQUIRE(traces.save(make_trace("t2", "run-1", 0.0, false)));
  REQUIRE(traces.save(make_trace("t3", "run-2", 0.5, false)));
  trace::Trace unscored = make_trace("t4", "run-2", 0.0, false);
  unscored.score.reset();
  REQUIRE(traces.save(unscored));

  trace::TraceQuery by_run;
  by_run.run_id = "run-1";
  auto listed = traces.list(by_run);
  REQUIRE(listed);
  REQUIRE(listed->size() == 2);

  trace::TraceQuery verified_only;
  verified_only.verified = true;
  listed = traces.list(verified_only);
  REQUIRE(listed);
  REQUIRE(listed->size() == 1);
  REQUIRE(listed->front().trace_id == "t1");

  trace::TraceQuery scored;
  scored.min_score = 0.5;
  listed = traces.list(scored);
  REQUIRE(listed);
  REQUIRE(listed->size() == 2);

  trace::TraceQuery limited;
  limited.limit = 2;
  listed = traces.list(limited);
  REQUIRE(listed);
  REQUIRE(listed->size() == 2);

  trace::TraceQuery by_task;
  by_task.suite = "ref-a";
  by_task.task_id = "case-t3";
  const auto full = traces.select(by_task);
  REQUIRE(full);
  REQUIRE(full->size() == 1);
  REQUIRE(full->front().transcript.size() == 2);
}
