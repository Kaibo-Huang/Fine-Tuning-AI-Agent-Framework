#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include "agents_framework/llm/message.hpp"
#include "agents_framework/trace/trace.hpp"

namespace llm = agents_framework::llm;
namespace trace = agents_framework::trace;

namespace {

trace::Trace make_trace() {
  trace::Trace t;
  t.trace_id = "trace-1";
  t.run_id = "run-1";
  t.suite = "ref-a";
  t.task_id = "count-customers";
  t.model = "mock";
  t.transcript = {
      llm::Message::user_text("How many customers are there?"),
      llm::Message{llm::Role::Assistant,
                   {llm::ToolUseBlock{"call_1", "sql", {{"query", "SELECT 1"}}}}},
      llm::Message{llm::Role::User, {llm::ToolResultBlock{"call_1", "[[\"4\"]]", false}}},
      llm::Message::assistant_text("SELECT COUNT(*) FROM customers"),
  };
  t.node_runs = {
      {1, "llm", true, ""},
      {2, "tools", true, ""},
      {3, "llm", true, ""},
  };
  t.final_output = "SELECT COUNT(*) FROM customers";
  t.score = 1.0;
  t.verified = true;
  t.metadata["note"] = "fixture";
  return t;
}

}

TEST_CASE("traces round-trip through JSON", "[trace]") {
  const trace::Trace original = make_trace();
  const nlohmann::json j = original;
  const auto restored = j.get<trace::Trace>();
  REQUIRE(restored == original);
}

TEST_CASE("an unscored trace serializes score as null", "[trace]") {
  trace::Trace t = make_trace();
  t.score.reset();
  const nlohmann::json j = t;
  REQUIRE(j.at("score").is_null());
  REQUIRE(j.get<trace::Trace>() == t);
}

TEST_CASE("final_assistant_text reads the last assistant message", "[trace]") {
  const trace::Trace t = make_trace();
  REQUIRE(trace::final_assistant_text(t.transcript) == "SELECT COUNT(*) FROM customers");

  REQUIRE(trace::final_assistant_text({}) == "");
  REQUIRE(trace::final_assistant_text({llm::Message::user_text("only a user turn")}) == "");
}

TEST_CASE("node runs record raw error text for failed nodes", "[trace]") {
  trace::Trace t = make_trace();
  t.node_runs.push_back({4, "tools", false, "Tool: no such table: ordersx"});
  const nlohmann::json j = t;
  const auto restored = j.get<trace::Trace>();
  REQUIRE(restored.node_runs.back().ok == false);
  REQUIRE(restored.node_runs.back().error == "Tool: no such table: ordersx");
}
