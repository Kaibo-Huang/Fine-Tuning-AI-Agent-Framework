#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/text_to_sql.hpp"
#include "agents_framework/tools/registry.hpp"

namespace core = agents_framework::core;
namespace eval = agents_framework::eval;
namespace tools = agents_framework::tools;

namespace {

eval::TextToSqlSuite make_suite() {
  auto suite = eval::TextToSqlSuite::create(eval::default_text_to_sql_spec());
  REQUIRE(suite);
  return std::move(*suite);
}

const eval::TaskInstance& find_case(const eval::TaskSuite& suite, std::string_view id) {
  for (const eval::TaskInstance& instance : suite.instances()) {
    if (instance.id == id) return instance;
  }
  FAIL("no case named " + std::string{id});
  static eval::TaskInstance unreachable;
  return unreachable;
}

}

TEST_CASE("the default REF-A suite loads with three database instances", "[eval][ref-a]") {
  auto suite = make_suite();
  REQUIRE(suite.name() == "ref-a-text-to-sql");
  REQUIRE(suite.database_count() == 3);
  REQUIRE(suite.instances().size() == 12);
  REQUIRE(suite.verifier().trusted());
}

TEST_CASE("every gold query verifies to 1.0 against itself", "[eval][ref-a]") {
  auto suite = make_suite();
  for (const eval::TaskInstance& instance : suite.instances()) {
    const std::string gold = instance.expected.at("gold_sql").get<std::string>();
    const auto score = suite.verifier().score(instance, gold);
    REQUIRE(score);
    REQUIRE(*score == 1.0);
  }
}

TEST_CASE("a wrong query scores 0", "[eval][ref-a]") {
  auto suite = make_suite();
  const auto& instance = find_case(suite, "count-customers");
  REQUIRE(suite.verifier().score(instance, "SELECT COUNT(*) FROM orders").value() == 0.0);
}

TEST_CASE("a query that only coincidentally matches on one database is rejected",
          "[eval][ref-a]") {
  auto suite = make_suite();
  const auto& instance = find_case(suite, "older-than-30");
  REQUIRE(suite.verifier()
              .score(instance, "SELECT COUNT(*) FROM customers WHERE age >= 30")
              .value() == 0.0);
  REQUIRE(suite.verifier()
              .score(instance, "SELECT COUNT(*) FROM customers WHERE age > 30")
              .value() == 1.0);
}

TEST_CASE("markdown-fenced SQL is accepted", "[eval][ref-a]") {
  auto suite = make_suite();
  const auto& instance = find_case(suite, "count-customers");
  REQUIRE(suite.verifier()
              .score(instance, "```sql\nSELECT COUNT(*) FROM customers\n```")
              .value() == 1.0);
}

TEST_CASE("unrunnable or empty predictions are wrong answers, not errors", "[eval][ref-a]") {
  auto suite = make_suite();
  const auto& instance = find_case(suite, "count-customers");
  REQUIRE(suite.verifier().score(instance, "SELECT nonsense FROM nowhere").value() == 0.0);
  REQUIRE(suite.verifier().score(instance, "").value() == 0.0);
  REQUIRE(suite.verifier().score(instance, "I don't know.").value() == 0.0);
}

TEST_CASE("mutating queries are rolled back and cannot poison later checks", "[eval][ref-a]") {
  auto suite = make_suite();
  const auto& pending = find_case(suite, "pending-count");

  REQUIRE(suite.verifier().score(pending, "DELETE FROM orders").value() == 0.0);
  REQUIRE(suite.verifier()
              .score(pending, pending.expected.at("gold_sql").get<std::string>())
              .value() == 1.0);

  REQUIRE(suite.verifier().score(pending, "DELETE FROM customers").value() == 0.0);
}

TEST_CASE("the sql tool explores the first database and rolls back mutations",
          "[eval][ref-a]") {
  auto suite = make_suite();
  tools::ToolRegistry registry;
  REQUIRE(registry.add(suite.make_sql_tool()));

  const auto result = registry.invoke("sql", {{"query", "SELECT COUNT(*) FROM customers"}});
  REQUIRE(result);
  const auto parsed = nlohmann::json::parse(*result);
  REQUIRE(parsed.at("rows").size() == 1);
  REQUIRE(parsed.at("rows")[0][0] == "4");

  REQUIRE(registry.invoke("sql", {{"query", "DELETE FROM orders"}}));
  const auto after = registry.invoke("sql", {{"query", "SELECT COUNT(*) FROM orders"}});
  REQUIRE(after);
  REQUIRE(nlohmann::json::parse(*after).at("rows")[0][0] == "5");

  const auto blocked = registry.invoke("sql", {{"query", "DELETE FROM customers"}});
  REQUIRE(!blocked);

  const auto named = registry.invoke("sql", {{"query", "SELECT name, city FROM customers"}});
  REQUIRE(named);
  REQUIRE(nlohmann::json::parse(*named).at("columns") ==
          nlohmann::json::array({"name", "city"}));

  const auto bad = registry.invoke("sql", {{"query", "SELECT FROM"}});
  REQUIRE(!bad);
}

TEST_CASE("suite creation fails loudly on broken specs", "[eval][ref-a]") {
  eval::TextToSqlSpec no_schema = eval::default_text_to_sql_spec();
  no_schema.schema_sql.clear();
  REQUIRE(!eval::TextToSqlSuite::create(no_schema));

  eval::TextToSqlSpec no_seeds = eval::default_text_to_sql_spec();
  no_seeds.seeds.clear();
  REQUIRE(!eval::TextToSqlSuite::create(no_seeds));

  eval::TextToSqlSpec bad_gold = eval::default_text_to_sql_spec();
  bad_gold.cases[0].gold_sql = "SELECT definitely_not_a_column FROM customers";
  const auto result = eval::TextToSqlSuite::create(bad_gold);
  REQUIRE(!result);
  REQUIRE(result.error().context.find("count-customers") != std::string::npos);
}
