#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/task_suite.hpp"
#include "agents_framework/eval/verifiers.hpp"

namespace core = agents_framework::core;
namespace eval = agents_framework::eval;

namespace {

std::vector<eval::TaskInstance> two_instances() {
  eval::TaskInstance a;
  a.id = "a";
  a.split = eval::Split::Train;
  a.expected["answer"] = "1";
  eval::TaskInstance b;
  b.id = "b";
  b.split = eval::Split::HeldOut;
  b.expected["answer"] = "2";
  return {a, b};
}

}

TEST_CASE("a list suite is a domain in one file", "[eval][suite]") {
  auto suite = eval::ListTaskSuite::create("mini", two_instances(),
                                           eval::make_exact_match_verifier());
  REQUIRE(suite);
  REQUIRE(suite->name() == "mini");
  REQUIRE(suite->instances().size() == 2);
  REQUIRE(suite->verifier().score(suite->instances()[0], "1").value() == 1.0);
}

TEST_CASE("suite validation rejects bad names, ids, and verifiers", "[eval][suite]") {
  REQUIRE(!eval::ListTaskSuite::create("", two_instances(),
                                       eval::make_exact_match_verifier()));
  REQUIRE(!eval::ListTaskSuite::create("mini", two_instances(), nullptr));

  auto duplicate = two_instances();
  duplicate[1].id = "a";
  const auto result =
      eval::ListTaskSuite::create("mini", duplicate, eval::make_exact_match_verifier());
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Invalid);

  auto unnamed = two_instances();
  unnamed[0].id.clear();
  REQUIRE(!eval::ListTaskSuite::create("mini", unnamed, eval::make_exact_match_verifier()));
}

TEST_CASE("split names round-trip", "[eval][suite]") {
  for (const auto split : {eval::Split::Train, eval::Split::HeldOut, eval::Split::Retention}) {
    REQUIRE(eval::split_from_string(eval::split_name(split)) == split);
  }
}
