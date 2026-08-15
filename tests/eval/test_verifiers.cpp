#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/verifiers.hpp"
#include "agents_framework/llm/mock_backend.hpp"

namespace core = agents_framework::core;
namespace eval = agents_framework::eval;
namespace llm = agents_framework::llm;

namespace {

eval::TaskInstance instance_with(nlohmann::json expected) {
  eval::TaskInstance instance;
  instance.id = "case-1";
  instance.expected = std::move(expected);
  return instance;
}

}

TEST_CASE("exact match normalizes whitespace and optionally case", "[eval][verifiers]") {
  auto verifier = eval::make_exact_match_verifier();
  const auto instance = instance_with({{"answer", "SELECT name FROM t"}});

  REQUIRE(verifier->trusted());
  REQUIRE(verifier->score(instance, "SELECT name FROM t").value() == 1.0);
  REQUIRE(verifier->score(instance, "  SELECT   name\n FROM t  ").value() == 1.0);
  REQUIRE(verifier->score(instance, "select name from t").value() == 0.0);

  eval::ExactMatchOptions options;
  options.case_insensitive = true;
  auto folded = eval::make_exact_match_verifier(options);
  REQUIRE(folded->score(instance, "select NAME from T").value() == 1.0);
}

TEST_CASE("exact match without gold data is an error, not a zero", "[eval][verifiers]") {
  auto verifier = eval::make_exact_match_verifier();
  const auto result = verifier->score(instance_with({}), "anything");
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("numeric verifier applies tolerances to the first number found",
          "[eval][verifiers]") {
  eval::NumericOptions options;
  options.abs_tolerance = 0.01;
  auto verifier = eval::make_numeric_verifier(options);
  const auto instance = instance_with({{"answer", 542.5}});

  REQUIRE(verifier->score(instance, "The total is 542.5 dollars.").value() == 1.0);
  REQUIRE(verifier->score(instance, "542.505").value() == 1.0);
  REQUIRE(verifier->score(instance, "542.6").value() == 0.0);
  REQUIRE(verifier->score(instance, "no number here").value() == 0.0);

  eval::NumericOptions relative;
  relative.abs_tolerance = 0.0;
  relative.rel_tolerance = 0.01;
  auto rel_verifier = eval::make_numeric_verifier(relative);
  REQUIRE(rel_verifier->score(instance, "545").value() == 1.0);
  REQUIRE(rel_verifier->score(instance, "560").value() == 0.0);
}

TEST_CASE("set match compares as an order-insensitive multiset", "[eval][verifiers]") {
  auto verifier = eval::make_set_match_verifier();
  const auto instance =
      instance_with({{"answer", nlohmann::json::array({"Ada", "Cleo", "Finn"})}});

  REQUIRE(verifier->score(instance, "Finn\nAda\nCleo").value() == 1.0);
  REQUIRE(verifier->score(instance, "Ada\nCleo").value() == 0.0);
  REQUIRE(verifier->score(instance, "Ada\nAda\nCleo\nFinn").value() == 0.0);

  const auto as_string = instance_with({{"answer", "Ada\nCleo\nFinn"}});
  REQUIRE(verifier->score(as_string, "  Cleo \nFinn\nAda\n").value() == 1.0);
}

TEST_CASE("subprocess verifier scores by exit code", "[eval][verifiers]") {
#ifdef _WIN32
  const std::vector<std::string> pass{"cmd", "/c", "exit 0"};
  const std::vector<std::string> fail{"cmd", "/c", "exit 3"};
#else
  const std::vector<std::string> pass{"sh", "-c", "exit 0"};
  const std::vector<std::string> fail{"sh", "-c", "exit 3"};
#endif

  eval::SubprocessOptions options;
  options.argv = pass;
  auto verifier = eval::make_subprocess_verifier(options);
  REQUIRE(verifier->trusted());
  REQUIRE(verifier->score(instance_with({}), "output").value() == 1.0);

  nlohmann::json expected;
  expected["argv"] = fail;
  REQUIRE(verifier->score(instance_with(expected), "output").value() == 0.0);

  eval::SubprocessOptions empty;
  auto broken = eval::make_subprocess_verifier(empty);
  const auto result = broken->score(instance_with({}), "output");
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("subprocess verifier substitutes the output placeholder", "[eval][verifiers]") {
#ifdef _WIN32
  const std::vector<std::string> check{"cmd", "/c", "if \"{output}\"==\"expected-answer\" "
                                      "(exit 0) else (exit 1)"};
#else
  const std::vector<std::string> check{"sh", "-c", "[ \"{output}\" = \"expected-answer\" ]"};
#endif
  eval::SubprocessOptions options;
  options.argv = check;
  auto verifier = eval::make_subprocess_verifier(options);
  REQUIRE(verifier->score(instance_with({}), "expected-answer").value() == 1.0);
  REQUIRE(verifier->score(instance_with({}), "wrong").value() == 0.0);
}

TEST_CASE("the LLM judge parses SCORE lines and is never trusted", "[eval][verifiers]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_handler([](const llm::ChatRequest& request) -> core::Result<llm::ChatResponse> {
    llm::ChatResponse response;
    REQUIRE(request.messages.front().content.size() == 1);
    const auto& text = std::get<llm::TextBlock>(request.messages.front().content.front()).text;
    REQUIRE(text.find("the-question") != std::string::npos);
    REQUIRE(text.find("the-answer") != std::string::npos);
    response.content.push_back(llm::TextBlock{"Solid answer.\nSCORE: 8"});
    return response;
  });

  auto judge = eval::make_llm_judge_verifier(backend);
  REQUIRE(!judge->trusted());

  eval::TaskInstance instance;
  instance.id = "case-1";
  instance.input = "the-question";
  const auto score = judge->score(instance, "the-answer");
  REQUIRE(score);
  REQUIRE(*score == 0.8);
}

TEST_CASE("a judge reply without a number is a Parse error", "[eval][verifiers]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_handler([](const llm::ChatRequest&) -> core::Result<llm::ChatResponse> {
    llm::ChatResponse response;
    response.content.push_back(llm::TextBlock{"I cannot grade this."});
    return response;
  });
  auto judge = eval::make_llm_judge_verifier(backend);
  const auto score = judge->score(instance_with({}), "answer");
  REQUIRE(!score);
  REQUIRE(score.error().code == core::ErrorCode::Parse);
}

TEST_CASE("judge scores are clamped into the rubric range", "[eval][verifiers]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_handler([](const llm::ChatRequest&) -> core::Result<llm::ChatResponse> {
    llm::ChatResponse response;
    response.content.push_back(llm::TextBlock{"SCORE: 15"});
    return response;
  });
  auto judge = eval::make_llm_judge_verifier(backend);
  REQUIRE(judge->score(instance_with({}), "answer").value() == 1.0);
}
