#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/task_suite.hpp"
#include "agents_framework/llm/backend.hpp"

namespace agents_framework::eval {

struct ExactMatchOptions {
  bool case_insensitive{false};
  bool collapse_whitespace{true};
  std::string expected_key{"answer"};
};
[[nodiscard]] std::unique_ptr<Verifier> make_exact_match_verifier(ExactMatchOptions options = {});

struct NumericOptions {
  double abs_tolerance{1e-6};
  double rel_tolerance{0.0};
  std::string expected_key{"answer"};
};
[[nodiscard]] std::unique_ptr<Verifier> make_numeric_verifier(NumericOptions options = {});

struct SetMatchOptions {
  bool case_insensitive{false};
  std::string expected_key{"answer"};
};
[[nodiscard]] std::unique_ptr<Verifier> make_set_match_verifier(SetMatchOptions options = {});

struct SubprocessOptions {
  std::vector<std::string> argv;
  std::string argv_key{"argv"};
  bool output_on_stdin{true};
  std::chrono::milliseconds timeout{30000};
};
[[nodiscard]] std::unique_ptr<Verifier> make_subprocess_verifier(SubprocessOptions options);

struct LlmJudgeOptions {
  std::string model;
  std::string prompt_template;
  double max_points{10.0};
  llm::SamplingParams sampling{.temperature = 0.0};
};
[[nodiscard]] std::unique_ptr<Verifier> make_llm_judge_verifier(
    std::shared_ptr<llm::LLMBackend> backend, LlmJudgeOptions options = {});

}
