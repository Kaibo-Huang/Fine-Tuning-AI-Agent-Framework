#include "agents_framework/eval/verifiers.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>

#include "agents_framework/tools/process.hpp"

namespace agents_framework::eval {

namespace {

[[nodiscard]] std::string normalize(std::string_view text, bool case_insensitive,
                                    bool collapse_whitespace) {
  std::string out;
  out.reserve(text.size());
  bool pending_space = false;
  for (const char c : text) {
    const bool space = std::isspace(static_cast<unsigned char>(c)) != 0;
    if (collapse_whitespace) {
      if (space) {
        pending_space = !out.empty();
        continue;
      }
      if (pending_space) {
        out += ' ';
        pending_space = false;
      }
    }
    out += case_insensitive ? static_cast<char>(std::tolower(static_cast<unsigned char>(c))) : c;
  }
  return out;
}

[[nodiscard]] core::Result<std::string> expected_string(const TaskInstance& instance,
                                                        const std::string& key) {
  const auto it = instance.expected.find(key);
  if (it == instance.expected.end()) {
    return core::fail(core::ErrorCode::Invalid, "instance is missing expected data",
                      "'" + instance.id + "' needs expected." + key);
  }
  if (it->is_string()) return it->get<std::string>();
  return it->dump();
}

[[nodiscard]] std::optional<double> first_number(std::string_view text) {
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    const bool digit = std::isdigit(static_cast<unsigned char>(c)) != 0;
    const bool sign_start =
        (c == '-' || c == '+' || c == '.') && i + 1 < text.size() &&
        std::isdigit(static_cast<unsigned char>(text[i + 1])) != 0;
    if (!digit && !sign_start) continue;
    const std::string candidate{text.substr(i)};
    char* end = nullptr;
    const double value = std::strtod(candidate.c_str(), &end);
    if (end != candidate.c_str()) return value;
  }
  return std::nullopt;
}

class ExactMatchVerifier final : public Verifier {
 public:
  explicit ExactMatchVerifier(ExactMatchOptions options) : options_(std::move(options)) {}

  core::Result<double> score(const TaskInstance& instance, std::string_view output) override {
    AF_TRY(const auto expected, expected_string(instance, options_.expected_key));
    const std::string want =
        normalize(expected, options_.case_insensitive, options_.collapse_whitespace);
    const std::string got =
        normalize(output, options_.case_insensitive, options_.collapse_whitespace);
    return want == got ? 1.0 : 0.0;
  }

 private:
  ExactMatchOptions options_;
};

class NumericVerifier final : public Verifier {
 public:
  explicit NumericVerifier(NumericOptions options) : options_(std::move(options)) {}

  core::Result<double> score(const TaskInstance& instance, std::string_view output) override {
    const auto it = instance.expected.find(options_.expected_key);
    if (it == instance.expected.end()) {
      return core::fail(core::ErrorCode::Invalid, "instance is missing expected data",
                        "'" + instance.id + "' needs expected." + options_.expected_key);
    }
    double want = 0.0;
    if (it->is_number()) {
      want = it->get<double>();
    } else if (it->is_string()) {
      const auto parsed = first_number(it->get<std::string>());
      if (!parsed) {
        return core::fail(core::ErrorCode::Invalid, "expected value is not numeric",
                          "'" + instance.id + "'");
      }
      want = *parsed;
    } else {
      return core::fail(core::ErrorCode::Invalid, "expected value is not numeric",
                        "'" + instance.id + "'");
    }

    const auto got = first_number(output);
    if (!got) return 0.0;
    const double tolerance =
        options_.abs_tolerance + options_.rel_tolerance * std::abs(want);
    return std::abs(*got - want) <= tolerance ? 1.0 : 0.0;
  }

 private:
  NumericOptions options_;
};

[[nodiscard]] std::vector<std::string> lines_of(std::string_view text, bool case_insensitive) {
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (start <= text.size()) {
    std::size_t end = text.find('\n', start);
    if (end == std::string_view::npos) end = text.size();
    std::string line = normalize(text.substr(start, end - start), case_insensitive, true);
    if (!line.empty()) lines.push_back(std::move(line));
    start = end + 1;
  }
  std::sort(lines.begin(), lines.end());
  return lines;
}

class SetMatchVerifier final : public Verifier {
 public:
  explicit SetMatchVerifier(SetMatchOptions options) : options_(std::move(options)) {}

  core::Result<double> score(const TaskInstance& instance, std::string_view output) override {
    const auto it = instance.expected.find(options_.expected_key);
    if (it == instance.expected.end()) {
      return core::fail(core::ErrorCode::Invalid, "instance is missing expected data",
                        "'" + instance.id + "' needs expected." + options_.expected_key);
    }
    std::vector<std::string> want;
    if (it->is_array()) {
      for (const auto& element : *it) {
        std::string line = normalize(element.is_string() ? element.get<std::string>()
                                                         : element.dump(),
                                     options_.case_insensitive, true);
        if (!line.empty()) want.push_back(std::move(line));
      }
      std::sort(want.begin(), want.end());
    } else if (it->is_string()) {
      want = lines_of(it->get<std::string>(), options_.case_insensitive);
    } else {
      return core::fail(core::ErrorCode::Invalid,
                        "expected value must be an array of strings or a string",
                        "'" + instance.id + "'");
    }

    const std::vector<std::string> got = lines_of(output, options_.case_insensitive);
    return want == got ? 1.0 : 0.0;
  }

 private:
  SetMatchOptions options_;
};

[[nodiscard]] std::string replace_all(std::string text, std::string_view placeholder,
                                      std::string_view value) {
  std::size_t pos = 0;
  while ((pos = text.find(placeholder, pos)) != std::string::npos) {
    text.replace(pos, placeholder.size(), value);
    pos += value.size();
  }
  return text;
}

class SubprocessVerifier final : public Verifier {
 public:
  explicit SubprocessVerifier(SubprocessOptions options) : options_(std::move(options)) {}

  core::Result<double> score(const TaskInstance& instance, std::string_view output) override {
    std::vector<std::string> argv = options_.argv;
    const auto override_argv = instance.expected.find(options_.argv_key);
    if (override_argv != instance.expected.end()) {
      if (!override_argv->is_array()) {
        return core::fail(core::ErrorCode::Invalid, "expected." + options_.argv_key +
                                                        " must be an array of strings",
                          "'" + instance.id + "'");
      }
      argv.clear();
      for (const auto& arg : *override_argv) {
        if (!arg.is_string()) {
          return core::fail(core::ErrorCode::Invalid, "expected." + options_.argv_key +
                                                          " must be an array of strings",
                            "'" + instance.id + "'");
        }
        argv.push_back(arg.get<std::string>());
      }
    }
    if (argv.empty()) {
      return core::fail(core::ErrorCode::Invalid, "subprocess verifier has no command to run",
                        "'" + instance.id + "'");
    }
    for (std::string& arg : argv) {
      arg = replace_all(std::move(arg), "{output}", output);
    }

    tools::ProcessOptions process;
    process.timeout = options_.timeout;
    if (options_.output_on_stdin) process.stdin_data = std::string{output};
    AF_TRY(const auto result, tools::run_process(argv, process));
    return result.exit_code == 0 ? 1.0 : 0.0;
  }

 private:
  SubprocessOptions options_;
};

constexpr std::string_view kDefaultJudgeTemplate =
    "You are grading an answer to a task.\n\n"
    "Task input:\n{input}\n\n"
    "Reference (may be empty):\n{expected}\n\n"
    "Answer to grade:\n{output}\n\n"
    "Grade the answer for correctness and completeness. Reply with a short justification, "
    "then a final line of exactly: SCORE: <number from 0 to 10>";

[[nodiscard]] std::string render_field(const nlohmann::json& value) {
  return value.is_string() ? value.get<std::string>() : value.dump();
}

class LlmJudgeVerifier final : public Verifier {
 public:
  LlmJudgeVerifier(std::shared_ptr<llm::LLMBackend> backend, LlmJudgeOptions options)
      : backend_(std::move(backend)), options_(std::move(options)) {
    if (!backend_) throw std::invalid_argument("LLM judge requires a non-null backend");
    if (options_.prompt_template.empty()) {
      options_.prompt_template = std::string{kDefaultJudgeTemplate};
    }
  }

  core::Result<double> score(const TaskInstance& instance, std::string_view output) override {
    std::string prompt = options_.prompt_template;
    prompt = replace_all(std::move(prompt), "{input}", render_field(instance.input));
    prompt = replace_all(std::move(prompt), "{expected}", render_field(instance.expected));
    prompt = replace_all(std::move(prompt), "{output}", output);

    llm::ChatRequest request;
    request.model = options_.model;
    request.sampling = options_.sampling;
    request.messages.push_back(llm::Message::user_text(std::move(prompt)));
    AF_TRY(const auto response, backend_->generate(request));

    const std::string text = response.text();
    std::optional<double> points;
    if (const std::size_t marker = text.rfind("SCORE:"); marker != std::string::npos) {
      points = first_number(std::string_view{text}.substr(marker + 6));
    }
    if (!points) points = first_number(text);
    if (!points) {
      return core::fail(core::ErrorCode::Parse, "judge response contains no score",
                        "'" + instance.id + "'");
    }
    const double clamped = std::clamp(*points, 0.0, options_.max_points);
    return options_.max_points > 0.0 ? clamped / options_.max_points : 0.0;
  }

  [[nodiscard]] bool trusted() const override { return false; }

 private:
  std::shared_ptr<llm::LLMBackend> backend_;
  LlmJudgeOptions options_;
};

}

std::unique_ptr<Verifier> make_exact_match_verifier(ExactMatchOptions options) {
  return std::make_unique<ExactMatchVerifier>(std::move(options));
}

std::unique_ptr<Verifier> make_numeric_verifier(NumericOptions options) {
  return std::make_unique<NumericVerifier>(std::move(options));
}

std::unique_ptr<Verifier> make_set_match_verifier(SetMatchOptions options) {
  return std::make_unique<SetMatchVerifier>(std::move(options));
}

std::unique_ptr<Verifier> make_subprocess_verifier(SubprocessOptions options) {
  return std::make_unique<SubprocessVerifier>(std::move(options));
}

std::unique_ptr<Verifier> make_llm_judge_verifier(std::shared_ptr<llm::LLMBackend> backend,
                                                  LlmJudgeOptions options) {
  return std::make_unique<LlmJudgeVerifier>(std::move(backend), std::move(options));
}

}
