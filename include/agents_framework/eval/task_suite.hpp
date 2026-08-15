#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"

namespace agents_framework::eval {

enum class Split { Train, HeldOut, Retention };

[[nodiscard]] inline std::string_view split_name(Split split) noexcept {
  switch (split) {
    case Split::Train:     return "train";
    case Split::HeldOut:   return "held_out";
    case Split::Retention: return "retention";
  }
  return "train";
}

[[nodiscard]] inline Split split_from_string(std::string_view text) noexcept {
  if (text == "held_out") return Split::HeldOut;
  if (text == "retention") return Split::Retention;
  return Split::Train;
}

struct TaskInstance {
  std::string id;
  Split split{Split::Train};
  nlohmann::json input = nlohmann::json::object();
  nlohmann::json expected = nlohmann::json::object();

  bool operator==(const TaskInstance&) const = default;
};

class Verifier {
 public:
  virtual ~Verifier() = default;

  virtual core::Result<double> score(const TaskInstance& instance, std::string_view output) = 0;

  [[nodiscard]] virtual bool trusted() const { return true; }
};

class TaskSuite {
 public:
  virtual ~TaskSuite() = default;

  [[nodiscard]] virtual std::string_view name() const = 0;
  [[nodiscard]] virtual std::span<const TaskInstance> instances() const = 0;
  [[nodiscard]] virtual Verifier& verifier() = 0;
};

[[nodiscard]] core::Result<void> validate_suite(std::string_view name,
                                                std::span<const TaskInstance> instances);

class ListTaskSuite final : public TaskSuite {
 public:
  static core::Result<ListTaskSuite> create(std::string name,
                                            std::vector<TaskInstance> instances,
                                            std::unique_ptr<Verifier> verifier);

  [[nodiscard]] std::string_view name() const override { return name_; }
  [[nodiscard]] std::span<const TaskInstance> instances() const override { return instances_; }
  [[nodiscard]] Verifier& verifier() override { return *verifier_; }

 private:
  ListTaskSuite(std::string name, std::vector<TaskInstance> instances,
                std::unique_ptr<Verifier> verifier)
      : name_(std::move(name)), instances_(std::move(instances)), verifier_(std::move(verifier)) {}

  std::string name_;
  std::vector<TaskInstance> instances_;
  std::unique_ptr<Verifier> verifier_;
};

}
