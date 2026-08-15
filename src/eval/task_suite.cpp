#include "agents_framework/eval/task_suite.hpp"

#include <algorithm>
#include <utility>

namespace agents_framework::eval {

core::Result<void> validate_suite(std::string_view name,
                                  std::span<const TaskInstance> instances) {
  if (name.empty()) {
    return core::fail(core::ErrorCode::Invalid, "a task suite needs a non-empty name");
  }
  std::vector<std::string_view> ids;
  ids.reserve(instances.size());
  for (const TaskInstance& instance : instances) {
    if (instance.id.empty()) {
      return core::fail(core::ErrorCode::Invalid, "every task instance needs a non-empty id",
                        std::string{name});
    }
    ids.push_back(instance.id);
  }
  std::sort(ids.begin(), ids.end());
  const auto duplicate = std::adjacent_find(ids.begin(), ids.end());
  if (duplicate != ids.end()) {
    return core::fail(core::ErrorCode::Invalid, "task instance ids must be unique",
                      std::string{*duplicate});
  }
  return {};
}

core::Result<ListTaskSuite> ListTaskSuite::create(std::string name,
                                                  std::vector<TaskInstance> instances,
                                                  std::unique_ptr<Verifier> verifier) {
  AF_TRY_VOID(validate_suite(name, instances));
  if (!verifier) {
    return core::fail(core::ErrorCode::Invalid, "a task suite needs a verifier", name);
  }
  return ListTaskSuite(std::move(name), std::move(instances), std::move(verifier));
}

}
