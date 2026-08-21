#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/harness.hpp"
#include "agents_framework/store/db.hpp"

namespace agents_framework::eval {

struct EvalRunInfo {
  std::string eval_id;
  std::string suite;
  std::string model;
  std::string adapter;
  std::string label;
  std::string split;
  std::uint64_t seed{0};
  std::size_t total{0};
  double mean_score{0.0};
  Interval ci;
  bool baseline{false};
  std::string created_at;
};

class EvalStore {
 public:
  static core::Result<EvalStore> open(std::shared_ptr<store::Db> db);

  core::Result<void> save(const EvalReport& report);
  core::Result<EvalReport> load(std::string_view eval_id);
  core::Result<std::vector<EvalRunInfo>> list(std::string_view suite = {});

  core::Result<void> set_baseline(std::string_view eval_id);
  core::Result<EvalReport> baseline(std::string_view suite);

 private:
  explicit EvalStore(std::shared_ptr<store::Db> db) : db_(std::move(db)) {}

  std::shared_ptr<store::Db> db_;
};

}
