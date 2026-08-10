#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/trace/trace.hpp"

namespace agents_framework::trace {

struct TraceQuery {
  std::string run_id;
  std::string suite;
  std::string task_id;
  std::optional<bool> verified;
  std::optional<double> min_score;
  std::size_t limit{0};
};

struct TraceInfo {
  std::string trace_id;
  std::string run_id;
  std::string suite;
  std::string task_id;
  std::optional<double> score;
  bool verified{false};
  std::string created_at;
};

class TraceStore {
 public:
  static core::Result<TraceStore> open(std::shared_ptr<store::Db> db);

  core::Result<void> save(const Trace& trace);
  core::Result<Trace> load(std::string_view trace_id);
  core::Result<std::vector<TraceInfo>> list(const TraceQuery& query = {});
  core::Result<std::vector<Trace>> select(const TraceQuery& query = {});

 private:
  explicit TraceStore(std::shared_ptr<store::Db> db) : db_(std::move(db)) {}

  std::shared_ptr<store::Db> db_;
};

}
