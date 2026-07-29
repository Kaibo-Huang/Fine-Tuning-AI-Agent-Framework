#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/checkpoint.hpp"
#include "agents_framework/store/db.hpp"

namespace agents_framework::store {

struct CheckpointInfo {
  std::uint64_t step{0};
  graph::CheckpointStatus status{graph::CheckpointStatus::Running};
  std::string created_at;
};

struct RunInfo {
  std::string run_id;
  std::optional<std::string> parent_run_id;
  std::uint64_t latest_step{0};
  graph::CheckpointStatus status{graph::CheckpointStatus::Running};
};

class CheckpointStore final : public graph::Checkpointer {
 public:
  static core::Result<CheckpointStore> open(std::shared_ptr<Db> db);

  core::Result<void> save(const graph::Checkpoint& checkpoint) override;

  core::Result<std::vector<CheckpointInfo>> list(std::string_view run_id);
  core::Result<graph::Checkpoint> load(std::string_view run_id, std::uint64_t step);
  core::Result<graph::Checkpoint> latest(std::string_view run_id);

  core::Result<graph::Checkpoint> fork(std::string_view run_id, std::uint64_t step,
                                       std::string new_run_id);

  core::Result<std::vector<RunInfo>> runs();

 private:
  explicit CheckpointStore(std::shared_ptr<Db> db) : db_(std::move(db)) {}

  std::shared_ptr<Db> db_;
};

}
