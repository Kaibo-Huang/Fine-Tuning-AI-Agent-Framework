#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"

namespace agents_framework::graph {

enum class CheckpointStatus { Running, Interrupted, Completed };

[[nodiscard]] inline std::string_view checkpoint_status_name(CheckpointStatus status) noexcept {
  switch (status) {
    case CheckpointStatus::Running:     return "running";
    case CheckpointStatus::Interrupted: return "interrupted";
    case CheckpointStatus::Completed:   return "completed";
  }
  return "running";
}

[[nodiscard]] inline CheckpointStatus checkpoint_status_from_string(std::string_view text) noexcept {
  if (text == "interrupted") return CheckpointStatus::Interrupted;
  if (text == "completed") return CheckpointStatus::Completed;
  return CheckpointStatus::Running;
}

struct Checkpoint {
  std::string run_id;
  std::uint64_t step{0};
  std::string state;
  std::vector<std::string> next_nodes;
  CheckpointStatus status{CheckpointStatus::Running};

  bool operator==(const Checkpoint&) const = default;
};

class Checkpointer {
 public:
  virtual ~Checkpointer() = default;

  virtual core::Result<void> save(const Checkpoint& checkpoint) = 0;
};

}
