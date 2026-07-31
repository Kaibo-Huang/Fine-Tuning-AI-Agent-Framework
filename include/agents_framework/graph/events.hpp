#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace agents_framework::graph {

struct RunStarted {
  std::string run_id;
  std::uint64_t step{0};
};

struct StepStarted {
  std::string run_id;
  std::uint64_t step{0};
  std::vector<std::string> nodes;
};

struct NodeStarted {
  std::string run_id;
  std::uint64_t step{0};
  std::string node;
};

struct NodeFinished {
  std::string run_id;
  std::uint64_t step{0};
  std::string node;
  bool ok{true};
  std::string error;
};

struct StateUpdated {
  std::string run_id;
  std::uint64_t step{0};
  nlohmann::json state;
};

struct TokenDelta {
  std::string node;
  std::string text;
};

struct RunInterrupted {
  std::string run_id;
  std::uint64_t step{0};
  std::vector<std::string> pending_nodes;
};

struct RunFinished {
  std::string run_id;
  std::uint64_t step{0};
  bool ok{true};
  std::string error;
};

using ExecEvent = std::variant<RunStarted, StepStarted, NodeStarted, NodeFinished, StateUpdated,
                               TokenDelta, RunInterrupted, RunFinished>;

using EventCallback = std::function<void(const ExecEvent&)>;

class EventBus {
 public:
  std::size_t subscribe(EventCallback callback);
  void unsubscribe(std::size_t id);

  void publish(const ExecEvent& event) const;

  [[nodiscard]] bool has_subscribers() const;

 private:
  mutable std::mutex mutex_;
  std::vector<std::pair<std::size_t, EventCallback>> subscribers_;
  std::size_t next_id_{1};
};

}
