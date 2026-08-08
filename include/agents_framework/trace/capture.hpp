#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/events.hpp"
#include "agents_framework/trace/trace.hpp"

namespace agents_framework::trace {

struct CaptureOptions {
  std::string messages_channel{"messages"};
  std::string run_id;
};

class TraceCapture {
 public:
  explicit TraceCapture(std::shared_ptr<graph::EventBus> events, CaptureOptions options = {});
  ~TraceCapture();

  TraceCapture(const TraceCapture&) = delete;
  TraceCapture& operator=(const TraceCapture&) = delete;

  [[nodiscard]] core::Result<Trace> take() const;

 private:
  void on_event(const graph::ExecEvent& event);

  std::shared_ptr<graph::EventBus> events_;
  CaptureOptions options_;
  std::size_t subscription_{0};

  mutable std::mutex mutex_;
  std::string run_id_;
  std::vector<NodeRun> node_runs_;
  nlohmann::json last_state_;
};

}
