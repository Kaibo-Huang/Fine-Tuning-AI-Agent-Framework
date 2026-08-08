#include "agents_framework/trace/capture.hpp"

#include <cstdint>
#include <exception>
#include <random>
#include <stdexcept>
#include <utility>

#include "agents_framework/core/clock.hpp"
#include "agents_framework/core/id.hpp"
#include "agents_framework/core/rng.hpp"

namespace agents_framework::trace {

namespace {

std::string generate_trace_id() {
  core::SystemClock clock;
  std::random_device device;
  core::Rng rng{(static_cast<std::uint64_t>(device()) << 32) ^ device()};
  return core::Ulid::generate(clock, rng).to_string();
}

}

TraceCapture::TraceCapture(std::shared_ptr<graph::EventBus> events, CaptureOptions options)
    : events_(std::move(events)), options_(std::move(options)) {
  if (!events_) throw std::invalid_argument("TraceCapture requires a non-null event bus");
  subscription_ = events_->subscribe([this](const graph::ExecEvent& event) { on_event(event); });
}

TraceCapture::~TraceCapture() { events_->unsubscribe(subscription_); }

void TraceCapture::on_event(const graph::ExecEvent& event) {
  const std::scoped_lock lock{mutex_};
  if (const auto* started = std::get_if<graph::RunStarted>(&event)) {
    if (run_id_.empty() && (options_.run_id.empty() || options_.run_id == started->run_id)) {
      run_id_ = started->run_id;
    }
    return;
  }
  if (const auto* finished = std::get_if<graph::NodeFinished>(&event)) {
    if (finished->run_id == run_id_ && !run_id_.empty()) {
      node_runs_.push_back(NodeRun{finished->step, finished->node, finished->ok, finished->error});
    }
    return;
  }
  if (const auto* updated = std::get_if<graph::StateUpdated>(&event)) {
    if (updated->run_id == run_id_ && !run_id_.empty()) {
      last_state_ = updated->state;
    }
  }
}

core::Result<Trace> TraceCapture::take() const {
  const std::scoped_lock lock{mutex_};
  Trace trace;
  trace.trace_id = generate_trace_id();
  trace.run_id = run_id_;
  trace.node_runs = node_runs_;
  if (last_state_.is_object()) {
    const auto messages = last_state_.find(options_.messages_channel);
    if (messages != last_state_.end() && !messages->is_null()) {
      try {
        trace.transcript = messages->get<std::vector<llm::Message>>();
      } catch (const std::exception& error) {
        return core::fail(core::ErrorCode::Parse, error.what(),
                          "channel '" + options_.messages_channel + "'");
      }
    }
  }
  trace.final_output = final_assistant_text(trace.transcript);
  return trace;
}

}
