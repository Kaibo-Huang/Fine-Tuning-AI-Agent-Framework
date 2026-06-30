#pragma once

#include <chrono>

namespace agents_framework::core {

using Timestamp = std::chrono::system_clock::time_point;

class Clock {
 public:
  virtual ~Clock() = default;
  [[nodiscard]] virtual Timestamp now() const = 0;
};

class SystemClock : public Clock {
 public:
  [[nodiscard]] Timestamp now() const override { return std::chrono::system_clock::now(); }
};

class ManualClock : public Clock {
 public:
  explicit ManualClock(Timestamp start = {}) : current_{start} {}

  [[nodiscard]] Timestamp now() const override { return current_; }

  void advance(std::chrono::milliseconds delta) { current_ += delta; }
  void set(Timestamp value) { current_ = value; }

 private:
  Timestamp current_;
};

}  // namespace agents_framework::core
