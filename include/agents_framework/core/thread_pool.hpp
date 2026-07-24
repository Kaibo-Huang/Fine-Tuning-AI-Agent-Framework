#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace agents_framework::core {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t workers);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void submit(std::function<void()> task);

  [[nodiscard]] std::size_t worker_count() const noexcept { return workers_.size(); }

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stopping_{false};
};

}
