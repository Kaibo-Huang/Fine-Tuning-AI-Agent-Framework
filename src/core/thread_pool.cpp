#include "agents_framework/core/thread_pool.hpp"

#include <algorithm>
#include <utility>

namespace agents_framework::core {

ThreadPool::ThreadPool(std::size_t workers) {
  const std::size_t count = std::max<std::size_t>(1, workers);
  workers_.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::scoped_lock lock(mutex_);
    stopping_ = true;
  }
  ready_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::submit(std::function<void()> task) {
  {
    std::scoped_lock lock(mutex_);
    queue_.push_back(std::move(task));
  }
  ready_.notify_one();
}

void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) return;
      task = std::move(queue_.front());
      queue_.pop_front();
    }
    task();
  }
}

}
