#include "agents_framework/graph/events.hpp"

#include <algorithm>

namespace agents_framework::graph {

std::size_t EventBus::subscribe(EventCallback callback) {
  const std::scoped_lock lock{mutex_};
  const std::size_t id = next_id_++;
  subscribers_.emplace_back(id, std::move(callback));
  return id;
}

void EventBus::unsubscribe(std::size_t id) {
  const std::scoped_lock lock{mutex_};
  std::erase_if(subscribers_, [id](const auto& entry) { return entry.first == id; });
}

void EventBus::publish(const ExecEvent& event) const {
  std::vector<EventCallback> callbacks;
  {
    const std::scoped_lock lock{mutex_};
    callbacks.reserve(subscribers_.size());
    for (const auto& [id, callback] : subscribers_) {
      callbacks.push_back(callback);
    }
  }
  for (const EventCallback& callback : callbacks) {
    if (callback) callback(event);
  }
}

bool EventBus::has_subscribers() const {
  const std::scoped_lock lock{mutex_};
  return !subscribers_.empty();
}

}
