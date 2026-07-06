#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace agents_framework::http {

struct SseEvent {
  std::string event;
  std::string data;
};

class SseParser {
 public:
  void feed(std::string_view bytes, const std::function<void(const SseEvent&)>& on_event);

 private:
  void handle_line(std::string_view line, const std::function<void(const SseEvent&)>& on_event);
  void dispatch(const std::function<void(const SseEvent&)>& on_event);

  std::string buffer_;
  std::string event_type_;
  std::string data_;
};

}
