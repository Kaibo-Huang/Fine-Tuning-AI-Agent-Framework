#include "agents_framework/http/sse.hpp"

namespace agents_framework::http {
namespace {

std::string_view strip_cr(std::string_view line) {
  if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
  return line;
}

}

void SseParser::feed(std::string_view bytes,
                     const std::function<void(const SseEvent&)>& on_event) {
  buffer_.append(bytes);
  std::size_t start = 0;
  for (std::size_t nl = buffer_.find('\n', start); nl != std::string::npos;
       nl = buffer_.find('\n', start)) {
    handle_line(strip_cr(std::string_view{buffer_}.substr(start, nl - start)), on_event);
    start = nl + 1;
  }
  buffer_.erase(0, start);
}

void SseParser::handle_line(std::string_view line,
                            const std::function<void(const SseEvent&)>& on_event) {
  if (line.empty()) {
    dispatch(on_event);
    return;
  }
  if (line.front() == ':') {
    return;
  }

  std::string_view field = line;
  std::string_view value;
  if (const auto colon = line.find(':'); colon != std::string_view::npos) {
    field = line.substr(0, colon);
    value = line.substr(colon + 1);
    if (!value.empty() && value.front() == ' ') value.remove_prefix(1);
  }

  if (field == "event") {
    event_type_.assign(value);
  } else if (field == "data") {
    if (!data_.empty()) data_ += '\n';
    data_.append(value);
  }
}

void SseParser::dispatch(const std::function<void(const SseEvent&)>& on_event) {
  if (!data_.empty()) {
    on_event(SseEvent{event_type_, data_});
  }
  event_type_.clear();
  data_.clear();
}

}
