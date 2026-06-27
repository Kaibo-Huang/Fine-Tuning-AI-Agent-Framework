#include "agents_framework/core/logging.hpp"

#include <mutex>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace agents_framework::core {
namespace {

std::mutex& redaction_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::string>& redaction_secrets() {
  static std::vector<std::string> secrets;
  return secrets;
}

bool g_json_output = false;

spdlog::level::level_enum to_spdlog(LogLevel level) {
  switch (level) {
    case LogLevel::Trace:    return spdlog::level::trace;
    case LogLevel::Debug:    return spdlog::level::debug;
    case LogLevel::Info:     return spdlog::level::info;
    case LogLevel::Warn:     return spdlog::level::warn;
    case LogLevel::Error:    return spdlog::level::err;
    case LogLevel::Critical: return spdlog::level::critical;
    case LogLevel::Off:      return spdlog::level::off;
  }
  return spdlog::level::info;
}

std::string_view level_name(LogLevel level) {
  switch (level) {
    case LogLevel::Trace:    return "trace";
    case LogLevel::Debug:    return "debug";
    case LogLevel::Info:     return "info";
    case LogLevel::Warn:     return "warn";
    case LogLevel::Error:    return "error";
    case LogLevel::Critical: return "critical";
    case LogLevel::Off:      return "off";
  }
  return "info";
}

std::string format_text(std::string_view event, std::initializer_list<LogField> fields) {
  std::string out{event};
  for (const auto& [key, value] : fields) {
    out += ' ';
    out.append(key);
    out += '=';
    out += value;
  }
  return out;
}

std::string format_json(LogLevel level, std::string_view event,
                        std::initializer_list<LogField> fields) {
  nlohmann::json object;
  object["level"] = level_name(level);
  object["event"] = event;
  for (const auto& [key, value] : fields) {
    object[std::string{key}] = value;
  }
  return object.dump();
}

}  

void init_logging(LogLevel level, bool json) {
  g_json_output = json;
  spdlog::set_level(to_spdlog(level));
  if (json) {
    spdlog::set_pattern("%v");
  } else {
    spdlog::set_pattern("%Y-%m-%dT%H:%M:%S.%e %^%l%$ %v");
  }
}

void log(LogLevel level, std::string_view event, std::initializer_list<LogField> fields) {
  std::string message = g_json_output ? format_json(level, event, fields)
                                      : format_text(event, fields);
  message = redact(message);
  spdlog::log(to_spdlog(level), "{}", message);
}

void add_redaction(std::string secret_value) {
  if (secret_value.empty()) {
    return;
  }
  std::lock_guard lock{redaction_mutex()};
  redaction_secrets().push_back(std::move(secret_value));
}

std::string redact(std::string_view text) {
  std::string out{text};
  std::lock_guard lock{redaction_mutex()};
  for (const auto& secret : redaction_secrets()) {
    std::string::size_type pos = 0;
    while ((pos = out.find(secret, pos)) != std::string::npos) {
      out.replace(pos, secret.size(), "***");
      pos += 3;
    }
  }
  return out;
}

}  
