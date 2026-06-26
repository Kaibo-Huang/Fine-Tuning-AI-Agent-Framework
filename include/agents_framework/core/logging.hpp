#pragma once

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

namespace agents_framework::core {

enum class LogLevel {
  Trace,
  Debug,
  Info,
  Warn,
  Error,
  Critical,
  Off,
};

using LogField = std::pair<std::string_view, std::string>;

void init_logging(LogLevel level = LogLevel::Info, bool json = false);

void log(LogLevel level, std::string_view event,
         std::initializer_list<LogField> fields = {});

void add_redaction(std::string secret_value);

[[nodiscard]] std::string redact(std::string_view text);

}  
