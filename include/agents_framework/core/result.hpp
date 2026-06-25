#pragma once

#include <string_view>

namespace agents_framework::core {

enum class ErrorCode {
  Unknown,
  Invalid,
  NotFound,
  Config,
  Io,
  Network,
  Timeout,
  Auth,
  RateLimited,
  Parse,
  Protocol,
  Tool,
  Cancelled,
};

[[nodiscard]] inline std::string_view error_code_name(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Unknown:     return "Unknown";
    case ErrorCode::Invalid:     return "Invalid";
    case ErrorCode::NotFound:    return "NotFound";
    case ErrorCode::Config:      return "Config";
    case ErrorCode::Io:          return "Io";
    case ErrorCode::Network:     return "Network";
    case ErrorCode::Timeout:     return "Timeout";
    case ErrorCode::Auth:        return "Auth";
    case ErrorCode::RateLimited: return "RateLimited";
    case ErrorCode::Parse:       return "Parse";
    case ErrorCode::Protocol:    return "Protocol";
    case ErrorCode::Tool:        return "Tool";
    case ErrorCode::Cancelled:   return "Cancelled";
  }
  return "Unknown";
}

}
