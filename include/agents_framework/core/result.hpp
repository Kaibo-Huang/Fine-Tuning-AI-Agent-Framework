#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <utility>

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

struct Error {
  ErrorCode code{ErrorCode::Unknown};
  std::string message;
  std::string context;

  [[nodiscard]] std::string to_string() const {
    std::string out{error_code_name(code)};
    if (!message.empty()) {
      out += ": ";
      out += message;
    }
    if (!context.empty()) {
      out += " (";
      out += context;
      out += ')';
    }
    return out;
  }
};

template <class T>
using Result = std::expected<T, Error>;

[[nodiscard]] inline std::unexpected<Error> fail(ErrorCode code, std::string message,
                                                 std::string context = {}) {
  return std::unexpected(Error{code, std::move(message), std::move(context)});
}

}

#define AF_DETAIL_CONCAT_(a, b) a##b
#define AF_DETAIL_CONCAT(a, b) AF_DETAIL_CONCAT_(a, b)

#define AF_TRY(decl, expr)                                          \
  auto&& AF_DETAIL_CONCAT(af_try_, __LINE__) = (expr);              \
  if (!AF_DETAIL_CONCAT(af_try_, __LINE__)) {                       \
    return std::unexpected(                                         \
        std::move(AF_DETAIL_CONCAT(af_try_, __LINE__)).error());    \
  }                                                                 \
  decl = *std::move(AF_DETAIL_CONCAT(af_try_, __LINE__))

#define AF_TRY_VOID(expr)                                           \
  if (auto&& AF_DETAIL_CONCAT(af_try_, __LINE__) = (expr);          \
      !AF_DETAIL_CONCAT(af_try_, __LINE__)) {                       \
    return std::unexpected(                                         \
        std::move(AF_DETAIL_CONCAT(af_try_, __LINE__)).error());    \
  }
