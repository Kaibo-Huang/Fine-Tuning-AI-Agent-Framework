#pragma once

#include <ostream>
#include <string>
#include <string_view>

#include "agents_framework/core/result.hpp"

namespace agents_framework::http {

class Secret {
 public:
  Secret() = default;
  explicit Secret(std::string value);

  [[nodiscard]] const std::string& reveal() const noexcept { return value_; }
  [[nodiscard]] bool empty() const noexcept { return value_.empty(); }

  friend std::ostream& operator<<(std::ostream& os, const Secret&) { return os << "***"; }

 private:
  std::string value_;
};

class SecretStore {
 public:
  static core::Result<Secret> from_env(std::string_view var = "ANTHROPIC_API_KEY");
};

}
