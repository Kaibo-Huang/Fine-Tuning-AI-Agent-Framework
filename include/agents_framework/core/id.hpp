#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <string>
#include <string_view>

#include "agents_framework/core/result.hpp"

namespace agents_framework::core {

class Clock;
class Rng;

class Ulid {
 public:
  Ulid() = default;
  explicit Ulid(std::array<std::byte, 16> bytes) : bytes_{bytes} {}

  static Ulid generate(const Clock& clock, Rng& rng);
  static Result<Ulid> parse(std::string_view text);

  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] const std::array<std::byte, 16>& bytes() const noexcept { return bytes_; }

  bool operator==(const Ulid&) const = default;
  auto operator<=>(const Ulid&) const = default;

 private:
  std::array<std::byte, 16> bytes_{};
};

}  // namespace agents_framework::core
