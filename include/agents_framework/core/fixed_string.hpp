#pragma once

#include <algorithm>
#include <cstddef>
#include <string_view>

namespace agents_framework::core {

template <std::size_t N>
struct FixedString {
  char value[N]{};

  constexpr FixedString(const char (&literal)[N]) { std::copy_n(literal, N, value); }

  [[nodiscard]] constexpr std::string_view view() const noexcept {
    return std::string_view{value, N - 1};
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return N - 1; }

  template <std::size_t M>
  [[nodiscard]] constexpr bool operator==(const FixedString<M>& other) const noexcept {
    return view() == other.view();
  }
};

}  
