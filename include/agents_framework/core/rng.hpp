#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>

namespace agents_framework::core {

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_{seed} {}

  std::uint64_t next_u64() { return engine_(); }

  double next_double() {
    return static_cast<double>(engine_() >> 11) * (1.0 / 9007199254740992.0);
  }

  void fill(std::span<std::byte> out) {
    std::size_t offset = 0;
    while (offset < out.size()) {
      const std::uint64_t value = engine_();
      const std::size_t chunk = std::min<std::size_t>(sizeof(value), out.size() - offset);
      std::memcpy(out.data() + offset, &value, chunk);
      offset += chunk;
    }
  }

 private:
  std::mt19937_64 engine_;
};

}  // namespace agents_framework::core
