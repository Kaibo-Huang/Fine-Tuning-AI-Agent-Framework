#include "agents_framework/core/id.hpp"

#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <span>

#include "agents_framework/core/clock.hpp"
#include "agents_framework/core/rng.hpp"

namespace agents_framework::core {
namespace {

constexpr char kAlphabet[] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
constexpr std::size_t kStringLength = 26;
constexpr std::size_t kTimestampChars = 10;

const std::array<std::int8_t, 256>& decode_table() {
  static const std::array<std::int8_t, 256> table = [] {
    std::array<std::int8_t, 256> t;
    t.fill(-1);
    for (std::int8_t i = 0; i < 32; ++i) {
      const auto upper = static_cast<unsigned char>(kAlphabet[i]);
      t[upper] = i;
      t[static_cast<unsigned char>(std::tolower(upper))] = i;
    }
    t[static_cast<unsigned char>('I')] = 1;
    t[static_cast<unsigned char>('i')] = 1;
    t[static_cast<unsigned char>('L')] = 1;
    t[static_cast<unsigned char>('l')] = 1;
    t[static_cast<unsigned char>('O')] = 0;
    t[static_cast<unsigned char>('o')] = 0;
    return t;
  }();
  return table;
}

}  // namespace

Ulid Ulid::generate(const Clock& clock, Rng& rng) {
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      clock.now().time_since_epoch())
                      .count();
  const auto timestamp = static_cast<std::uint64_t>(ms) & ((std::uint64_t{1} << 48) - 1);

  std::array<std::byte, 16> bytes{};
  for (int i = 0; i < 6; ++i) {
    bytes[i] = static_cast<std::byte>((timestamp >> (8 * (5 - i))) & 0xFF);
  }
  rng.fill(std::span<std::byte>{bytes}.subspan(6, 10));
  return Ulid{bytes};
}

std::string Ulid::to_string() const {
  std::string out(kStringLength, '0');

  std::uint64_t timestamp = 0;
  for (int i = 0; i < 6; ++i) {
    timestamp = (timestamp << 8) | static_cast<std::uint8_t>(bytes_[i]);
  }
  for (std::size_t i = 0; i < kTimestampChars; ++i) {
    const unsigned shift = 5 * static_cast<unsigned>(kTimestampChars - 1 - i);
    out[i] = kAlphabet[(timestamp >> shift) & 0x1F];
  }

  std::uint32_t buffer = 0;
  int bit_count = 0;
  std::size_t out_index = kTimestampChars;
  for (int i = 6; i < 16; ++i) {
    buffer = (buffer << 8) | static_cast<std::uint8_t>(bytes_[i]);
    bit_count += 8;
    while (bit_count >= 5) {
      bit_count -= 5;
      out[out_index++] = kAlphabet[(buffer >> bit_count) & 0x1F];
    }
  }
  return out;
}

Result<Ulid> Ulid::parse(std::string_view text) {
  if (text.size() != kStringLength) {
    return fail(ErrorCode::Parse, "ULID must be 26 characters", std::string{text});
  }

  std::array<std::int8_t, kStringLength> values{};
  for (std::size_t i = 0; i < kStringLength; ++i) {
    const std::int8_t value = decode_table()[static_cast<unsigned char>(text[i])];
    if (value < 0) {
      return fail(ErrorCode::Parse, "invalid ULID character", std::string{text});
    }
    values[i] = value;
  }

  std::uint64_t timestamp = 0;
  for (std::size_t i = 0; i < kTimestampChars; ++i) {
    timestamp = (timestamp << 5) | static_cast<std::uint64_t>(values[i]);
  }
  if (timestamp >= (std::uint64_t{1} << 48)) {
    return fail(ErrorCode::Parse, "ULID timestamp overflow", std::string{text});
  }

  std::array<std::byte, 16> bytes{};
  for (int i = 0; i < 6; ++i) {
    bytes[i] = static_cast<std::byte>((timestamp >> (8 * (5 - i))) & 0xFF);
  }

  std::uint32_t buffer = 0;
  int bit_count = 0;
  int byte_index = 6;
  for (std::size_t i = kTimestampChars; i < kStringLength; ++i) {
    buffer = (buffer << 5) | static_cast<std::uint32_t>(values[i]);
    bit_count += 5;
    if (bit_count >= 8) {
      bit_count -= 8;
      bytes[byte_index++] = static_cast<std::byte>((buffer >> bit_count) & 0xFF);
    }
  }
  return Ulid{bytes};
}

}  // namespace agents_framework::core
