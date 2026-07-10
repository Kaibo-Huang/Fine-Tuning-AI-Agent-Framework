#pragma once

#include <any>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"

namespace agents_framework::graph {

struct ChannelDef {
  std::string_view name;
  std::any (*make_default)();
  void (*reduce)(std::any& current, std::any&& update);
  nlohmann::json (*serialize)(const std::any& value);
  std::any (*deserialize)(const nlohmann::json& j);
};

class StateUpdate {
 public:
  struct Write {
    std::size_t channel;
    std::any value;
  };

  void write(std::size_t channel, std::any value) {
    writes_.push_back(Write{channel, std::move(value)});
  }

  [[nodiscard]] bool empty() const noexcept { return writes_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return writes_.size(); }
  [[nodiscard]] std::span<const Write> writes() const noexcept { return writes_; }
  [[nodiscard]] std::vector<Write> take_writes() && noexcept { return std::move(writes_); }

 private:
  std::vector<Write> writes_;
};

class ChannelMap {
 public:
  explicit ChannelMap(std::span<const ChannelDef> channels);

  [[nodiscard]] std::span<const ChannelDef> channels() const noexcept { return channels_; }
  [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }

  [[nodiscard]] const std::any& at(std::size_t channel) const { return slots_.at(channel); }
  [[nodiscard]] std::any& at(std::size_t channel) { return slots_.at(channel); }

  void apply(StateUpdate update);

  [[nodiscard]] nlohmann::json to_json() const;
  [[nodiscard]] static core::Result<ChannelMap> from_json(std::span<const ChannelDef> channels,
                                                          const nlohmann::json& j);

  [[nodiscard]] std::string serialize() const;
  [[nodiscard]] static core::Result<ChannelMap> deserialize(std::span<const ChannelDef> channels,
                                                            std::string_view bytes);

 private:
  std::span<const ChannelDef> channels_;
  std::vector<std::any> slots_;
};

}
