#include "agents_framework/graph/channel_map.hpp"

#include <algorithm>
#include <stdexcept>

namespace agents_framework::graph {

ChannelMap::ChannelMap(std::span<const ChannelDef> channels) : channels_(channels) {
  slots_.reserve(channels_.size());
  for (const ChannelDef& def : channels_) {
    slots_.push_back(def.make_default());
  }
}

void ChannelMap::apply(StateUpdate update) {
  auto writes = std::move(update).take_writes();
  for (auto& [channel, value] : writes) {
    if (channel >= slots_.size()) {
      throw std::out_of_range("StateUpdate write to unknown channel index " +
                              std::to_string(channel));
    }
    channels_[channel].reduce(slots_[channel], std::move(value));
  }
}

nlohmann::json ChannelMap::to_json() const {
  nlohmann::json j = nlohmann::json::object();
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    j[std::string{channels_[i].name}] = channels_[i].serialize(slots_[i]);
  }
  return j;
}

core::Result<ChannelMap> ChannelMap::from_json(std::span<const ChannelDef> channels,
                                               const nlohmann::json& j) {
  if (!j.is_object()) {
    return core::fail(core::ErrorCode::Parse, "serialized state must be a JSON object");
  }
  ChannelMap map{channels};
  for (std::size_t i = 0; i < channels.size(); ++i) {
    const ChannelDef& def = channels[i];
    const auto it = j.find(std::string{def.name});
    if (it == j.end()) {
      return core::fail(core::ErrorCode::Parse, "serialized state is missing a channel",
                        std::string{def.name});
    }
    try {
      map.slots_[i] = def.deserialize(*it);
    } catch (const std::exception& error) {
      return core::fail(core::ErrorCode::Parse, error.what(), std::string{def.name});
    }
  }
  if (j.size() != channels.size()) {
    for (const auto& [key, value] : j.items()) {
      const bool known = std::any_of(channels.begin(), channels.end(),
                                     [&key](const ChannelDef& def) { return def.name == key; });
      if (!known) {
        return core::fail(core::ErrorCode::Parse, "unknown channel in serialized state", key);
      }
    }
  }
  return map;
}

std::string ChannelMap::serialize() const { return to_json().dump(); }

core::Result<ChannelMap> ChannelMap::deserialize(std::span<const ChannelDef> channels,
                                                 std::string_view bytes) {
  const nlohmann::json j = nlohmann::json::parse(bytes, nullptr, false);
  if (j.is_discarded()) {
    return core::fail(core::ErrorCode::Parse, "serialized state is not valid JSON");
  }
  return from_json(channels, j);
}

}
