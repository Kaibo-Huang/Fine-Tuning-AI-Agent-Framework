#pragma once

#include <any>
#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <nlohmann/json.hpp>

#include "agents_framework/core/fixed_string.hpp"
#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/channel.hpp"
#include "agents_framework/graph/channel_map.hpp"

namespace agents_framework::graph {

namespace detail {

template <class... Cs>
[[nodiscard]] consteval bool channel_names_unique() {
  const std::array<std::string_view, sizeof...(Cs)> names{Cs::name.view()...};
  for (std::size_t i = 0; i < names.size(); ++i) {
    for (std::size_t k = i + 1; k < names.size(); ++k) {
      if (names[i] == names[k]) return false;
    }
  }
  return true;
}

template <core::FixedString Name, class... Cs>
[[nodiscard]] consteval std::size_t channel_index() {
  const std::array<std::string_view, sizeof...(Cs)> names{Cs::name.view()...};
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] == Name.view()) return i;
  }
  return sizeof...(Cs);
}

}

template <class... Cs>
struct Schema {
  static_assert(sizeof...(Cs) > 0, "a Schema needs at least one channel");
  static_assert(detail::channel_names_unique<Cs...>(), "Schema channel names must be unique");

  static constexpr std::size_t size = sizeof...(Cs);

  template <core::FixedString Name>
  static constexpr bool has = ((Cs::name == Name) || ...);

  template <core::FixedString Name>
  [[nodiscard]] static consteval std::size_t index_of() {
    static_assert(has<Name>, "Schema has no channel with this name");
    return detail::channel_index<Name, Cs...>();
  }

  template <core::FixedString Name>
  using channel_of = std::tuple_element_t<index_of<Name>(), std::tuple<Cs...>>;

  template <core::FixedString Name>
  using value_of = typename channel_of<Name>::value_type;

  [[nodiscard]] static std::span<const ChannelDef> channels() {
    static const std::array<ChannelDef, size> defs{Cs::def()...};
    return defs;
  }
};

template <class S>
class StateView {
 public:
  explicit StateView(const ChannelMap& map) : map_(&map) {
    assert(map.channels().data() == S::channels().data() &&
           "ChannelMap does not belong to this Schema");
  }

  template <core::FixedString Name>
  [[nodiscard]] const typename S::template value_of<Name>& get() const {
    return std::any_cast<const typename S::template value_of<Name>&>(
        map_->at(S::template index_of<Name>()));
  }

  [[nodiscard]] const ChannelMap& map() const noexcept { return *map_; }

 private:
  const ChannelMap* map_;
};

template <class S>
class Update {
 public:
  template <core::FixedString Name>
  Update& write(typename S::template value_of<Name> value) & {
    write_impl<Name>(std::move(value));
    return *this;
  }

  template <core::FixedString Name>
  Update&& write(typename S::template value_of<Name> value) && {
    write_impl<Name>(std::move(value));
    return std::move(*this);
  }

  [[nodiscard]] bool empty() const noexcept { return update_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return update_.size(); }
  [[nodiscard]] StateUpdate take() && noexcept { return std::move(update_); }

 private:
  template <core::FixedString Name>
  void write_impl(typename S::template value_of<Name> value) {
    update_.write(S::template index_of<Name>(),
                  std::any(std::in_place_type<typename S::template value_of<Name>>,
                           std::move(value)));
  }

  StateUpdate update_;
};

template <class S>
class State {
 public:
  using schema_type = S;

  State() : map_(S::channels()) {}

  explicit State(ChannelMap map) : map_(std::move(map)) {
    assert(map_.channels().data() == S::channels().data() &&
           "ChannelMap does not belong to this Schema");
  }

  template <core::FixedString Name>
  [[nodiscard]] const typename S::template value_of<Name>& get() const {
    return view().template get<Name>();
  }

  template <core::FixedString Name>
  void set(typename S::template value_of<Name> value) {
    map_.at(S::template index_of<Name>()) = std::any(
        std::in_place_type<typename S::template value_of<Name>>, std::move(value));
  }

  template <core::FixedString Name>
  void reduce(typename S::template value_of<Name> update) {
    using Ch = typename S::template channel_of<Name>;
    Ch::reducer_type::apply(
        std::any_cast<typename Ch::value_type&>(map_.at(S::template index_of<Name>())),
        std::move(update));
  }

  void apply(Update<S> update) { map_.apply(std::move(update).take()); }
  void apply(StateUpdate update) { map_.apply(std::move(update)); }

  [[nodiscard]] nlohmann::json to_json() const { return map_.to_json(); }
  [[nodiscard]] std::string serialize() const { return map_.serialize(); }

  [[nodiscard]] static core::Result<State> from_json(const nlohmann::json& j) {
    AF_TRY(auto map, ChannelMap::from_json(S::channels(), j));
    return State(std::move(map));
  }

  [[nodiscard]] static core::Result<State> deserialize(std::string_view bytes) {
    AF_TRY(auto map, ChannelMap::deserialize(S::channels(), bytes));
    return State(std::move(map));
  }

  [[nodiscard]] StateView<S> view() const { return StateView<S>(map_); }
  [[nodiscard]] const ChannelMap& map() const noexcept { return map_; }
  [[nodiscard]] ChannelMap& map() noexcept { return map_; }

 private:
  ChannelMap map_;
};

}
