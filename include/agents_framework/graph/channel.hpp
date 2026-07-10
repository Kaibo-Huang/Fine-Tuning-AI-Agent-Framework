#pragma once

#include <any>
#include <concepts>
#include <iterator>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/fixed_string.hpp"
#include "agents_framework/graph/channel_map.hpp"

namespace agents_framework::graph {

template <class R, class T>
concept ReducerFor = requires(T& current, T update) {
  R::apply(current, std::move(update));
};

struct LastValue {
  template <class T>
  static void apply(T& current, T update) {
    current = std::move(update);
  }
};

struct Append {
  template <class T>
  static void apply(std::vector<T>& current, std::vector<T> update) {
    current.insert(current.end(), std::make_move_iterator(update.begin()),
                   std::make_move_iterator(update.end()));
  }
};

template <class S, class T>
concept SerdeFor = requires(const T& value, const nlohmann::json& j) {
  { S::to_json(value) } -> std::convertible_to<nlohmann::json>;
  { S::from_json(j) } -> std::convertible_to<T>;
};

template <class T>
struct JsonSerde {
  static nlohmann::json to_json(const T& value) { return nlohmann::json(value); }
  static T from_json(const nlohmann::json& j) { return j.get<T>(); }
};

template <core::FixedString Name, class T, class Reducer = LastValue, class Serde = JsonSerde<T>>
  requires std::copyable<T> && std::default_initializable<T> && ReducerFor<Reducer, T> &&
           SerdeFor<Serde, T>
struct Channel {
  static_assert(Name.size() > 0, "channel name must not be empty");

  static constexpr auto name = Name;
  using value_type = T;
  using reducer_type = Reducer;
  using serde_type = Serde;

  [[nodiscard]] static ChannelDef def() {
    return ChannelDef{
        .name = name.view(),
        .make_default = []() -> std::any { return std::any(std::in_place_type<T>); },
        .reduce =
            [](std::any& current, std::any&& update) {
              Reducer::apply(std::any_cast<T&>(current), std::move(std::any_cast<T&>(update)));
            },
        .serialize =
            [](const std::any& value) -> nlohmann::json {
          return Serde::to_json(std::any_cast<const T&>(value));
        },
        .deserialize =
            [](const nlohmann::json& j) -> std::any {
          return std::any(std::in_place_type<T>, Serde::from_json(j));
        },
    };
  }
};

}
