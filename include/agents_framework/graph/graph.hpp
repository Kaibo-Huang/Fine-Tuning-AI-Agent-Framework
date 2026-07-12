#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "agents_framework/graph/node.hpp"

namespace agents_framework::graph {

inline constexpr std::string_view kStart{"__start__"};
inline constexpr std::string_view kEnd{"__end__"};

struct Edge {
  std::string from;
  std::string to;
  bool operator==(const Edge&) const = default;
};

class GraphSpec {
 public:
  struct NodeEntry {
    std::string name;
    std::unique_ptr<Node> node;
  };

  void add_node(std::string name, std::unique_ptr<Node> node);
  void add_edge(std::string from, std::string to);

  [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
  [[nodiscard]] bool has_node(std::string_view name) const noexcept;
  [[nodiscard]] std::span<const NodeEntry> nodes() const noexcept { return nodes_; }
  [[nodiscard]] std::span<const Edge> edges() const noexcept { return edges_; }

 private:
  std::vector<NodeEntry> nodes_;
  std::vector<Edge> edges_;
};

}
