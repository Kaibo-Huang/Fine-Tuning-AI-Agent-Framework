#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/node.hpp"

namespace agents_framework::graph {

inline constexpr std::string_view kStart{"__start__"};
inline constexpr std::string_view kEnd{"__end__"};

struct Edge {
  std::string from;
  std::string to;
  bool operator==(const Edge&) const = default;
};

class CompiledGraph {
 public:
  [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
  [[nodiscard]] std::string_view node_name(std::size_t index) const {
    return nodes_.at(index).name;
  }
  [[nodiscard]] std::optional<std::size_t> find_node(std::string_view name) const noexcept;
  [[nodiscard]] std::span<const std::size_t> entry_nodes() const noexcept { return entry_; }
  [[nodiscard]] std::span<const std::size_t> static_targets(std::size_t index) const {
    return nodes_.at(index).targets;
  }
  [[nodiscard]] Node& node(std::size_t index) { return *nodes_.at(index).node; }

 private:
  friend class GraphSpec;

  struct CompiledNode {
    std::string name;
    std::unique_ptr<Node> node;
    std::vector<std::size_t> targets;
  };

  CompiledGraph() = default;

  std::vector<CompiledNode> nodes_;
  std::vector<std::size_t> entry_;
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

  [[nodiscard]] core::Result<CompiledGraph> compile() &&;

 private:
  std::vector<NodeEntry> nodes_;
  std::vector<Edge> edges_;
};

}
