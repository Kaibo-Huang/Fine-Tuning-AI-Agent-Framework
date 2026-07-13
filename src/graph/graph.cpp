#include "agents_framework/graph/graph.hpp"

#include <algorithm>

namespace agents_framework::graph {

std::optional<std::size_t> CompiledGraph::find_node(std::string_view name) const noexcept {
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].name == name) return i;
  }
  return std::nullopt;
}

void GraphSpec::add_node(std::string name, std::unique_ptr<Node> node) {
  nodes_.push_back(NodeEntry{std::move(name), std::move(node)});
}

void GraphSpec::add_edge(std::string from, std::string to) {
  Edge edge{std::move(from), std::move(to)};
  if (std::find(edges_.begin(), edges_.end(), edge) == edges_.end()) {
    edges_.push_back(std::move(edge));
  }
}

bool GraphSpec::has_node(std::string_view name) const noexcept {
  return std::any_of(nodes_.begin(), nodes_.end(),
                     [name](const NodeEntry& entry) { return entry.name == name; });
}

core::Result<CompiledGraph> GraphSpec::compile() && {
  if (nodes_.empty()) {
    return core::fail(core::ErrorCode::Invalid, "graph has no nodes");
  }
  for (const NodeEntry& entry : nodes_) {
    if (entry.name.empty()) {
      return core::fail(core::ErrorCode::Invalid, "node name must not be empty");
    }
    if (entry.name == kStart || entry.name == kEnd) {
      return core::fail(core::ErrorCode::Invalid, "node name is reserved", entry.name);
    }
    if (!entry.node) {
      return core::fail(core::ErrorCode::Invalid, "node has no implementation", entry.name);
    }
  }
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    for (std::size_t k = i + 1; k < nodes_.size(); ++k) {
      if (nodes_[i].name == nodes_[k].name) {
        return core::fail(core::ErrorCode::Invalid, "duplicate node name", nodes_[i].name);
      }
    }
  }

  const auto find = [this](std::string_view name) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].name == name) return i;
    }
    return std::nullopt;
  };

  std::vector<std::size_t> entry;
  std::vector<std::vector<std::size_t>> targets(nodes_.size());

  for (const Edge& edge : edges_) {
    if (edge.to == kStart) {
      return core::fail(core::ErrorCode::Invalid, "edge may not target __start__",
                        edge.from + " -> " + edge.to);
    }
    if (edge.from == kEnd) {
      return core::fail(core::ErrorCode::Invalid, "edge may not leave __end__",
                        edge.from + " -> " + edge.to);
    }
    if (edge.from == kStart && edge.to == kEnd) {
      return core::fail(core::ErrorCode::Invalid, "edge from __start__ must target a node");
    }

    std::optional<std::size_t> to;
    if (edge.to != kEnd) {
      to = find(edge.to);
      if (!to) {
        return core::fail(core::ErrorCode::Invalid, "edge references unknown node", edge.to);
      }
    }
    if (edge.from == kStart) {
      entry.push_back(*to);
      continue;
    }
    const auto from = find(edge.from);
    if (!from) {
      return core::fail(core::ErrorCode::Invalid, "edge references unknown node", edge.from);
    }
    if (to) {
      targets[*from].push_back(*to);
    }
  }

  if (entry.empty()) {
    return core::fail(core::ErrorCode::Invalid,
                      "graph has no entry point; add an edge from __start__");
  }

  std::sort(entry.begin(), entry.end());
  entry.erase(std::unique(entry.begin(), entry.end()), entry.end());
  for (auto& list : targets) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }

  CompiledGraph compiled;
  compiled.entry_ = std::move(entry);
  compiled.nodes_.reserve(nodes_.size());
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    compiled.nodes_.push_back(CompiledGraph::CompiledNode{
        std::move(nodes_[i].name), std::move(nodes_[i].node), std::move(targets[i])});
  }
  return compiled;
}

}
