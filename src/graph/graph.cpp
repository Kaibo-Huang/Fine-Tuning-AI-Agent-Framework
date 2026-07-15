#include "agents_framework/graph/graph.hpp"

#include <algorithm>
#include <deque>

namespace agents_framework::graph {

std::optional<std::size_t> CompiledGraph::find_node(std::string_view name) const noexcept {
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].name == name) return i;
  }
  return std::nullopt;
}

core::Result<std::vector<std::size_t>> CompiledGraph::route(std::size_t index,
                                                            const ChannelMap& state) const {
  const CompiledNode& entry = nodes_.at(index);
  AF_TRY(auto names, entry.router(state));
  std::vector<std::size_t> targets;
  for (const std::string& name : names) {
    if (!entry.router_targets.empty() &&
        std::find(entry.router_targets.begin(), entry.router_targets.end(), name) ==
            entry.router_targets.end()) {
      return core::fail(core::ErrorCode::Invalid, "conditional edge returned undeclared target",
                        name);
    }
    if (name == kEnd) continue;
    const auto target = find_node(name);
    if (!target) {
      return core::fail(core::ErrorCode::Invalid, "conditional edge returned unknown target",
                        name);
    }
    targets.push_back(*target);
  }
  return targets;
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

void GraphSpec::add_router(std::string from, Router router, std::vector<std::string> targets) {
  routers_.push_back(RouterEntry{std::move(from), std::move(router), std::move(targets)});
}

bool GraphSpec::has_node(std::string_view name) const noexcept {
  return std::any_of(nodes_.begin(), nodes_.end(),
                     [name](const NodeEntry& entry) { return entry.name == name; });
}

namespace {

std::vector<bool> reachable_from(std::span<const std::size_t> roots,
                                 const std::vector<std::vector<std::size_t>>& targets) {
  std::vector<bool> seen(targets.size(), false);
  std::deque<std::size_t> frontier(roots.begin(), roots.end());
  while (!frontier.empty()) {
    const std::size_t index = frontier.front();
    frontier.pop_front();
    if (seen[index]) continue;
    seen[index] = true;
    for (const std::size_t next : targets[index]) {
      if (!seen[next]) frontier.push_back(next);
    }
  }
  return seen;
}

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
  std::vector<bool> exits(nodes_.size(), false);

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
    } else {
      exits[*from] = true;
    }
  }

  if (entry.empty()) {
    return core::fail(core::ErrorCode::Invalid,
                      "graph has no entry point; add an edge from __start__");
  }

  std::vector<Router> routers(nodes_.size());
  std::vector<std::vector<std::string>> router_targets(nodes_.size());
  std::vector<std::vector<std::size_t>> analysis = targets;
  for (RouterEntry& entry_fn : routers_) {
    const auto from = find(entry_fn.from);
    if (!from) {
      return core::fail(core::ErrorCode::Invalid, "conditional edge references unknown node", entry_fn.from);
    }
    if (routers[*from]) {
      return core::fail(core::ErrorCode::Invalid, "node already has a conditional edge", entry_fn.from);
    }
    if (!entry_fn.router) {
      return core::fail(core::ErrorCode::Invalid, "conditional edge has no router function", entry_fn.from);
    }
    if (entry_fn.targets.empty()) {
      exits[*from] = true;
      for (std::size_t i = 0; i < nodes_.size(); ++i) analysis[*from].push_back(i);
    } else {
      for (const std::string& target : entry_fn.targets) {
        if (target == kEnd) {
          exits[*from] = true;
          continue;
        }
        const auto to = find(target);
        if (!to) {
          return core::fail(core::ErrorCode::Invalid, "conditional edge declares unknown target", target);
        }
        analysis[*from].push_back(*to);
      }
    }
    routers[*from] = std::move(entry_fn.router);
    router_targets[*from] = std::move(entry_fn.targets);
  }

  std::sort(entry.begin(), entry.end());
  entry.erase(std::unique(entry.begin(), entry.end()), entry.end());
  for (auto& list : targets) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }
  for (auto& list : analysis) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }

  const std::vector<bool> reachable = reachable_from(entry, analysis);
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (!reachable[i]) {
      return core::fail(core::ErrorCode::Invalid, "node is unreachable from __start__", nodes_[i].name);
    }
    if (targets[i].empty() && !exits[i] && !routers[i]) {
      return core::fail(core::ErrorCode::Invalid, "node is a dead end; add an edge to __end__ or another node", nodes_[i].name);
    }
  }

  std::vector<std::vector<std::size_t>> reversed(nodes_.size());
  std::vector<std::size_t> exit_roots;
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (exits[i]) exit_roots.push_back(i);
    for (const std::size_t next : analysis[i]) reversed[next].push_back(i);
  }
  const std::vector<bool> terminates = reachable_from(exit_roots, reversed);
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (!terminates[i]) {
      return core::fail(core::ErrorCode::Invalid, "node has no path to __end__", nodes_[i].name);
    }
  }

  CompiledGraph compiled;
  compiled.entry_ = std::move(entry);
  compiled.nodes_.reserve(nodes_.size());
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    compiled.nodes_.push_back(CompiledGraph::CompiledNode{
        std::move(nodes_[i].name), std::move(nodes_[i].node), std::move(targets[i]),
        std::move(routers[i]), std::move(router_targets[i])});
  }
  return compiled;
}

}
