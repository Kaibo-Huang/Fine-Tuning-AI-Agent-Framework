#include "agents_framework/graph/graph.hpp"

#include <algorithm>

namespace agents_framework::graph {

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

}
