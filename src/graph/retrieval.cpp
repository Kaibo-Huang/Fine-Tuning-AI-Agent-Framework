#include "agents_framework/graph/retrieval_node.hpp"

namespace agents_framework::graph {

std::string render_documents(std::span<const Document> documents) {
  if (documents.empty()) return {};
  std::string out = "Relevant context:\n";
  for (std::size_t i = 0; i < documents.size(); ++i) {
    out += "\n[" + std::to_string(i + 1) + "] " + documents[i].content + "\n";
  }
  return out;
}

}
