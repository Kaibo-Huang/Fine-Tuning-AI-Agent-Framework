#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/tool.hpp"
#include "agents_framework/tools/tool.hpp"

namespace agents_framework::tools {

class ToolRegistry {
 public:
  core::Result<void> add(std::unique_ptr<Tool> tool);

  // The common case in one call: a JSON-schema definition plus its callback.
  core::Result<void> add(llm::ToolDef def, ToolFn fn) {
    return add(make_tool(std::move(def), std::move(fn)));
  }

  [[nodiscard]] Tool* find(std::string_view name) noexcept;
  [[nodiscard]] const Tool* find(std::string_view name) const noexcept;
  [[nodiscard]] std::vector<llm::ToolDef> defs() const;
  [[nodiscard]] std::size_t size() const noexcept { return tools_.size(); }

  core::Result<std::string> invoke(std::string_view name, const nlohmann::json& args);

 private:
  std::vector<std::unique_ptr<Tool>> tools_;
};

}
