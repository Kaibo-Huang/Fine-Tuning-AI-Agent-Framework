#include "agents_framework/tools/registry.hpp"

#include <string>

namespace agents_framework::tools {

core::Result<void> ToolRegistry::add(std::unique_ptr<Tool> tool) {
  if (!tool) {
    return core::fail(core::ErrorCode::Invalid, "tool must not be null");
  }
  if (tool->name().empty()) {
    return core::fail(core::ErrorCode::Invalid, "tool name must not be empty");
  }
  if (find(tool->name()) != nullptr) {
    return core::fail(core::ErrorCode::Invalid, "duplicate tool name",
                      std::string{tool->name()});
  }
  tools_.push_back(std::move(tool));
  return {};
}

Tool* ToolRegistry::find(std::string_view name) noexcept {
  for (const auto& tool : tools_) {
    if (tool->name() == name) return tool.get();
  }
  return nullptr;
}

const Tool* ToolRegistry::find(std::string_view name) const noexcept {
  for (const auto& tool : tools_) {
    if (tool->name() == name) return tool.get();
  }
  return nullptr;
}

std::vector<llm::ToolDef> ToolRegistry::defs() const {
  std::vector<llm::ToolDef> defs;
  defs.reserve(tools_.size());
  for (const auto& tool : tools_) {
    defs.push_back(tool->def());
  }
  return defs;
}

core::Result<std::string> ToolRegistry::invoke(std::string_view name,
                                               const nlohmann::json& args) {
  Tool* tool = find(name);
  if (!tool) {
    return core::fail(core::ErrorCode::NotFound, "unknown tool", std::string{name});
  }
  AF_TRY_VOID(validate_args(tool->def().input_schema, args));
  auto result = tool->call(args);
  if (!result && result.error().context.empty()) {
    result.error().context = std::string{name};
  }
  return result;
}

}
