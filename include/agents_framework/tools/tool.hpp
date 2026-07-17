#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/tool.hpp"

namespace agents_framework::tools {

class Tool {
 public:
  virtual ~Tool() = default;

  [[nodiscard]] virtual const llm::ToolDef& def() const = 0;
  virtual core::Result<std::string> call(const nlohmann::json& args) = 0;

  [[nodiscard]] std::string_view name() const { return def().name; }
};

[[nodiscard]] core::Result<void> validate_args(const nlohmann::json& schema,
                                               const nlohmann::json& args);

using ToolFn = std::function<core::Result<std::string>(const nlohmann::json&)>;

class FnTool final : public Tool {
 public:
  FnTool(llm::ToolDef def, ToolFn fn) : def_(std::move(def)), fn_(std::move(fn)) {}

  [[nodiscard]] const llm::ToolDef& def() const override { return def_; }
  core::Result<std::string> call(const nlohmann::json& args) override { return fn_(args); }

 private:
  llm::ToolDef def_;
  ToolFn fn_;
};

[[nodiscard]] inline std::unique_ptr<Tool> make_tool(llm::ToolDef def, ToolFn fn) {
  return std::make_unique<FnTool>(std::move(def), std::move(fn));
}

}
