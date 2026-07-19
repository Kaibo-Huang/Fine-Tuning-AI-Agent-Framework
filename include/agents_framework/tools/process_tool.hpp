#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/tools/process.hpp"
#include "agents_framework/tools/tool.hpp"

namespace agents_framework::tools {

struct ProcessToolOptions {
  std::string name;
  std::string description;
  std::vector<std::string> command;
  bool allow_args{false};
  bool allow_stdin{false};
  std::chrono::milliseconds timeout{30000};
  std::size_t max_output_bytes{1 << 20};
};

class ProcessTool final : public Tool {
 public:
  explicit ProcessTool(ProcessToolOptions options);

  [[nodiscard]] const llm::ToolDef& def() const override { return def_; }
  core::Result<std::string> call(const nlohmann::json& args) override;

 private:
  ProcessToolOptions options_;
  llm::ToolDef def_;
};

[[nodiscard]] inline std::unique_ptr<Tool> make_process_tool(ProcessToolOptions options) {
  return std::make_unique<ProcessTool>(std::move(options));
}

}
