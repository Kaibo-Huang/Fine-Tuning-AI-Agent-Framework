#include "agents_framework/tools/process_tool.hpp"

#include <utility>

namespace agents_framework::tools {

ProcessTool::ProcessTool(ProcessToolOptions options) : options_(std::move(options)) {
  def_.name = options_.name;
  def_.description = options_.description;
  nlohmann::json properties = nlohmann::json::object();
  if (options_.allow_args) {
    properties["args"] = {{"type", "array"},
                          {"items", {{"type", "string"}}},
                          {"description", "Additional command-line arguments."}};
  }
  if (options_.allow_stdin) {
    properties["stdin"] = {{"type", "string"},
                           {"description", "Text piped to the command's standard input."}};
  }
  def_.input_schema = {
      {"type", "object"}, {"properties", properties}, {"additionalProperties", false}};
}

core::Result<std::string> ProcessTool::call(const nlohmann::json& args) {
  if (options_.command.empty()) {
    return core::fail(core::ErrorCode::Config, "process tool has no command configured",
                      options_.name);
  }

  std::vector<std::string> argv = options_.command;
  if (options_.allow_args) {
    if (const auto extra = args.find("args"); extra != args.end()) {
      for (const auto& arg : *extra) {
        argv.push_back(arg.get<std::string>());
      }
    }
  }

  ProcessOptions run_options;
  run_options.timeout = options_.timeout;
  run_options.max_output_bytes = options_.max_output_bytes;
  if (options_.allow_stdin) {
    run_options.stdin_data = args.value("stdin", std::string{});
  }

  AF_TRY(auto result, run_process(argv, run_options));

  std::string out;
  if (result.exit_code != 0) {
    out = "exit code " + std::to_string(result.exit_code) + "\n";
  }
  out += result.output;
  if (result.truncated) {
    out += "\n[output truncated]";
  }
  return out;
}

}
