#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"

namespace agents_framework::tools {

struct ProcessOptions {
  std::string stdin_data;
  std::chrono::milliseconds timeout{30000};
  std::size_t max_output_bytes{1 << 20};
};

struct ProcessResult {
  int exit_code{0};
  std::string output;
  bool truncated{false};
};

[[nodiscard]] core::Result<ProcessResult> run_process(const std::vector<std::string>& argv,
                                                      const ProcessOptions& options = {});

}
