#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"

namespace agents_framework::eval {

struct ContinuationScore {
  double logprob{0.0};
  std::size_t tokens{0};

  bool operator==(const ContinuationScore&) const = default;
};

class SequenceScorer {
 public:
  virtual ~SequenceScorer() = default;

  virtual core::Result<std::vector<ContinuationScore>> score(
      std::string_view prompt, std::span<const std::string> continuations) = 0;
};

[[nodiscard]] core::Result<std::size_t> acc_norm_pick(std::span<const ContinuationScore> scores);

}
