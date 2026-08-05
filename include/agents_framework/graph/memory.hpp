#pragma once

#include <cstddef>
#include <vector>

#include "agents_framework/llm/message.hpp"

namespace agents_framework::graph {

[[nodiscard]] std::vector<llm::Message> window_messages(
    const std::vector<llm::Message>& messages, std::size_t max_messages);

}
