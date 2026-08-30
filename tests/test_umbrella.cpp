#include <catch2/catch_test_macros.hpp>

#include "agents_framework/agents_framework.hpp"

TEST_CASE("agents_framework.hpp: every module compiles through one include", "[include]") {
  CHECK(!agents_framework::version().empty());
  CHECK(agents_framework::core::error_code_name(agents_framework::core::ErrorCode::Io) == "Io");
  CHECK(agents_framework::http::error_code_for_status(429) ==
        agents_framework::core::ErrorCode::RateLimited);
  CHECK(agents_framework::llm::role_name(agents_framework::llm::Role::Tool) == "tool");
  CHECK(agents_framework::tools::ToolRegistry{}.size() == 0);
  CHECK(agents_framework::graph::checkpoint_status_name(
            agents_framework::graph::CheckpointStatus::Completed) == "completed");
  const auto metric = agents_framework::store::Metric::Cosine;
  CHECK(agents_framework::store::metric_from_string(agents_framework::store::metric_name(metric)) ==
        metric);
  CHECK(agents_framework::trace::TrainingExample{}.turns.empty());
  CHECK(agents_framework::eval::split_name(agents_framework::eval::Split::HeldOut) == "held_out");
}
