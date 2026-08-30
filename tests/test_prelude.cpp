#include <catch2/catch_test_macros.hpp>

#include "agents_framework/prelude.hpp"

TEST_CASE("prelude: every module is reachable without a qualifier", "[include]") {
  CHECK(!version().empty());

  const Result<int> value = 1;
  REQUIRE(value);
  CHECK(*value == 1);

  CHECK(error_code_for_status(404) == ErrorCode::NotFound);
  CHECK(Message::user_text("hi").role == Role::User);
  CHECK(ToolRegistry{}.size() == 0);

  const auto state = chat_state("hi");
  CHECK(state.get<"messages">().size() == 1);

  const auto db = Db::open_memory();
  CHECK(db);

  CHECK(TrainingExample{}.turns.empty());
  CHECK(split_name(Split::Train) == "train");

  const json document = {{"answer", 42}};
  CHECK(document.at("answer") == 42);
}
