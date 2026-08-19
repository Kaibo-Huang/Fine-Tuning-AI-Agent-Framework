#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/task_suite.hpp"
#include "agents_framework/tools/tool.hpp"

namespace agents_framework::eval {

namespace detail {
struct SqlEnv;
}

struct SqlCase {
  std::string id;
  Split split{Split::Train};
  std::string question;
  std::string gold_sql;
};

struct TextToSqlSpec {
  std::string name{"ref-a-text-to-sql"};
  std::string schema_sql;
  std::vector<std::string> seeds;
  std::vector<SqlCase> cases;
};

class TextToSqlSuite final : public TaskSuite {
 public:
  static core::Result<TextToSqlSuite> create(TextToSqlSpec spec);

  [[nodiscard]] std::string_view name() const override { return name_; }
  [[nodiscard]] std::span<const TaskInstance> instances() const override { return instances_; }
  [[nodiscard]] Verifier& verifier() override { return *verifier_; }

  [[nodiscard]] std::unique_ptr<tools::Tool> make_sql_tool() const;

  [[nodiscard]] std::size_t database_count() const noexcept;

 private:
  TextToSqlSuite(std::string name, std::vector<TaskInstance> instances,
                 std::shared_ptr<detail::SqlEnv> env, std::unique_ptr<Verifier> verifier)
      : name_(std::move(name)),
        instances_(std::move(instances)),
        env_(std::move(env)),
        verifier_(std::move(verifier)) {}

  std::string name_;
  std::vector<TaskInstance> instances_;
  std::shared_ptr<detail::SqlEnv> env_;
  std::unique_ptr<Verifier> verifier_;
};

[[nodiscard]] TextToSqlSpec default_text_to_sql_spec();

}
