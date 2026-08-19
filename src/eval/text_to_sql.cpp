#include "agents_framework/eval/text_to_sql.hpp"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <optional>
#include <utility>

#include <nlohmann/json.hpp>

#include "agents_framework/store/db.hpp"

namespace agents_framework::eval {

namespace {

using Row = std::vector<std::optional<std::string>>;
using ResultSet = std::vector<Row>;

constexpr std::size_t kMaxRows = 10000;
constexpr std::size_t kToolRowLimit = 50;

}

namespace detail {

struct SqlEnv {
  std::vector<store::Db> dbs;
  std::mutex mutex;

  core::Result<ResultSet> execute(store::Db& db, std::string_view sql) {
    AF_TRY_VOID(db.exec("BEGIN"));
    auto rows = collect(db, sql);
    const auto rolled_back = db.exec("ROLLBACK");
    if (rows && !rolled_back) return std::unexpected(rolled_back.error());
    return rows;
  }

 private:
  static core::Result<ResultSet> collect(store::Db& db, std::string_view sql) {
    AF_TRY(auto stmt, db.prepare(sql));
    ResultSet rows;
    while (true) {
      AF_TRY(const bool row, stmt.step());
      if (!row) break;
      if (rows.size() >= kMaxRows) {
        return core::fail(core::ErrorCode::Invalid, "query returned too many rows",
                          "cap is " + std::to_string(kMaxRows));
      }
      Row values;
      const int columns = stmt.column_count();
      values.reserve(static_cast<std::size_t>(columns));
      for (int c = 0; c < columns; ++c) {
        values.push_back(stmt.column_is_null(c) ? std::nullopt
                                                : std::optional{stmt.column_text(c)});
      }
      rows.push_back(std::move(values));
    }
    return rows;
  }
};

}

namespace {

[[nodiscard]] std::string_view extract_sql(std::string_view output) {
  const auto trim = [](std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
      text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
      text.remove_suffix(1);
    }
    return text;
  };
  std::string_view sql = trim(output);
  if (sql.starts_with("```")) {
    sql.remove_prefix(3);
    if (sql.starts_with("sql")) sql.remove_prefix(3);
    if (const std::size_t fence = sql.find("```"); fence != std::string_view::npos) {
      sql = sql.substr(0, fence);
    }
    sql = trim(sql);
  }
  return sql;
}

class ExecutionMatchVerifier final : public Verifier {
 public:
  explicit ExecutionMatchVerifier(std::shared_ptr<detail::SqlEnv> env)
      : env_(std::move(env)) {}

  core::Result<double> score(const TaskInstance& instance, std::string_view output) override {
    const auto gold_it = instance.expected.find("gold_sql");
    if (gold_it == instance.expected.end() || !gold_it->is_string()) {
      return core::fail(core::ErrorCode::Invalid, "instance is missing expected.gold_sql",
                        "'" + instance.id + "'");
    }
    const std::string gold = gold_it->get<std::string>();
    const std::string_view predicted = extract_sql(output);
    if (predicted.empty()) return 0.0;

    const std::scoped_lock lock{env_->mutex};
    for (store::Db& db : env_->dbs) {
      auto want = env_->execute(db, gold);
      if (!want) {
        core::Error error = std::move(want).error();
        error.context = "gold query for '" + instance.id + "'" +
                        (error.context.empty() ? "" : ": " + error.context);
        return std::unexpected(std::move(error));
      }
      auto got = env_->execute(db, predicted);
      if (!got) return 0.0;

      std::sort(want->begin(), want->end());
      std::sort(got->begin(), got->end());
      if (*want != *got) return 0.0;
    }
    return 1.0;
  }

 private:
  std::shared_ptr<detail::SqlEnv> env_;
};

class SqlTool final : public tools::Tool {
 public:
  explicit SqlTool(std::shared_ptr<detail::SqlEnv> env) : env_(std::move(env)) {
    def_.name = "sql";
    def_.description =
        "Execute a SQL query against the task database and return the rows as JSON. "
        "Changes are always rolled back; use SELECT to explore the schema and data.";
    def_.input_schema = {
        {"type", "object"},
        {"properties",
         {{"query", {{"type", "string"}, {"description", "The SQL query to execute."}}}}},
        {"required", nlohmann::json::array({"query"})},
    };
  }

  [[nodiscard]] const llm::ToolDef& def() const override { return def_; }

  core::Result<std::string> call(const nlohmann::json& args) override {
    const std::string query = args.value("query", std::string{});
    if (query.empty()) {
      return core::fail(core::ErrorCode::Invalid, "the sql tool needs a non-empty query");
    }

    const std::scoped_lock lock{env_->mutex};
    store::Db& db = env_->dbs.front();

    AF_TRY(auto stmt, db.prepare(query));
    nlohmann::json columns = nlohmann::json::array();
    for (int c = 0; c < stmt.column_count(); ++c) columns.push_back(stmt.column_name(c));
    stmt = store::Statement{};

    AF_TRY(const auto rows, env_->execute(db, query));
    nlohmann::json out = nlohmann::json::object();
    out["columns"] = std::move(columns);
    nlohmann::json values = nlohmann::json::array();
    for (std::size_t i = 0; i < rows.size() && i < kToolRowLimit; ++i) {
      nlohmann::json row = nlohmann::json::array();
      for (const auto& cell : rows[i]) {
        if (cell) {
          row.push_back(*cell);
        } else {
          row.push_back(nullptr);
        }
      }
      values.push_back(std::move(row));
    }
    out["rows"] = std::move(values);
    out["row_count"] = rows.size();
    if (rows.size() > kToolRowLimit) out["truncated_to"] = kToolRowLimit;
    return out.dump();
  }

 private:
  std::shared_ptr<detail::SqlEnv> env_;
  llm::ToolDef def_;
};

}

core::Result<TextToSqlSuite> TextToSqlSuite::create(TextToSqlSpec spec) {
  if (spec.schema_sql.empty()) {
    return core::fail(core::ErrorCode::Invalid, "a text-to-SQL suite needs schema DDL",
                      spec.name);
  }
  if (spec.seeds.empty()) {
    return core::fail(core::ErrorCode::Invalid,
                      "a text-to-SQL suite needs at least one database instance", spec.name);
  }
  if (spec.cases.empty()) {
    return core::fail(core::ErrorCode::Invalid, "a text-to-SQL suite needs cases", spec.name);
  }

  auto env = std::make_shared<detail::SqlEnv>();
  env->dbs.reserve(spec.seeds.size());
  for (std::size_t i = 0; i < spec.seeds.size(); ++i) {
    AF_TRY(auto db, store::Db::open_memory());
    if (auto applied = db.exec(spec.schema_sql); !applied) {
      core::Error error = std::move(applied).error();
      error.context = "schema for database " + std::to_string(i) +
                      (error.context.empty() ? "" : ": " + error.context);
      return std::unexpected(std::move(error));
    }
    if (auto seeded = db.exec(spec.seeds[i]); !seeded) {
      core::Error error = std::move(seeded).error();
      error.context = "seed for database " + std::to_string(i) +
                      (error.context.empty() ? "" : ": " + error.context);
      return std::unexpected(std::move(error));
    }
    env->dbs.push_back(std::move(db));
  }

  std::vector<TaskInstance> instances;
  instances.reserve(spec.cases.size());
  for (const SqlCase& sql_case : spec.cases) {
    TaskInstance instance;
    instance.id = sql_case.id;
    instance.split = sql_case.split;
    instance.input["question"] = sql_case.question;
    instance.expected["gold_sql"] = sql_case.gold_sql;
    instances.push_back(std::move(instance));
  }

  auto verifier = std::make_unique<ExecutionMatchVerifier>(env);

  {
    const std::scoped_lock lock{env->mutex};
    for (std::size_t i = 0; i < instances.size(); ++i) {
      for (store::Db& db : env->dbs) {
        if (auto rows = env->execute(db, spec.cases[i].gold_sql); !rows) {
          core::Error error = std::move(rows).error();
          error.context = "gold query for '" + instances[i].id + "'" +
                          (error.context.empty() ? "" : ": " + error.context);
          return std::unexpected(std::move(error));
        }
      }
    }
  }

  AF_TRY_VOID(validate_suite(spec.name, instances));

  return TextToSqlSuite(std::move(spec.name), std::move(instances), std::move(env),
                        std::move(verifier));
}

std::unique_ptr<tools::Tool> TextToSqlSuite::make_sql_tool() const {
  return std::make_unique<SqlTool>(env_);
}

std::size_t TextToSqlSuite::database_count() const noexcept { return env_->dbs.size(); }

TextToSqlSpec default_text_to_sql_spec() {
  TextToSqlSpec spec;
  spec.schema_sql =
      "CREATE TABLE customers ("
      "  id INTEGER PRIMARY KEY,"
      "  name TEXT NOT NULL,"
      "  city TEXT NOT NULL,"
      "  age INTEGER NOT NULL"
      ");"
      "CREATE TABLE orders ("
      "  id INTEGER PRIMARY KEY,"
      "  customer_id INTEGER NOT NULL REFERENCES customers(id),"
      "  total REAL NOT NULL,"
      "  status TEXT NOT NULL"
      ");";

  spec.seeds = {
      "INSERT INTO customers VALUES (1,'Ada','Toronto',34),(2,'Ben','Waterloo',28),"
      " (3,'Cleo','Toronto',45),(4,'Dev','Montreal',22);"
      "INSERT INTO orders VALUES (1,1,120.0,'completed'),(2,1,80.0,'refunded'),"
      " (3,2,42.5,'completed'),(4,3,300.0,'completed'),(5,4,15.0,'pending');",

      "INSERT INTO customers VALUES (1,'Elle','Waterloo',30),(2,'Finn','Toronto',31),"
      " (3,'Gus','Ottawa',29),(4,'Hana','Toronto',52),(5,'Iris','Waterloo',30);"
      "INSERT INTO orders VALUES (1,1,60.0,'completed'),(2,2,10.0,'pending'),"
      " (3,3,250.0,'completed'),(4,4,99.0,'refunded'),(5,5,120.0,'completed'),"
      " (6,2,75.0,'completed');",

      "INSERT INTO customers VALUES (1,'Jo','Montreal',61),(2,'Kai','Toronto',19);"
      "INSERT INTO orders VALUES (1,1,500.0,'completed'),(2,2,20.0,'pending');",
  };

  spec.cases = {
      {"count-customers", Split::Train, "How many customers are there?",
       "SELECT COUNT(*) FROM customers"},
      {"toronto-names", Split::Train, "List the names of customers in Toronto.",
       "SELECT name FROM customers WHERE city = 'Toronto'"},
      {"older-than-30", Split::Train, "How many customers are older than 30?",
       "SELECT COUNT(*) FROM customers WHERE age > 30"},
      {"completed-revenue", Split::Train, "What is the total revenue from completed orders?",
       "SELECT SUM(total) FROM orders WHERE status = 'completed'"},
      {"orders-per-customer", Split::Train,
       "For each customer, how many orders have they placed? Return name and count.",
       "SELECT c.name, COUNT(o.id) FROM customers c LEFT JOIN orders o"
       " ON o.customer_id = c.id GROUP BY c.id, c.name"},
      {"cities", Split::Train, "Which distinct cities do customers live in?",
       "SELECT DISTINCT city FROM customers"},

      {"youngest-customer", Split::HeldOut, "What is the name of the youngest customer?",
       "SELECT name FROM customers ORDER BY age ASC, id ASC LIMIT 1"},
      {"pending-count", Split::HeldOut, "How many orders are pending?",
       "SELECT COUNT(*) FROM orders WHERE status = 'pending'"},
      {"big-spenders", Split::HeldOut,
       "Which customers have spent more than 100 in total across completed orders? "
       "Return their names.",
       "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id"
       " WHERE o.status = 'completed' GROUP BY c.id, c.name HAVING SUM(o.total) > 100"},
      {"average-age", Split::HeldOut, "What is the average customer age?",
       "SELECT AVG(age) FROM customers"},

      {"refunded-total", Split::Retention, "What is the total value of refunded orders?",
       "SELECT SUM(total) FROM orders WHERE status = 'refunded'"},
      {"waterloo-count", Split::Retention, "How many customers live in Waterloo?",
       "SELECT COUNT(*) FROM customers WHERE city = 'Waterloo'"},
  };
  return spec;
}

}
