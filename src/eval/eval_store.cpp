#include "agents_framework/eval/eval_store.hpp"

#include <array>
#include <string>
#include <utility>

#include "agents_framework/store/migrate.hpp"

namespace agents_framework::eval {

namespace {

constexpr std::array<store::Migration, 1> kMigrations{{
    {1,
     "CREATE TABLE IF NOT EXISTS eval_runs ("
     "  eval_id TEXT PRIMARY KEY,"
     "  suite TEXT NOT NULL,"
     "  model TEXT NOT NULL DEFAULT '',"
     "  adapter TEXT NOT NULL DEFAULT '',"
     "  label TEXT NOT NULL DEFAULT '',"
     "  split TEXT NOT NULL DEFAULT '',"
     "  seed INTEGER NOT NULL DEFAULT 0,"
     "  framework_version TEXT NOT NULL DEFAULT '',"
     "  framework_commit TEXT NOT NULL DEFAULT '',"
     "  total INTEGER NOT NULL DEFAULT 0,"
     "  failures INTEGER NOT NULL DEFAULT 0,"
     "  mean_score REAL NOT NULL DEFAULT 0,"
     "  ci_low REAL NOT NULL DEFAULT 0,"
     "  ci_high REAL NOT NULL DEFAULT 0,"
     "  baseline INTEGER NOT NULL DEFAULT 0,"
     "  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
     ");"
     "CREATE TABLE IF NOT EXISTS eval_results ("
     "  eval_id TEXT NOT NULL,"
     "  task_id TEXT NOT NULL,"
     "  split TEXT NOT NULL DEFAULT '',"
     "  score REAL NOT NULL DEFAULT 0,"
     "  verified INTEGER NOT NULL DEFAULT 0,"
     "  ok INTEGER NOT NULL DEFAULT 1,"
     "  error TEXT NOT NULL DEFAULT '',"
     "  detail TEXT NOT NULL DEFAULT '',"
     "  trace_id TEXT NOT NULL DEFAULT '',"
     "  PRIMARY KEY (eval_id, task_id)"
     ");"
     "CREATE INDEX IF NOT EXISTS eval_runs_by_suite ON eval_runs(suite);"},
}};

}

core::Result<EvalStore> EvalStore::open(std::shared_ptr<store::Db> db) {
  if (!db || !db->is_open()) {
    return core::fail(core::ErrorCode::Invalid, "EvalStore requires an open database");
  }
  AF_TRY_VOID(store::migrate(*db, "eval", kMigrations));
  return EvalStore(std::move(db));
}

core::Result<void> EvalStore::save(const EvalReport& report) {
  if (report.eval_id.empty()) {
    return core::fail(core::ErrorCode::Invalid, "eval_id must not be empty");
  }
  if (report.suite.empty()) {
    return core::fail(core::ErrorCode::Invalid, "an eval report needs a suite name",
                      report.eval_id);
  }

  AF_TRY(auto transaction, store::Transaction::begin(*db_));
  AF_TRY(auto run, db_->prepare(
                       "INSERT INTO eval_runs (eval_id, suite, model, adapter, label, split,"
                       "  seed, framework_version, framework_commit, total, failures,"
                       "  mean_score, ci_low, ci_high)"
                       " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)"));
  AF_TRY_VOID(run.bind(1, report.eval_id));
  AF_TRY_VOID(run.bind(2, report.suite));
  AF_TRY_VOID(run.bind(3, report.model));
  AF_TRY_VOID(run.bind(4, report.adapter));
  AF_TRY_VOID(run.bind(5, report.label));
  AF_TRY_VOID(run.bind(6, report.split));
  AF_TRY_VOID(run.bind(7, static_cast<std::int64_t>(report.seed)));
  AF_TRY_VOID(run.bind(8, report.framework_version));
  AF_TRY_VOID(run.bind(9, report.framework_commit));
  AF_TRY_VOID(run.bind(10, static_cast<std::int64_t>(report.total)));
  AF_TRY_VOID(run.bind(11, static_cast<std::int64_t>(report.failures)));
  AF_TRY_VOID(run.bind(12, report.mean_score));
  AF_TRY_VOID(run.bind(13, report.ci.low));
  AF_TRY_VOID(run.bind(14, report.ci.high));
  AF_TRY_VOID(run.step());

  AF_TRY(auto row, db_->prepare(
                       "INSERT INTO eval_results (eval_id, task_id, split, score, verified,"
                       "  ok, error, detail, trace_id)"
                       " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"));
  for (const InstanceResult& result : report.results) {
    AF_TRY_VOID(row.reset());
    AF_TRY_VOID(row.bind(1, report.eval_id));
    AF_TRY_VOID(row.bind(2, result.task_id));
    AF_TRY_VOID(row.bind(3, split_name(result.split)));
    AF_TRY_VOID(row.bind(4, result.score));
    AF_TRY_VOID(row.bind(5, static_cast<std::int64_t>(result.verified ? 1 : 0)));
    AF_TRY_VOID(row.bind(6, static_cast<std::int64_t>(result.ok ? 1 : 0)));
    AF_TRY_VOID(row.bind(7, result.error));
    AF_TRY_VOID(row.bind(8, result.detail));
    AF_TRY_VOID(row.bind(9, result.trace_id));
    AF_TRY_VOID(row.step());
  }
  return transaction.commit();
}

core::Result<EvalReport> EvalStore::load(std::string_view eval_id) {
  AF_TRY(auto run, db_->prepare(
                       "SELECT eval_id, suite, model, adapter, label, split, seed,"
                       " framework_version, framework_commit, total, failures, mean_score,"
                       " ci_low, ci_high FROM eval_runs WHERE eval_id = ?1"));
  AF_TRY_VOID(run.bind(1, eval_id));
  AF_TRY(const bool found, run.step());
  if (!found) {
    return core::fail(core::ErrorCode::NotFound, "no eval run with this id",
                      std::string{eval_id});
  }

  EvalReport report;
  report.eval_id = run.column_text(0);
  report.suite = run.column_text(1);
  report.model = run.column_text(2);
  report.adapter = run.column_text(3);
  report.label = run.column_text(4);
  report.split = run.column_text(5);
  report.seed = static_cast<std::uint64_t>(run.column_int64(6));
  report.framework_version = run.column_text(7);
  report.framework_commit = run.column_text(8);
  report.total = static_cast<std::size_t>(run.column_int64(9));
  report.failures = static_cast<std::size_t>(run.column_int64(10));
  report.mean_score = run.column_double(11);
  report.ci = {run.column_double(12), run.column_double(13)};

  AF_TRY(auto rows, db_->prepare(
                        "SELECT task_id, split, score, verified, ok, error, detail, trace_id"
                        " FROM eval_results WHERE eval_id = ?1 ORDER BY task_id"));
  AF_TRY_VOID(rows.bind(1, eval_id));
  while (true) {
    AF_TRY(const bool row, rows.step());
    if (!row) break;
    InstanceResult result;
    result.task_id = rows.column_text(0);
    result.split = split_from_string(rows.column_text(1));
    result.score = rows.column_double(2);
    result.verified = rows.column_int64(3) != 0;
    result.ok = rows.column_int64(4) != 0;
    result.error = rows.column_text(5);
    result.detail = rows.column_text(6);
    result.trace_id = rows.column_text(7);
    report.results.push_back(std::move(result));
  }
  return report;
}

core::Result<std::vector<EvalRunInfo>> EvalStore::list(std::string_view suite) {
  std::string sql =
      "SELECT eval_id, suite, model, adapter, label, split, seed, total, mean_score,"
      " ci_low, ci_high, baseline, created_at FROM eval_runs";
  if (!suite.empty()) sql += " WHERE suite = ?1";
  sql += " ORDER BY created_at, eval_id";

  AF_TRY(auto stmt, db_->prepare(sql));
  if (!suite.empty()) AF_TRY_VOID(stmt.bind(1, suite));

  std::vector<EvalRunInfo> out;
  while (true) {
    AF_TRY(const bool row, stmt.step());
    if (!row) break;
    EvalRunInfo info;
    info.eval_id = stmt.column_text(0);
    info.suite = stmt.column_text(1);
    info.model = stmt.column_text(2);
    info.adapter = stmt.column_text(3);
    info.label = stmt.column_text(4);
    info.split = stmt.column_text(5);
    info.seed = static_cast<std::uint64_t>(stmt.column_int64(6));
    info.total = static_cast<std::size_t>(stmt.column_int64(7));
    info.mean_score = stmt.column_double(8);
    info.ci = {stmt.column_double(9), stmt.column_double(10)};
    info.baseline = stmt.column_int64(11) != 0;
    info.created_at = stmt.column_text(12);
    out.push_back(std::move(info));
  }
  return out;
}

core::Result<void> EvalStore::set_baseline(std::string_view eval_id) {
  AF_TRY(auto find, db_->prepare("SELECT suite FROM eval_runs WHERE eval_id = ?1"));
  AF_TRY_VOID(find.bind(1, eval_id));
  AF_TRY(const bool found, find.step());
  if (!found) {
    return core::fail(core::ErrorCode::NotFound, "no eval run with this id",
                      std::string{eval_id});
  }
  const std::string suite = find.column_text(0);

  AF_TRY(auto transaction, store::Transaction::begin(*db_));
  AF_TRY(auto clear, db_->prepare("UPDATE eval_runs SET baseline = 0 WHERE suite = ?1"));
  AF_TRY_VOID(clear.bind(1, suite));
  AF_TRY_VOID(clear.step());
  AF_TRY(auto mark, db_->prepare("UPDATE eval_runs SET baseline = 1 WHERE eval_id = ?1"));
  AF_TRY_VOID(mark.bind(1, eval_id));
  AF_TRY_VOID(mark.step());
  return transaction.commit();
}

core::Result<EvalReport> EvalStore::baseline(std::string_view suite) {
  AF_TRY(auto stmt,
         db_->prepare("SELECT eval_id FROM eval_runs WHERE suite = ?1 AND baseline = 1"));
  AF_TRY_VOID(stmt.bind(1, suite));
  AF_TRY(const bool found, stmt.step());
  if (!found) {
    return core::fail(core::ErrorCode::NotFound, "no baseline recorded for this suite",
                      std::string{suite});
  }
  return load(stmt.column_text(0));
}

}
