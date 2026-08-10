#include "agents_framework/trace/trace_store.hpp"

#include <array>
#include <exception>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "agents_framework/store/migrate.hpp"

namespace agents_framework::trace {

namespace {

constexpr std::array<store::Migration, 1> kMigrations{{
    {1,
     "CREATE TABLE IF NOT EXISTS traces ("
     "  trace_id TEXT PRIMARY KEY,"
     "  run_id TEXT NOT NULL,"
     "  suite TEXT NOT NULL DEFAULT '',"
     "  task_id TEXT NOT NULL DEFAULT '',"
     "  model TEXT NOT NULL DEFAULT '',"
     "  transcript TEXT NOT NULL DEFAULT '[]',"
     "  node_runs TEXT NOT NULL DEFAULT '[]',"
     "  final_output TEXT NOT NULL DEFAULT '',"
     "  score REAL,"
     "  verified INTEGER NOT NULL DEFAULT 0,"
     "  metadata TEXT NOT NULL DEFAULT '{}',"
     "  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
     ");"
     "CREATE INDEX IF NOT EXISTS traces_by_run ON traces(run_id);"
     "CREATE INDEX IF NOT EXISTS traces_by_suite_task ON traces(suite, task_id);"},
}};

struct QueryParts {
  std::string where;
  std::vector<std::string> texts;
  std::optional<std::int64_t> verified;
  std::optional<double> min_score;
  std::optional<std::int64_t> limit;
};

QueryParts build_query(const TraceQuery& query) {
  QueryParts parts;
  const auto add = [&parts](std::string_view clause) {
    parts.where += parts.where.empty() ? " WHERE " : " AND ";
    parts.where += clause;
  };
  if (!query.run_id.empty()) {
    add("run_id = ?");
    parts.texts.push_back(query.run_id);
  }
  if (!query.suite.empty()) {
    add("suite = ?");
    parts.texts.push_back(query.suite);
  }
  if (!query.task_id.empty()) {
    add("task_id = ?");
    parts.texts.push_back(query.task_id);
  }
  if (query.verified) {
    add("verified = ?");
    parts.verified = *query.verified ? 1 : 0;
  }
  if (query.min_score) {
    add("score IS NOT NULL AND score >= ?");
    parts.min_score = *query.min_score;
  }
  return parts;
}

core::Result<store::Statement> prepare_query(store::Db& db, std::string_view columns,
                                             const TraceQuery& query) {
  QueryParts parts = build_query(query);
  std::string sql = "SELECT " + std::string{columns} + " FROM traces" + parts.where +
                    " ORDER BY created_at, trace_id";
  if (query.limit > 0) sql += " LIMIT ?";

  AF_TRY(auto stmt, db.prepare(sql));
  int index = 0;
  for (const std::string& text : parts.texts) {
    AF_TRY_VOID(stmt.bind(++index, text));
  }
  if (parts.verified) AF_TRY_VOID(stmt.bind(++index, *parts.verified));
  if (parts.min_score) AF_TRY_VOID(stmt.bind(++index, *parts.min_score));
  if (query.limit > 0) AF_TRY_VOID(stmt.bind(++index, static_cast<std::int64_t>(query.limit)));
  return stmt;
}

core::Result<nlohmann::json> parse_column(const std::string& text, std::string_view what) {
  nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
  if (j.is_discarded()) {
    return core::fail(core::ErrorCode::Parse, "stored trace column is not valid JSON",
                      std::string{what});
  }
  return j;
}

}

core::Result<TraceStore> TraceStore::open(std::shared_ptr<store::Db> db) {
  if (!db || !db->is_open()) {
    return core::fail(core::ErrorCode::Invalid, "TraceStore requires an open database");
  }
  AF_TRY_VOID(store::migrate(*db, "traces", kMigrations));
  return TraceStore(std::move(db));
}

core::Result<void> TraceStore::save(const Trace& trace) {
  if (trace.trace_id.empty()) {
    return core::fail(core::ErrorCode::Invalid, "trace_id must not be empty");
  }
  AF_TRY(auto stmt,
         db_->prepare("INSERT INTO traces (trace_id, run_id, suite, task_id, model, transcript,"
                      "  node_runs, final_output, score, verified, metadata)"
                      " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"
                      " ON CONFLICT(trace_id) DO UPDATE SET"
                      "  run_id = excluded.run_id, suite = excluded.suite,"
                      "  task_id = excluded.task_id, model = excluded.model,"
                      "  transcript = excluded.transcript, node_runs = excluded.node_runs,"
                      "  final_output = excluded.final_output, score = excluded.score,"
                      "  verified = excluded.verified, metadata = excluded.metadata"));
  AF_TRY_VOID(stmt.bind(1, trace.trace_id));
  AF_TRY_VOID(stmt.bind(2, trace.run_id));
  AF_TRY_VOID(stmt.bind(3, trace.suite));
  AF_TRY_VOID(stmt.bind(4, trace.task_id));
  AF_TRY_VOID(stmt.bind(5, trace.model));
  AF_TRY_VOID(stmt.bind(6, nlohmann::json(trace.transcript).dump()));
  AF_TRY_VOID(stmt.bind(7, nlohmann::json(trace.node_runs).dump()));
  AF_TRY_VOID(stmt.bind(8, trace.final_output));
  if (trace.score) {
    AF_TRY_VOID(stmt.bind(9, *trace.score));
  } else {
    AF_TRY_VOID(stmt.bind_null(9));
  }
  AF_TRY_VOID(stmt.bind(10, static_cast<std::int64_t>(trace.verified ? 1 : 0)));
  AF_TRY_VOID(stmt.bind(11, trace.metadata.dump()));
  AF_TRY_VOID(stmt.step());
  return {};
}

core::Result<Trace> TraceStore::load(std::string_view trace_id) {
  AF_TRY(auto stmt, db_->prepare("SELECT trace_id, run_id, suite, task_id, model, transcript,"
                                 " node_runs, final_output, score, verified, metadata"
                                 " FROM traces WHERE trace_id = ?1"));
  AF_TRY_VOID(stmt.bind(1, trace_id));
  AF_TRY(const bool row, stmt.step());
  if (!row) {
    return core::fail(core::ErrorCode::NotFound, "no trace with this id", std::string{trace_id});
  }

  Trace trace;
  trace.trace_id = stmt.column_text(0);
  trace.run_id = stmt.column_text(1);
  trace.suite = stmt.column_text(2);
  trace.task_id = stmt.column_text(3);
  trace.model = stmt.column_text(4);
  AF_TRY(const auto transcript, parse_column(stmt.column_text(5), "transcript"));
  AF_TRY(const auto node_runs, parse_column(stmt.column_text(6), "node_runs"));
  try {
    trace.transcript = transcript.get<std::vector<llm::Message>>();
    trace.node_runs = node_runs.get<std::vector<NodeRun>>();
  } catch (const std::exception& error) {
    return core::fail(core::ErrorCode::Parse, error.what(), "trace '" + trace.trace_id + "'");
  }
  trace.final_output = stmt.column_text(7);
  trace.score = stmt.column_is_null(8) ? std::nullopt : std::optional{stmt.column_double(8)};
  trace.verified = stmt.column_int64(9) != 0;
  AF_TRY(trace.metadata, parse_column(stmt.column_text(10), "metadata"));
  return trace;
}

core::Result<std::vector<TraceInfo>> TraceStore::list(const TraceQuery& query) {
  AF_TRY(auto stmt, prepare_query(*db_, "trace_id, run_id, suite, task_id, score, verified,"
                                        " created_at",
                                  query));
  std::vector<TraceInfo> out;
  while (true) {
    AF_TRY(const bool row, stmt.step());
    if (!row) break;
    TraceInfo info;
    info.trace_id = stmt.column_text(0);
    info.run_id = stmt.column_text(1);
    info.suite = stmt.column_text(2);
    info.task_id = stmt.column_text(3);
    info.score = stmt.column_is_null(4) ? std::nullopt : std::optional{stmt.column_double(4)};
    info.verified = stmt.column_int64(5) != 0;
    info.created_at = stmt.column_text(6);
    out.push_back(std::move(info));
  }
  return out;
}

core::Result<std::vector<Trace>> TraceStore::select(const TraceQuery& query) {
  AF_TRY(const auto infos, list(query));
  std::vector<Trace> out;
  out.reserve(infos.size());
  for (const TraceInfo& info : infos) {
    AF_TRY(auto trace, load(info.trace_id));
    out.push_back(std::move(trace));
  }
  return out;
}

}
