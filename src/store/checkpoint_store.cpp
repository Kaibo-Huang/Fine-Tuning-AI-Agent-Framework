#include "agents_framework/store/checkpoint_store.hpp"

#include <array>
#include <utility>

#include <nlohmann/json.hpp>

#include "agents_framework/store/migrate.hpp"

namespace agents_framework::store {

namespace {

constexpr std::array<Migration, 1> kMigrations{{
    {1,
     "CREATE TABLE checkpoints ("
     "  run_id TEXT NOT NULL,"
     "  step INTEGER NOT NULL,"
     "  status TEXT NOT NULL,"
     "  state TEXT NOT NULL,"
     "  next_nodes TEXT NOT NULL,"
     "  parent_run_id TEXT,"
     "  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
     "  PRIMARY KEY (run_id, step)"
     ")"},
}};

core::Result<std::vector<std::string>> parse_next_nodes(const std::string& text) {
  const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
  if (j.is_discarded() || !j.is_array()) {
    return core::fail(core::ErrorCode::Parse, "stored next_nodes is not a JSON array");
  }
  std::vector<std::string> nodes;
  nodes.reserve(j.size());
  for (const auto& entry : j) {
    if (!entry.is_string()) {
      return core::fail(core::ErrorCode::Parse, "stored next_nodes entry is not a string");
    }
    nodes.push_back(entry.get<std::string>());
  }
  return nodes;
}

core::Result<graph::Checkpoint> read_checkpoint(Statement& stmt, std::string_view run_id) {
  graph::Checkpoint checkpoint;
  checkpoint.run_id = std::string{run_id};
  checkpoint.step = static_cast<std::uint64_t>(stmt.column_int64(0));
  checkpoint.status = graph::checkpoint_status_from_string(stmt.column_text(1));
  checkpoint.state = stmt.column_text(2);
  AF_TRY(checkpoint.next_nodes, parse_next_nodes(stmt.column_text(3)));
  return checkpoint;
}

}

core::Result<CheckpointStore> CheckpointStore::open(std::shared_ptr<Db> db) {
  if (!db || !db->is_open()) {
    return core::fail(core::ErrorCode::Invalid, "CheckpointStore requires an open database");
  }
  AF_TRY_VOID(migrate(*db, "checkpoints", kMigrations));
  return CheckpointStore{std::move(db)};
}

core::Result<void> CheckpointStore::save(const graph::Checkpoint& checkpoint) {
  if (checkpoint.run_id.empty()) {
    return core::fail(core::ErrorCode::Invalid, "checkpoint run_id must not be empty");
  }
  AF_TRY(auto stmt,
         db_->prepare("INSERT INTO checkpoints (run_id, step, status, state, next_nodes) "
                      "VALUES (?1, ?2, ?3, ?4, ?5) "
                      "ON CONFLICT (run_id, step) DO UPDATE SET "
                      "status = excluded.status, state = excluded.state, "
                      "next_nodes = excluded.next_nodes"));
  AF_TRY_VOID(stmt.bind(1, checkpoint.run_id));
  AF_TRY_VOID(stmt.bind(2, static_cast<std::int64_t>(checkpoint.step)));
  AF_TRY_VOID(stmt.bind(3, checkpoint_status_name(checkpoint.status)));
  AF_TRY_VOID(stmt.bind(4, checkpoint.state));
  AF_TRY_VOID(stmt.bind(5, nlohmann::json(checkpoint.next_nodes).dump()));
  AF_TRY_VOID(stmt.step());
  return {};
}

core::Result<std::vector<CheckpointInfo>> CheckpointStore::list(std::string_view run_id) {
  AF_TRY(auto stmt, db_->prepare("SELECT step, status, created_at FROM checkpoints "
                                 "WHERE run_id = ?1 ORDER BY step"));
  AF_TRY_VOID(stmt.bind(1, run_id));

  std::vector<CheckpointInfo> checkpoints;
  while (true) {
    AF_TRY(const bool row, stmt.step());
    if (!row) break;
    CheckpointInfo info;
    info.step = static_cast<std::uint64_t>(stmt.column_int64(0));
    info.status = graph::checkpoint_status_from_string(stmt.column_text(1));
    info.created_at = stmt.column_text(2);
    checkpoints.push_back(std::move(info));
  }
  return checkpoints;
}

core::Result<graph::Checkpoint> CheckpointStore::load(std::string_view run_id,
                                                      std::uint64_t step) {
  AF_TRY(auto stmt, db_->prepare("SELECT step, status, state, next_nodes FROM checkpoints "
                                 "WHERE run_id = ?1 AND step = ?2"));
  AF_TRY_VOID(stmt.bind(1, run_id));
  AF_TRY_VOID(stmt.bind(2, static_cast<std::int64_t>(step)));
  AF_TRY(const bool row, stmt.step());
  if (!row) {
    return core::fail(core::ErrorCode::NotFound, "no checkpoint found",
                      "run '" + std::string{run_id} + "' step " + std::to_string(step));
  }
  return read_checkpoint(stmt, run_id);
}

core::Result<graph::Checkpoint> CheckpointStore::latest(std::string_view run_id) {
  AF_TRY(auto stmt, db_->prepare("SELECT step, status, state, next_nodes FROM checkpoints "
                                 "WHERE run_id = ?1 ORDER BY step DESC LIMIT 1"));
  AF_TRY_VOID(stmt.bind(1, run_id));
  AF_TRY(const bool row, stmt.step());
  if (!row) {
    return core::fail(core::ErrorCode::NotFound, "no checkpoints found for run",
                      std::string{run_id});
  }
  return read_checkpoint(stmt, run_id);
}

core::Result<graph::Checkpoint> CheckpointStore::fork(std::string_view run_id,
                                                      std::uint64_t step,
                                                      std::string new_run_id) {
  if (new_run_id.empty()) {
    return core::fail(core::ErrorCode::Invalid, "fork requires a non-empty new run id");
  }
  AF_TRY(auto existing, db_->prepare("SELECT COUNT(*) FROM checkpoints WHERE run_id = ?1"));
  AF_TRY_VOID(existing.bind(1, new_run_id));
  AF_TRY_VOID(existing.step());
  if (existing.column_int64(0) != 0) {
    return core::fail(core::ErrorCode::Invalid, "fork target run id already exists", new_run_id);
  }

  AF_TRY(auto checkpoint, load(run_id, step));
  checkpoint.run_id = new_run_id;

  AF_TRY(auto stmt,
         db_->prepare("INSERT INTO checkpoints (run_id, step, status, state, next_nodes, "
                      "parent_run_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"));
  AF_TRY_VOID(stmt.bind(1, checkpoint.run_id));
  AF_TRY_VOID(stmt.bind(2, static_cast<std::int64_t>(checkpoint.step)));
  AF_TRY_VOID(stmt.bind(3, checkpoint_status_name(checkpoint.status)));
  AF_TRY_VOID(stmt.bind(4, checkpoint.state));
  AF_TRY_VOID(stmt.bind(5, nlohmann::json(checkpoint.next_nodes).dump()));
  AF_TRY_VOID(stmt.bind(6, run_id));
  AF_TRY_VOID(stmt.step());
  return checkpoint;
}

core::Result<std::vector<RunInfo>> CheckpointStore::runs() {
  AF_TRY(auto stmt,
         db_->prepare("SELECT c.run_id, c.step, c.status, "
                      "  (SELECT parent_run_id FROM checkpoints p WHERE p.run_id = c.run_id "
                      "   AND p.parent_run_id IS NOT NULL LIMIT 1) "
                      "FROM checkpoints c "
                      "JOIN (SELECT run_id, MAX(step) AS latest FROM checkpoints "
                      "      GROUP BY run_id) m "
                      "ON c.run_id = m.run_id AND c.step = m.latest "
                      "ORDER BY c.run_id"));

  std::vector<RunInfo> runs;
  while (true) {
    AF_TRY(const bool row, stmt.step());
    if (!row) break;
    RunInfo info;
    info.run_id = stmt.column_text(0);
    info.latest_step = static_cast<std::uint64_t>(stmt.column_int64(1));
    info.status = graph::checkpoint_status_from_string(stmt.column_text(2));
    if (!stmt.column_is_null(3)) {
      info.parent_run_id = stmt.column_text(3);
    }
    runs.push_back(std::move(info));
  }
  return runs;
}

}
