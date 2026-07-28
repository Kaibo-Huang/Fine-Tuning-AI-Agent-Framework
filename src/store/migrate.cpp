#include "agents_framework/store/migrate.hpp"

#include <string>
#include <utility>

namespace agents_framework::store {

namespace {

constexpr std::string_view kMigrationsTable =
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "  component TEXT NOT NULL,"
    "  version INTEGER NOT NULL,"
    "  applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
    "  PRIMARY KEY (component, version)"
    ")";

core::Error annotate(core::Error error, std::string_view component, int version) {
  error.context = "migration v" + std::to_string(version) + " for '" + std::string{component} +
                  "'" + (error.context.empty() ? "" : ": " + error.context);
  return error;
}

}

core::Result<int> schema_version(Db& db, std::string_view component) {
  AF_TRY_VOID(db.exec(kMigrationsTable));
  AF_TRY(auto stmt, db.prepare("SELECT COALESCE(MAX(version), 0) FROM schema_migrations "
                               "WHERE component = ?1"));
  AF_TRY_VOID(stmt.bind(1, component));
  AF_TRY(const bool row, stmt.step());
  if (!row) {
    return core::fail(core::ErrorCode::Io, "schema version query returned no rows");
  }
  return static_cast<int>(stmt.column_int64(0));
}

core::Result<int> migrate(Db& db, std::string_view component,
                          std::span<const Migration> migrations) {
  int previous = 0;
  for (const Migration& migration : migrations) {
    if (migration.version <= previous) {
      return core::fail(core::ErrorCode::Invalid,
                        "migrations must have strictly increasing versions starting at 1",
                        "component '" + std::string{component} + "'");
    }
    previous = migration.version;
  }

  AF_TRY(int version, schema_version(db, component));
  for (const Migration& migration : migrations) {
    if (migration.version <= version) continue;

    AF_TRY(auto transaction, Transaction::begin(db));
    if (auto applied = db.exec(migration.sql); !applied) {
      return std::unexpected(annotate(std::move(applied).error(), component, migration.version));
    }
    AF_TRY(auto record,
           db.prepare("INSERT INTO schema_migrations (component, version) VALUES (?1, ?2)"));
    AF_TRY_VOID(record.bind(1, component));
    AF_TRY_VOID(record.bind(2, static_cast<std::int64_t>(migration.version)));
    AF_TRY_VOID(record.step());
    if (auto committed = transaction.commit(); !committed) {
      return std::unexpected(annotate(std::move(committed).error(), component, migration.version));
    }
    version = migration.version;
  }
  return version;
}

}
