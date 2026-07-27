#include "agents_framework/store/db.hpp"

#include <sqlite3.h>

#include <utility>

namespace agents_framework::store {

namespace {

std::unexpected<core::Error> fail_db(sqlite3* db, std::string message) {
  std::string context;
  if (db != nullptr) context = sqlite3_errmsg(db);
  return std::unexpected(core::Error{core::ErrorCode::Io, std::move(message), std::move(context)});
}

}

Statement::Statement(Statement&& other) noexcept : db_(other.db_), stmt_(other.stmt_) {
  other.db_ = nullptr;
  other.stmt_ = nullptr;
}

Statement& Statement::operator=(Statement&& other) noexcept {
  if (this != &other) {
    if (stmt_ != nullptr) sqlite3_finalize(stmt_);
    db_ = std::exchange(other.db_, nullptr);
    stmt_ = std::exchange(other.stmt_, nullptr);
  }
  return *this;
}

Statement::~Statement() {
  if (stmt_ != nullptr) sqlite3_finalize(stmt_);
}

core::Result<void> Statement::bind(int index, std::int64_t value) {
  if (sqlite3_bind_int64(stmt_, index, value) != SQLITE_OK) {
    return fail_db(db_, "failed to bind integer parameter " + std::to_string(index));
  }
  return {};
}

core::Result<void> Statement::bind(int index, double value) {
  if (sqlite3_bind_double(stmt_, index, value) != SQLITE_OK) {
    return fail_db(db_, "failed to bind real parameter " + std::to_string(index));
  }
  return {};
}

core::Result<void> Statement::bind(int index, std::string_view value) {
  if (sqlite3_bind_text(stmt_, index, value.data(), static_cast<int>(value.size()),
                        SQLITE_TRANSIENT) != SQLITE_OK) {
    return fail_db(db_, "failed to bind text parameter " + std::to_string(index));
  }
  return {};
}

core::Result<void> Statement::bind_blob(int index, std::span<const std::byte> value) {
  if (sqlite3_bind_blob(stmt_, index, value.data(), static_cast<int>(value.size()),
                        SQLITE_TRANSIENT) != SQLITE_OK) {
    return fail_db(db_, "failed to bind blob parameter " + std::to_string(index));
  }
  return {};
}

core::Result<void> Statement::bind_null(int index) {
  if (sqlite3_bind_null(stmt_, index) != SQLITE_OK) {
    return fail_db(db_, "failed to bind null parameter " + std::to_string(index));
  }
  return {};
}

core::Result<bool> Statement::step() {
  const int rc = sqlite3_step(stmt_);
  if (rc == SQLITE_ROW) return true;
  if (rc == SQLITE_DONE) return false;
  return fail_db(db_, "failed to step statement");
}

core::Result<void> Statement::reset() {
  if (sqlite3_reset(stmt_) != SQLITE_OK) {
    return fail_db(db_, "failed to reset statement");
  }
  sqlite3_clear_bindings(stmt_);
  return {};
}

int Statement::column_count() const { return sqlite3_column_count(stmt_); }

bool Statement::column_is_null(int column) const {
  return sqlite3_column_type(stmt_, column) == SQLITE_NULL;
}

std::int64_t Statement::column_int64(int column) const {
  return sqlite3_column_int64(stmt_, column);
}

double Statement::column_double(int column) const { return sqlite3_column_double(stmt_, column); }

std::string Statement::column_text(int column) const {
  const unsigned char* text = sqlite3_column_text(stmt_, column);
  if (text == nullptr) return {};
  const int size = sqlite3_column_bytes(stmt_, column);
  return std::string{reinterpret_cast<const char*>(text), static_cast<std::size_t>(size)};
}

std::vector<std::byte> Statement::column_blob(int column) const {
  const void* data = sqlite3_column_blob(stmt_, column);
  if (data == nullptr) return {};
  const int size = sqlite3_column_bytes(stmt_, column);
  const auto* bytes = static_cast<const std::byte*>(data);
  return std::vector<std::byte>{bytes, bytes + size};
}

Db::Db(Db&& other) noexcept : db_(std::exchange(other.db_, nullptr)) {}

Db& Db::operator=(Db&& other) noexcept {
  if (this != &other) {
    if (db_ != nullptr) sqlite3_close(db_);
    db_ = std::exchange(other.db_, nullptr);
  }
  return *this;
}

Db::~Db() {
  if (db_ != nullptr) sqlite3_close(db_);
}

core::Result<Db> Db::open(const std::string& path) {
  sqlite3* raw = nullptr;
  const int rc = sqlite3_open_v2(
      path.c_str(), &raw, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
      nullptr);
  if (rc != SQLITE_OK) {
    std::string context = raw != nullptr ? sqlite3_errmsg(raw) : sqlite3_errstr(rc);
    if (raw != nullptr) sqlite3_close(raw);
    return core::fail(core::ErrorCode::Io, "failed to open database '" + path + "'",
                      std::move(context));
  }

  Db db{raw};
  AF_TRY_VOID(db.exec("PRAGMA journal_mode=WAL;"
                      "PRAGMA synchronous=NORMAL;"
                      "PRAGMA busy_timeout=5000;"
                      "PRAGMA foreign_keys=ON;"));
  return db;
}

core::Result<Db> Db::open_memory() { return open(":memory:"); }

core::Result<void> Db::exec(std::string_view sql) {
  char* message = nullptr;
  const int rc = sqlite3_exec(db_, std::string{sql}.c_str(), nullptr, nullptr, &message);
  if (rc != SQLITE_OK) {
    std::string context = message != nullptr ? message : sqlite3_errstr(rc);
    sqlite3_free(message);
    return core::fail(core::ErrorCode::Io, "failed to execute SQL", std::move(context));
  }
  return {};
}

core::Result<Statement> Db::prepare(std::string_view sql) {
  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()), &stmt, nullptr) !=
      SQLITE_OK) {
    return fail_db(db_, "failed to prepare statement");
  }
  return Statement{db_, stmt};
}

std::int64_t Db::last_insert_rowid() const { return sqlite3_last_insert_rowid(db_); }

std::int64_t Db::changes() const { return sqlite3_changes64(db_); }

core::Result<Transaction> Transaction::begin(Db& db) {
  AF_TRY_VOID(db.exec("BEGIN IMMEDIATE"));
  return Transaction{&db};
}

Transaction::~Transaction() {
  if (db_ != nullptr) {
    (void)db_->exec("ROLLBACK");
  }
}

core::Result<void> Transaction::commit() {
  if (db_ == nullptr) {
    return core::fail(core::ErrorCode::Invalid, "transaction already finished");
  }
  Db* db = std::exchange(db_, nullptr);
  return db->exec("COMMIT");
}

}
