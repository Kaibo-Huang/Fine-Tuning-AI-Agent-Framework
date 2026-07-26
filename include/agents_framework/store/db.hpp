#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"

struct sqlite3;
struct sqlite3_stmt;

namespace agents_framework::store {

class Statement {
 public:
  Statement() = default;
  Statement(Statement&& other) noexcept;
  Statement& operator=(Statement&& other) noexcept;
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;
  ~Statement();

  core::Result<void> bind(int index, std::int64_t value);
  core::Result<void> bind(int index, double value);
  core::Result<void> bind(int index, std::string_view value);
  core::Result<void> bind_blob(int index, std::span<const std::byte> value);
  core::Result<void> bind_null(int index);

  core::Result<bool> step();
  core::Result<void> reset();

  [[nodiscard]] int column_count() const;
  [[nodiscard]] bool column_is_null(int column) const;
  [[nodiscard]] std::int64_t column_int64(int column) const;
  [[nodiscard]] double column_double(int column) const;
  [[nodiscard]] std::string column_text(int column) const;
  [[nodiscard]] std::vector<std::byte> column_blob(int column) const;

 private:
  friend class Db;
  Statement(sqlite3* db, sqlite3_stmt* stmt) : db_(db), stmt_(stmt) {}

  sqlite3* db_{nullptr};
  sqlite3_stmt* stmt_{nullptr};
};

class Db {
 public:
  Db() = default;
  Db(Db&& other) noexcept;
  Db& operator=(Db&& other) noexcept;
  Db(const Db&) = delete;
  Db& operator=(const Db&) = delete;
  ~Db();

  static core::Result<Db> open(const std::string& path);
  static core::Result<Db> open_memory();

  [[nodiscard]] bool is_open() const noexcept { return db_ != nullptr; }

  core::Result<void> exec(std::string_view sql);
  core::Result<Statement> prepare(std::string_view sql);

  [[nodiscard]] std::int64_t last_insert_rowid() const;
  [[nodiscard]] std::int64_t changes() const;

  [[nodiscard]] sqlite3* handle() const noexcept { return db_; }

 private:
  explicit Db(sqlite3* db) : db_(db) {}

  sqlite3* db_{nullptr};
};

class Transaction {
 public:
  static core::Result<Transaction> begin(Db& db);

  Transaction(Transaction&& other) noexcept : db_(other.db_) { other.db_ = nullptr; }
  Transaction& operator=(Transaction&&) = delete;
  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;
  ~Transaction();

  core::Result<void> commit();

 private:
  explicit Transaction(Db* db) : db_(db) {}

  Db* db_{nullptr};
};

}
