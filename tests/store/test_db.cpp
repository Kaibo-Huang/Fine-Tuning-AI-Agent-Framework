#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/store/db.hpp"

namespace core = agents_framework::core;
namespace store = agents_framework::store;

TEST_CASE("an in-memory database opens and executes SQL", "[store][db]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(db->is_open());
  REQUIRE(db->exec("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"));
  REQUIRE(db->exec("INSERT INTO items (name) VALUES ('alpha'), ('beta')"));
  REQUIRE(db->changes() == 2);
  REQUIRE(db->last_insert_rowid() == 2);
}

TEST_CASE("invalid SQL fails with an Io error and the SQLite message", "[store][db]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  const auto result = db->exec("SELEKT * FROM nowhere");
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Io);
  REQUIRE(!result.error().context.empty());
}

TEST_CASE("prepared statements bind, step, and read every column type", "[store][db]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(db->exec("CREATE TABLE t (i INTEGER, r REAL, s TEXT, b BLOB, n TEXT)"));

  auto insert = db->prepare("INSERT INTO t (i, r, s, b, n) VALUES (?1, ?2, ?3, ?4, ?5)");
  REQUIRE(insert);
  const std::vector<std::byte> blob{std::byte{0x01}, std::byte{0x00}, std::byte{0xFF}};
  REQUIRE(insert->bind(1, std::int64_t{42}));
  REQUIRE(insert->bind(2, 2.5));
  REQUIRE(insert->bind(3, std::string_view{"hello"}));
  REQUIRE(insert->bind_blob(4, blob));
  REQUIRE(insert->bind_null(5));
  auto stepped = insert->step();
  REQUIRE(stepped);
  REQUIRE(*stepped == false);

  auto select = db->prepare("SELECT i, r, s, b, n FROM t");
  REQUIRE(select);
  auto row = select->step();
  REQUIRE(row);
  REQUIRE(*row == true);
  REQUIRE(select->column_count() == 5);
  REQUIRE(select->column_int64(0) == 42);
  REQUIRE(select->column_double(1) == 2.5);
  REQUIRE(select->column_text(2) == "hello");
  REQUIRE(select->column_blob(3) == blob);
  REQUIRE(select->column_is_null(4));
  auto done = select->step();
  REQUIRE(done);
  REQUIRE(*done == false);
}

TEST_CASE("statements reset and rebind for reuse", "[store][db]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(db->exec("CREATE TABLE t (v TEXT)"));

  auto insert = db->prepare("INSERT INTO t (v) VALUES (?1)");
  REQUIRE(insert);
  for (const std::string_view value : {"one", "two", "three"}) {
    REQUIRE(insert->bind(1, value));
    REQUIRE(insert->step());
    REQUIRE(insert->reset());
  }

  auto count = db->prepare("SELECT COUNT(*) FROM t");
  REQUIRE(count);
  REQUIRE(count->step());
  REQUIRE(count->column_int64(0) == 3);
}

TEST_CASE("a committed transaction persists its writes", "[store][db]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(db->exec("CREATE TABLE t (v TEXT)"));

  {
    auto transaction = store::Transaction::begin(*db);
    REQUIRE(transaction);
    REQUIRE(db->exec("INSERT INTO t (v) VALUES ('kept')"));
    REQUIRE(transaction->commit());
  }

  auto count = db->prepare("SELECT COUNT(*) FROM t");
  REQUIRE(count);
  REQUIRE(count->step());
  REQUIRE(count->column_int64(0) == 1);
}

TEST_CASE("an uncommitted transaction rolls back on destruction", "[store][db]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(db->exec("CREATE TABLE t (v TEXT)"));

  {
    auto transaction = store::Transaction::begin(*db);
    REQUIRE(transaction);
    REQUIRE(db->exec("INSERT INTO t (v) VALUES ('dropped')"));
  }

  auto count = db->prepare("SELECT COUNT(*) FROM t");
  REQUIRE(count);
  REQUIRE(count->step());
  REQUIRE(count->column_int64(0) == 0);
}

TEST_CASE("a database is movable and closes cleanly", "[store][db]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  store::Db moved = std::move(*db);
  REQUIRE(moved.is_open());
  REQUIRE(moved.exec("CREATE TABLE t (v TEXT)"));
}
