#include <catch2/catch_test_macros.hpp>

#include <array>

#include "agents_framework/core/result.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/store/migrate.hpp"

namespace core = agents_framework::core;
namespace store = agents_framework::store;

namespace {

constexpr std::array<store::Migration, 2> kMigrations{{
    {1, "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)"},
    {2, "ALTER TABLE notes ADD COLUMN starred INTEGER NOT NULL DEFAULT 0"},
}};

}

TEST_CASE("migrations apply in order and record the schema version", "[store][migrate]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);

  const auto version = store::migrate(*db, "notes", kMigrations);
  REQUIRE(version);
  REQUIRE(*version == 2);
  REQUIRE(db->exec("INSERT INTO notes (body, starred) VALUES ('hi', 1)"));

  const auto recorded = store::schema_version(*db, "notes");
  REQUIRE(recorded);
  REQUIRE(*recorded == 2);
}

TEST_CASE("migrating twice is a no-op", "[store][migrate]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(store::migrate(*db, "notes", kMigrations));
  const auto again = store::migrate(*db, "notes", kMigrations);
  REQUIRE(again);
  REQUIRE(*again == 2);
}

TEST_CASE("only migrations above the current version run", "[store][migrate]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(store::migrate(*db, "notes", std::span{kMigrations}.first(1)));
  const auto version = store::migrate(*db, "notes", kMigrations);
  REQUIRE(version);
  REQUIRE(*version == 2);
  REQUIRE(db->exec("INSERT INTO notes (body, starred) VALUES ('hi', 1)"));
}

TEST_CASE("out-of-order migration lists are rejected", "[store][migrate]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  const std::array<store::Migration, 2> unsorted{{
      {2, "CREATE TABLE a (id INTEGER)"},
      {1, "CREATE TABLE b (id INTEGER)"},
  }};
  const auto result = store::migrate(*db, "bad", unsorted);
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("a failing migration rolls back and leaves the version unchanged", "[store][migrate]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  REQUIRE(store::migrate(*db, "notes", std::span{kMigrations}.first(1)));

  const std::array<store::Migration, 2> broken{{
      kMigrations[0],
      {2, "ALTER TABLE missing ADD COLUMN nope INTEGER"},
  }};
  const auto result = store::migrate(*db, "notes", broken);
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Io);
  REQUIRE(result.error().context.find("migration v2") != std::string::npos);

  const auto version = store::schema_version(*db, "notes");
  REQUIRE(version);
  REQUIRE(*version == 1);
}

TEST_CASE("components track schema versions independently", "[store][migrate]") {
  auto db = store::Db::open_memory();
  REQUIRE(db);
  constexpr std::array<store::Migration, 1> other{{
      {1, "CREATE TABLE tags (id INTEGER PRIMARY KEY, label TEXT)"},
  }};
  REQUIRE(store::migrate(*db, "notes", kMigrations));
  REQUIRE(store::migrate(*db, "tags", other));

  const auto notes = store::schema_version(*db, "notes");
  const auto tags = store::schema_version(*db, "tags");
  REQUIRE(notes);
  REQUIRE(tags);
  REQUIRE(*notes == 2);
  REQUIRE(*tags == 1);
}
