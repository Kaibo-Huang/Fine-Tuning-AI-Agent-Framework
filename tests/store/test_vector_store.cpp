#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/store/db.hpp"
#include "agents_framework/store/vector_store.hpp"

namespace core = agents_framework::core;
namespace store = agents_framework::store;

namespace {

std::shared_ptr<store::Db> open_db() {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  return std::make_shared<store::Db>(std::move(*opened));
}

store::VectorStore make_vectors(std::shared_ptr<store::Db> db, std::string collection = "docs",
                                store::Metric metric = store::Metric::Cosine) {
  auto vectors = store::VectorStore::open(std::move(db), std::move(collection), 3, metric);
  REQUIRE(vectors);
  return std::move(*vectors);
}

}

TEST_CASE("vector records round-trip through the store", "[store][vectors]") {
  auto vectors = make_vectors(open_db());
  const store::VectorRecord record{"doc-1", {1.0F, 0.0F, 0.5F}, "hello world", {{"tag", "greeting"}}};
  REQUIRE(vectors.upsert(record));

  const auto loaded = vectors.get("doc-1");
  REQUIRE(loaded);
  REQUIRE(loaded->has_value());
  REQUIRE(**loaded == record);

  const auto missing = vectors.get("ghost");
  REQUIRE(missing);
  REQUIRE(!missing->has_value());
}

TEST_CASE("upsert replaces an existing record", "[store][vectors]") {
  auto vectors = make_vectors(open_db());
  REQUIRE(vectors.upsert({"doc-1", {1.0F, 0.0F, 0.0F}, "old", {}}));
  REQUIRE(vectors.upsert({"doc-1", {0.0F, 1.0F, 0.0F}, "new", {}}));

  const auto loaded = vectors.get("doc-1");
  REQUIRE(loaded);
  REQUIRE((*loaded)->content == "new");
  const auto count = vectors.size();
  REQUIRE(count);
  REQUIRE(*count == 1);
}

TEST_CASE("batch upsert stores every record atomically", "[store][vectors]") {
  auto vectors = make_vectors(open_db());
  const std::array<store::VectorRecord, 3> records{{
      {"a", {1.0F, 0.0F, 0.0F}, "first", {}},
      {"b", {0.0F, 1.0F, 0.0F}, "second", {}},
      {"c", {0.0F, 0.0F, 1.0F}, "third", {}},
  }};
  REQUIRE(vectors.upsert(records));
  const auto count = vectors.size();
  REQUIRE(count);
  REQUIRE(*count == 3);

  const std::array<store::VectorRecord, 2> broken{{
      {"d", {1.0F, 1.0F, 0.0F}, "ok", {}},
      {"e", {1.0F}, "wrong dimensions", {}},
  }};
  REQUIRE(!vectors.upsert(broken));
  const auto after = vectors.size();
  REQUIRE(after);
  REQUIRE(*after == 3);
}

TEST_CASE("cosine queries rank by angular similarity", "[store][vectors]") {
  auto vectors = make_vectors(open_db());
  REQUIRE(vectors.upsert({"aligned", {2.0F, 0.0F, 0.0F}, "same direction", {}}));
  REQUIRE(vectors.upsert({"orthogonal", {0.0F, 1.0F, 0.0F}, "sideways", {}}));
  REQUIRE(vectors.upsert({"diagonal", {1.0F, 1.0F, 0.0F}, "between", {}}));

  const std::array<float, 3> query{1.0F, 0.0F, 0.0F};
  const auto matches = vectors.query(query, 2);
  REQUIRE(matches);
  REQUIRE(matches->size() == 2);
  REQUIRE(matches->at(0).record.id == "aligned");
  REQUIRE(matches->at(0).score > 0.99F);
  REQUIRE(matches->at(1).record.id == "diagonal");
}

TEST_CASE("l2 queries rank by distance with higher-is-better scores", "[store][vectors]") {
  auto vectors = make_vectors(open_db(), "l2docs", store::Metric::L2);
  REQUIRE(vectors.upsert({"near", {1.0F, 0.0F, 0.0F}, "", {}}));
  REQUIRE(vectors.upsert({"far", {5.0F, 5.0F, 5.0F}, "", {}}));

  const std::array<float, 3> query{0.9F, 0.0F, 0.0F};
  const auto matches = vectors.query(query, 2);
  REQUIRE(matches);
  REQUIRE(matches->at(0).record.id == "near");
  REQUIRE(matches->at(0).score > matches->at(1).score);
}

TEST_CASE("querying with the wrong dimensions is rejected", "[store][vectors]") {
  auto vectors = make_vectors(open_db());
  const std::array<float, 2> query{1.0F, 0.0F};
  const auto matches = vectors.query(query, 1);
  REQUIRE(!matches);
  REQUIRE(matches.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("remove deletes a record and reports whether it existed", "[store][vectors]") {
  auto vectors = make_vectors(open_db());
  REQUIRE(vectors.upsert({"doc-1", {1.0F, 0.0F, 0.0F}, "", {}}));

  const auto removed = vectors.remove("doc-1");
  REQUIRE(removed);
  REQUIRE(*removed);
  const auto again = vectors.remove("doc-1");
  REQUIRE(again);
  REQUIRE(!*again);
}

TEST_CASE("collections are isolated and validated on reopen", "[store][vectors]") {
  auto db = open_db();
  auto docs = make_vectors(db, "docs");
  auto skills = make_vectors(db, "skills");
  REQUIRE(docs.upsert({"only-docs", {1.0F, 0.0F, 0.0F}, "", {}}));

  const auto in_skills = skills.get("only-docs");
  REQUIRE(in_skills);
  REQUIRE(!in_skills->has_value());

  const auto reopened = store::VectorStore::open(db, "docs", 3, store::Metric::Cosine);
  REQUIRE(reopened);
  const auto count = reopened->size();
  REQUIRE(count);
  REQUIRE(*count == 1);

  const auto mismatched = store::VectorStore::open(db, "docs", 5, store::Metric::Cosine);
  REQUIRE(!mismatched);
  REQUIRE(mismatched.error().code == core::ErrorCode::Invalid);
}
