#include "agents_framework/store/vector_store.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <utility>

#include "agents_framework/store/migrate.hpp"

namespace agents_framework::store {

namespace {

constexpr std::array<Migration, 1> kMigrations{{
    {1,
     "CREATE TABLE vector_collections ("
     "  collection TEXT PRIMARY KEY,"
     "  dimensions INTEGER NOT NULL,"
     "  metric TEXT NOT NULL"
     ");"
     "CREATE TABLE vectors ("
     "  collection TEXT NOT NULL REFERENCES vector_collections (collection),"
     "  id TEXT NOT NULL,"
     "  content TEXT NOT NULL,"
     "  metadata TEXT NOT NULL,"
     "  embedding BLOB NOT NULL,"
     "  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
     "  PRIMARY KEY (collection, id)"
     ")"},
}};

std::vector<std::byte> encode_embedding(std::span<const float> embedding) {
  std::vector<std::byte> bytes(embedding.size() * sizeof(float));
  std::memcpy(bytes.data(), embedding.data(), bytes.size());
  return bytes;
}

core::Result<std::vector<float>> decode_embedding(const std::vector<std::byte>& bytes,
                                                  std::size_t dimensions) {
  if (bytes.size() != dimensions * sizeof(float)) {
    return core::fail(core::ErrorCode::Parse, "stored embedding has the wrong size",
                      std::to_string(bytes.size()) + " bytes for " +
                          std::to_string(dimensions) + " dimensions");
  }
  std::vector<float> embedding(dimensions);
  std::memcpy(embedding.data(), bytes.data(), bytes.size());
  return embedding;
}

float score(Metric metric, std::span<const float> query, std::span<const float> candidate) {
  float dot = 0.0F;
  float query_norm = 0.0F;
  float candidate_norm = 0.0F;
  float distance = 0.0F;
  for (std::size_t i = 0; i < query.size(); ++i) {
    dot += query[i] * candidate[i];
    query_norm += query[i] * query[i];
    candidate_norm += candidate[i] * candidate[i];
    const float diff = query[i] - candidate[i];
    distance += diff * diff;
  }
  switch (metric) {
    case Metric::Cosine: {
      const float norms = std::sqrt(query_norm) * std::sqrt(candidate_norm);
      return norms > 0.0F ? dot / norms : 0.0F;
    }
    case Metric::Dot:
      return dot;
    case Metric::L2:
      return -std::sqrt(distance);
  }
  return 0.0F;
}

core::Result<nlohmann::json> parse_metadata(const std::string& text) {
  nlohmann::json metadata = nlohmann::json::parse(text, nullptr, false);
  if (metadata.is_discarded()) {
    return core::fail(core::ErrorCode::Parse, "stored metadata is not valid JSON");
  }
  return metadata;
}

}

std::string_view metric_name(Metric metric) noexcept {
  switch (metric) {
    case Metric::Cosine: return "cosine";
    case Metric::Dot:    return "dot";
    case Metric::L2:     return "l2";
  }
  return "cosine";
}

Metric metric_from_string(std::string_view text) noexcept {
  if (text == "dot") return Metric::Dot;
  if (text == "l2") return Metric::L2;
  return Metric::Cosine;
}

core::Result<VectorStore> VectorStore::open(std::shared_ptr<Db> db, std::string collection,
                                            std::size_t dimensions, Metric metric) {
  if (!db || !db->is_open()) {
    return core::fail(core::ErrorCode::Invalid, "VectorStore requires an open database");
  }
  if (collection.empty() || dimensions == 0) {
    return core::fail(core::ErrorCode::Invalid,
                      "VectorStore requires a collection name and non-zero dimensions");
  }
  AF_TRY_VOID(migrate(*db, "vectors", kMigrations));

  AF_TRY(auto existing, db->prepare("SELECT dimensions, metric FROM vector_collections "
                                    "WHERE collection = ?1"));
  AF_TRY_VOID(existing.bind(1, collection));
  AF_TRY(const bool row, existing.step());
  if (row) {
    const auto stored_dimensions = static_cast<std::size_t>(existing.column_int64(0));
    const Metric stored_metric = metric_from_string(existing.column_text(1));
    if (stored_dimensions != dimensions || stored_metric != metric) {
      return core::fail(core::ErrorCode::Invalid,
                        "collection already exists with a different configuration",
                        "'" + collection + "' has " + std::to_string(stored_dimensions) +
                            " dimensions, metric " +
                            std::string{metric_name(stored_metric)});
    }
  } else {
    AF_TRY(auto insert, db->prepare("INSERT INTO vector_collections "
                                    "(collection, dimensions, metric) VALUES (?1, ?2, ?3)"));
    AF_TRY_VOID(insert.bind(1, collection));
    AF_TRY_VOID(insert.bind(2, static_cast<std::int64_t>(dimensions)));
    AF_TRY_VOID(insert.bind(3, metric_name(metric)));
    AF_TRY_VOID(insert.step());
  }
  return VectorStore{std::move(db), std::move(collection), dimensions, metric};
}

core::Result<void> VectorStore::upsert_one(const VectorRecord& record) {
  if (record.id.empty()) {
    return core::fail(core::ErrorCode::Invalid, "vector record id must not be empty");
  }
  if (record.embedding.size() != dimensions_) {
    return core::fail(core::ErrorCode::Invalid, "vector record has the wrong dimensions",
                      "'" + record.id + "' has " + std::to_string(record.embedding.size()) +
                          ", collection expects " + std::to_string(dimensions_));
  }
  AF_TRY(auto stmt,
         db_->prepare("INSERT INTO vectors (collection, id, content, metadata, embedding) "
                      "VALUES (?1, ?2, ?3, ?4, ?5) "
                      "ON CONFLICT (collection, id) DO UPDATE SET "
                      "content = excluded.content, metadata = excluded.metadata, "
                      "embedding = excluded.embedding"));
  AF_TRY_VOID(stmt.bind(1, collection_));
  AF_TRY_VOID(stmt.bind(2, record.id));
  AF_TRY_VOID(stmt.bind(3, record.content));
  AF_TRY_VOID(stmt.bind(4, record.metadata.dump()));
  AF_TRY_VOID(stmt.bind_blob(5, encode_embedding(record.embedding)));
  AF_TRY_VOID(stmt.step());
  return {};
}

core::Result<void> VectorStore::upsert(const VectorRecord& record) {
  return upsert_one(record);
}

core::Result<void> VectorStore::upsert(std::span<const VectorRecord> records) {
  AF_TRY(auto transaction, Transaction::begin(*db_));
  for (const VectorRecord& record : records) {
    AF_TRY_VOID(upsert_one(record));
  }
  return transaction.commit();
}

core::Result<std::optional<VectorRecord>> VectorStore::get(std::string_view id) const {
  AF_TRY(auto stmt, db_->prepare("SELECT content, metadata, embedding FROM vectors "
                                 "WHERE collection = ?1 AND id = ?2"));
  AF_TRY_VOID(stmt.bind(1, collection_));
  AF_TRY_VOID(stmt.bind(2, id));
  AF_TRY(const bool row, stmt.step());
  if (!row) return std::optional<VectorRecord>{};

  VectorRecord record;
  record.id = std::string{id};
  record.content = stmt.column_text(0);
  AF_TRY(record.metadata, parse_metadata(stmt.column_text(1)));
  AF_TRY(record.embedding, decode_embedding(stmt.column_blob(2), dimensions_));
  return std::optional<VectorRecord>{std::move(record)};
}

core::Result<bool> VectorStore::remove(std::string_view id) {
  AF_TRY(auto stmt, db_->prepare("DELETE FROM vectors WHERE collection = ?1 AND id = ?2"));
  AF_TRY_VOID(stmt.bind(1, collection_));
  AF_TRY_VOID(stmt.bind(2, id));
  AF_TRY_VOID(stmt.step());
  return db_->changes() > 0;
}

core::Result<std::int64_t> VectorStore::size() const {
  AF_TRY(auto stmt, db_->prepare("SELECT COUNT(*) FROM vectors WHERE collection = ?1"));
  AF_TRY_VOID(stmt.bind(1, collection_));
  AF_TRY_VOID(stmt.step());
  return stmt.column_int64(0);
}

core::Result<std::vector<VectorMatch>> VectorStore::query(std::span<const float> embedding,
                                                          std::size_t k) const {
  if (embedding.size() != dimensions_) {
    return core::fail(core::ErrorCode::Invalid, "query embedding has the wrong dimensions",
                      std::to_string(embedding.size()) + ", collection expects " +
                          std::to_string(dimensions_));
  }
  if (k == 0) return std::vector<VectorMatch>{};

  AF_TRY(auto stmt, db_->prepare("SELECT id, content, metadata, embedding FROM vectors "
                                 "WHERE collection = ?1"));
  AF_TRY_VOID(stmt.bind(1, collection_));

  std::vector<VectorMatch> matches;
  while (true) {
    AF_TRY(const bool row, stmt.step());
    if (!row) break;
    VectorMatch match;
    match.record.id = stmt.column_text(0);
    match.record.content = stmt.column_text(1);
    AF_TRY(match.record.metadata, parse_metadata(stmt.column_text(2)));
    AF_TRY(match.record.embedding, decode_embedding(stmt.column_blob(3), dimensions_));
    match.score = score(metric_, embedding, match.record.embedding);
    matches.push_back(std::move(match));
  }

  const auto better = [](const VectorMatch& a, const VectorMatch& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.record.id < b.record.id;
  };
  if (matches.size() > k) {
    std::partial_sort(matches.begin(), matches.begin() + static_cast<std::ptrdiff_t>(k),
                      matches.end(), better);
    matches.resize(k);
  } else {
    std::sort(matches.begin(), matches.end(), better);
  }
  return matches;
}

}
