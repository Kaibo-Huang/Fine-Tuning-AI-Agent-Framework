#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/store/db.hpp"

namespace agents_framework::store {

enum class Metric { Cosine, Dot, L2 };

[[nodiscard]] std::string_view metric_name(Metric metric) noexcept;
[[nodiscard]] Metric metric_from_string(std::string_view text) noexcept;

struct VectorRecord {
  std::string id;
  std::vector<float> embedding;
  std::string content;
  nlohmann::json metadata = nlohmann::json::object();

  bool operator==(const VectorRecord&) const = default;
};

struct VectorMatch {
  VectorRecord record;
  float score{0.0F};
};

class VectorStore {
 public:
  static core::Result<VectorStore> open(std::shared_ptr<Db> db, std::string collection,
                                        std::size_t dimensions, Metric metric = Metric::Cosine);

  core::Result<void> upsert(const VectorRecord& record);
  core::Result<void> upsert(std::span<const VectorRecord> records);
  core::Result<std::optional<VectorRecord>> get(std::string_view id) const;
  core::Result<bool> remove(std::string_view id);
  core::Result<std::int64_t> size() const;

  core::Result<std::vector<VectorMatch>> query(std::span<const float> embedding,
                                               std::size_t k) const;

  [[nodiscard]] const std::string& collection() const noexcept { return collection_; }
  [[nodiscard]] std::size_t dimensions() const noexcept { return dimensions_; }
  [[nodiscard]] Metric metric() const noexcept { return metric_; }

 private:
  VectorStore(std::shared_ptr<Db> db, std::string collection, std::size_t dimensions,
              Metric metric)
      : db_(std::move(db)), collection_(std::move(collection)), dimensions_(dimensions),
        metric_(metric) {}

  core::Result<void> upsert_one(const VectorRecord& record);

  std::shared_ptr<Db> db_;
  std::string collection_;
  std::size_t dimensions_;
  Metric metric_;
};

}
