#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace agents_framework::llm {

// A tool is smth the model can call
struct ToolDef {
  std::string name;
  std::string description;
  nlohmann::json input_schema;
  bool operator==(const ToolDef&) const = default;
};

inline void to_json(nlohmann::json& j, const ToolDef& tool) {
  j = nlohmann::json::object();
  j["name"] = tool.name;
  j["description"] = tool.description;
  j["input_schema"] = tool.input_schema;
}

inline void from_json(const nlohmann::json& j, ToolDef& tool) {
  tool.name = j.at("name").get<std::string>();
  tool.description = j.value("description", std::string{});
  tool.input_schema = j.value("input_schema", nlohmann::json::object());
}

}  
