#include "agents_framework/tools/tool.hpp"

#include <string>

namespace agents_framework::tools {

namespace {

bool matches_type(const nlohmann::json& value, const std::string& type) {
  if (type == "string") return value.is_string();
  if (type == "number") return value.is_number();
  if (type == "integer") return value.is_number_integer() || value.is_number_unsigned();
  if (type == "boolean") return value.is_boolean();
  if (type == "array") return value.is_array();
  if (type == "object") return value.is_object();
  if (type == "null") return value.is_null();
  return true;
}

core::Result<void> validate_value(const nlohmann::json& schema, const nlohmann::json& value,
                                  const std::string& path) {
  if (!schema.is_object() || schema.empty()) return {};

  if (const auto type = schema.find("type"); type != schema.end()) {
    bool ok = false;
    std::string expected;
    if (type->is_array()) {
      for (const auto& candidate : *type) {
        if (candidate.is_string() && matches_type(value, candidate.get<std::string>())) {
          ok = true;
          break;
        }
      }
      expected = type->dump();
    } else if (type->is_string()) {
      expected = type->get<std::string>();
      ok = matches_type(value, expected);
    } else {
      ok = true;
    }
    if (!ok) {
      return core::fail(core::ErrorCode::Invalid,
                        "argument '" + path + "' must have type " + expected);
    }
  }

  if (const auto allowed = schema.find("enum"); allowed != schema.end() && allowed->is_array()) {
    bool found = false;
    for (const auto& candidate : *allowed) {
      if (candidate == value) {
        found = true;
        break;
      }
    }
    if (!found) {
      return core::fail(core::ErrorCode::Invalid,
                        "argument '" + path + "' must be one of " + allowed->dump());
    }
  }

  if (value.is_object()) {
    const auto properties = schema.find("properties");
    if (const auto required = schema.find("required");
        required != schema.end() && required->is_array()) {
      for (const auto& name : *required) {
        if (name.is_string() && !value.contains(name.get<std::string>())) {
          return core::fail(core::ErrorCode::Invalid,
                            "missing required argument '" + name.get<std::string>() + "'");
        }
      }
    }
    if (properties != schema.end() && properties->is_object()) {
      for (const auto& [name, property_schema] : properties->items()) {
        if (const auto found = value.find(name); found != value.end()) {
          const std::string child = path.empty() ? name : path + "." + name;
          AF_TRY_VOID(validate_value(property_schema, *found, child));
        }
      }
      if (const auto additional = schema.find("additionalProperties");
          additional != schema.end() && additional->is_boolean() && !additional->get<bool>()) {
        for (const auto& [name, ignored] : value.items()) {
          if (!properties->contains(name)) {
            return core::fail(core::ErrorCode::Invalid, "unexpected argument '" + name + "'");
          }
        }
      }
    }
  }

  if (value.is_array()) {
    if (const auto items = schema.find("items"); items != schema.end() && items->is_object()) {
      for (std::size_t i = 0; i < value.size(); ++i) {
        AF_TRY_VOID(validate_value(*items, value[i], path + "[" + std::to_string(i) + "]"));
      }
    }
  }

  return {};
}

}

core::Result<void> validate_args(const nlohmann::json& schema, const nlohmann::json& args) {
  if (!args.is_object()) {
    return core::fail(core::ErrorCode::Invalid, "tool arguments must be a JSON object");
  }
  return validate_value(schema, args, "");
}

}
