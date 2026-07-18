#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/tools/registry.hpp"
#include "agents_framework/tools/tool.hpp"

namespace core = agents_framework::core;
namespace llm = agents_framework::llm;
namespace tools = agents_framework::tools;

namespace {

llm::ToolDef adder_def() {
  llm::ToolDef def;
  def.name = "add";
  def.description = "Add two numbers.";
  def.input_schema = {
      {"type", "object"},
      {"properties",
       {{"a", {{"type", "number"}}}, {"b", {{"type", "number"}}}}},
      {"required", nlohmann::json::array({"a", "b"})}};
  return def;
}

std::unique_ptr<tools::Tool> adder() {
  return tools::make_tool(adder_def(), [](const nlohmann::json& args) -> core::Result<std::string> {
    return std::to_string(args.at("a").get<double>() + args.at("b").get<double>());
  });
}

}

TEST_CASE("a native tool registers and is invoked with valid args", "[tools][registry]") {
  tools::ToolRegistry registry;
  REQUIRE(registry.add(adder()));
  REQUIRE(registry.size() == 1);
  REQUIRE(registry.find("add") != nullptr);
  REQUIRE(registry.find("missing") == nullptr);

  const auto result = registry.invoke("add", {{"a", 19}, {"b", 23}});
  REQUIRE(result);
  REQUIRE(result->starts_with("42"));
}

TEST_CASE("registry exposes the tool definitions for an llm request", "[tools][registry]") {
  tools::ToolRegistry registry;
  REQUIRE(registry.add(adder()));
  const auto defs = registry.defs();
  REQUIRE(defs.size() == 1);
  REQUIRE(defs.front() == adder_def());
}

TEST_CASE("duplicate, empty, and null tools are rejected", "[tools][registry]") {
  tools::ToolRegistry registry;
  REQUIRE(registry.add(adder()));

  auto duplicate = registry.add(adder());
  REQUIRE(!duplicate);
  REQUIRE(duplicate.error().message == "duplicate tool name");

  auto null_tool = registry.add(nullptr);
  REQUIRE(!null_tool);

  auto unnamed = registry.add(tools::make_tool(
      llm::ToolDef{}, [](const nlohmann::json&) -> core::Result<std::string> { return ""; }));
  REQUIRE(!unnamed);
}

TEST_CASE("invoking an unknown tool fails with NotFound", "[tools][registry]") {
  tools::ToolRegistry registry;
  const auto result = registry.invoke("ghost", nlohmann::json::object());
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::NotFound);
  REQUIRE(result.error().context == "ghost");
}

TEST_CASE("argument validation runs before the tool body", "[tools][registry]") {
  tools::ToolRegistry registry;
  bool called = false;
  auto def = adder_def();
  REQUIRE(registry.add(tools::make_tool(
      std::move(def), [&called](const nlohmann::json&) -> core::Result<std::string> {
        called = true;
        return "ran";
      })));

  SECTION("missing required argument") {
    const auto result = registry.invoke("add", {{"a", 1}});
    REQUIRE(!result);
    REQUIRE(result.error().message == "missing required argument 'b'");
  }
  SECTION("wrong argument type") {
    const auto result = registry.invoke("add", {{"a", 1}, {"b", "two"}});
    REQUIRE(!result);
    REQUIRE(result.error().message == "argument 'b' must have type number");
  }
  SECTION("non-object arguments") {
    const auto result = registry.invoke("add", nlohmann::json::array());
    REQUIRE(!result);
    REQUIRE(result.error().message == "tool arguments must be a JSON object");
  }
  REQUIRE(!called);
}

TEST_CASE("tool errors surface with the tool name as context", "[tools][registry]") {
  tools::ToolRegistry registry;
  llm::ToolDef def;
  def.name = "flaky";
  REQUIRE(registry.add(tools::make_tool(
      std::move(def), [](const nlohmann::json&) -> core::Result<std::string> {
        return core::fail(core::ErrorCode::Tool, "backend unavailable");
      })));

  const auto result = registry.invoke("flaky", nlohmann::json::object());
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Tool);
  REQUIRE(result.error().context == "flaky");
}

TEST_CASE("the schema validator covers enums, arrays, and closed objects", "[tools][validate]") {
  const nlohmann::json schema = {
      {"type", "object"},
      {"properties",
       {{"unit", {{"type", "string"}, {"enum", nlohmann::json::array({"C", "F"})}}},
        {"tags", {{"type", "array"}, {"items", {{"type", "string"}}}}},
        {"nested",
         {{"type", "object"},
          {"properties", {{"depth", {{"type", "integer"}}}}},
          {"required", nlohmann::json::array({"depth"})}}}}},
      {"additionalProperties", false}};

  REQUIRE(tools::validate_args(
      schema, {{"unit", "C"}, {"tags", nlohmann::json::array({"x", "y"})},
               {"nested", {{"depth", 3}}}}));

  auto bad_enum = tools::validate_args(schema, {{"unit", "K"}});
  REQUIRE(!bad_enum);
  REQUIRE(bad_enum.error().message.starts_with("argument 'unit' must be one of"));

  auto bad_item = tools::validate_args(schema, {{"tags", nlohmann::json::array({"x", 7})}});
  REQUIRE(!bad_item);
  REQUIRE(bad_item.error().message == "argument 'tags[1]' must have type string");

  auto bad_nested = tools::validate_args(schema, {{"nested", nlohmann::json::object()}});
  REQUIRE(!bad_nested);
  REQUIRE(bad_nested.error().message == "missing required argument 'depth'");

  auto extra = tools::validate_args(schema, {{"surprise", 1}});
  REQUIRE(!extra);
  REQUIRE(extra.error().message == "unexpected argument 'surprise'");

  REQUIRE(tools::validate_args(nlohmann::json::object(), {{"anything", "goes"}}));
}
