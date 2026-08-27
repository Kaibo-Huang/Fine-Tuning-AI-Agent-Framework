# Step 3: Wire the ReAct Tool Loop

A ReAct agent alternates between reasoning (the model) and acting (tools) until it
has an answer. In this framework that is two nodes and two edges.

## 1. Define a tool

A tool is a JSON-schema definition the model sees, paired with a C++ callback the
framework runs with validated arguments:

```cpp
ToolDef calculator;
calculator.name = "calculator";
calculator.description = "Evaluate a basic arithmetic operation on two numbers.";
calculator.input_schema = {
    {"type", "object"},
    {"properties", {{"op", {{"type", "string"}}},
                    {"a", {{"type", "number"}}},
                    {"b", {{"type", "number"}}}}},
    {"required", json::array({"op", "a", "b"})}};

auto evaluate = [](const json& args) -> Result<std::string> {
  const double a = args.at("a").get<double>();
  const double b = args.at("b").get<double>();
  if (args.at("op") == "divide" && b == 0.0)
    return fail(ErrorCode::Tool, "division by zero");
  // ... compute ...
  return std::to_string(a * b);
};
```

Returning an error from the callback is fine; it flows back to the model as a tool
result, and the model gets a chance to recover.

## 2. Register it

```cpp
auto registry = std::make_shared<ToolRegistry>();
(void)registry->add(make_tool(std::move(calculator), evaluate));
```

The registry is shared by two consumers: the LLM node advertises its definitions to
the model, and the tool node executes whatever the model calls.

## 3. Build the loop

```cpp
LlmNodeOptions options;
options.system = "Use the calculator tool for any arithmetic, then answer briefly.";
options.tools = registry->defs();

GraphBuilder<AgentSchema> builder;
builder.add_node("agent", make_llm_node<AgentSchema>(backend, options))
    .add_node("tools", make_tool_node<AgentSchema>(registry))
    .set_entry("agent")
    .add_conditional_edge("agent", tools_router<AgentSchema>("tools"),
                          {"tools", std::string{kEnd}})
    .add_edge("tools", "agent");
```

Read the wiring as a sentence: start at **agent**; after it runs, the conditional
edge (`tools_router`) checks whether the model asked for a tool. If yes, go to
**tools**; if no, the run **ends**. **tools** always hands the conversation back to
**agent**.

`tools_router` is a prebuilt router. A conditional edge can be any function of the
state that returns the next node's name, so routing on your own logic looks the same.

## 4. Run with a budget

Loops need bounds. `max_steps` caps the number of super-steps, so a model that keeps
calling tools cannot spin forever:

```cpp
State<AgentSchema> state;
state.set<"messages">({Message::user_text("What is 987654321 times 123456789?")});

Executor executor;
const auto stats = executor.run(*graph, state, RunOptions{.max_steps = 10});
```

Afterwards the `messages` channel contains the full trajectory (the tool call, the
tool result, and the final answer) because every node appended to it.

## 5. The shortcut

This wiring is the standard pattern, so the library ships it prebuilt. Everything
above collapses to:

```cpp
auto graph = make_react_agent(backend, registry, {.system = "Use the calculator."});
auto state = chat_state("What is 987654321 times 123456789?");

Executor executor;
auto stats = executor.run(*graph, state, {.max_steps = 10});

std::cout << last_assistant_text(state.get<"messages">()) << "\n";
```

`make_react_agent` builds exactly the graph from this step over the library's
`ChatSchema`, advertises the registry's tools to the model automatically, and
returns the same `CompiledGraph` a hand-built one produces. Start from the
prebuilt; reach for the manual wiring when the loop needs extra nodes or channels.

## Complete example

[`examples/react_demo.cpp`](../../examples/react_demo.cpp) runs the shortcut form,
plus transcript printing and the scripted mock trajectory. The manual wiring above
is what `make_react_agent` does inside.

**Next:** [Step 4: Stream events from a run](04-streaming-and-events.md)
