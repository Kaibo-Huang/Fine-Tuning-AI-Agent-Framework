# Step 2: Build Your First Graph

An agent is a graph: nodes do work, edges decide what runs next, and all of them
share one typed state. This step builds the smallest possible one.

## 1. Declare the state

State is a compile-time schema of named channels. Each channel has a name, a type,
and optionally a **reducer** that defines how concurrent writes merge:

```cpp
using Messages    = Channel<"messages", std::vector<Message>, Append>;
using AgentSchema = Schema<Messages>;
```

`Append` means every node's output is appended, exactly right for a conversation.
Without a reducer, the last write wins. Because names are template parameters,
`state.get<"messages">()` is fully typed and checked at compile time; there is no
string lookup at run time.

This messages-plus-`Append` schema is so common the library ships it prebuilt as
`ChatSchema`, with a `chat_state("question")` helper to seed it. The guide declares
it by hand once so you can see what it is; after that, use the prebuilt one.

## 2. Write a node

A node reads the state through a `StateView` and returns an `Update`. It never
mutates shared state directly; the executor applies updates through the reducers:

```cpp
auto greet = [](StateView<AgentSchema> view) {
  const auto count = view.get<"messages">().size();
  return Update<AgentSchema>{}.write<"messages">(
      {Message::assistant_text("The conversation has " + std::to_string(count) +
                               " message(s) so far.")});
};
```

An LLM call is just a prebuilt node of the same shape:

```cpp
LlmNodeOptions options;
options.system = "You are a concise assistant.";

auto answer = make_llm_node<AgentSchema>(backend, options);
```

It reads the `messages` channel, calls the backend, and appends the reply.

## 3. Wire and compile

```cpp
GraphBuilder<AgentSchema> builder;
builder.add_node("answer", make_llm_node<AgentSchema>(backend, options))
    .set_entry("answer")
    .set_finish("answer");

auto graph = std::move(builder).compile();
if (!graph) { /* compile() validates the wiring: unknown nodes, unreachable states */ }
```

Compilation is where structural mistakes surface, before anything runs.

## 4. Run it

```cpp
State<AgentSchema> state;
state.set<"messages">({Message::user_text("What is a directed graph?")});

Executor executor;
const auto stats = executor.run(*graph, state, RunOptions{.max_steps = 10});

std::cout << stats->steps << " super-steps, " << stats->node_runs << " node runs\n";
```

The executor advances in **super-steps**: every active node runs (concurrently, when
there are several), the outputs are buffered, and at a barrier they are applied to
the state in a deterministic order. Given the same inputs, seeds, and model outputs,
a run is reproducible. That guarantee is what later makes replay and evaluation
trustworthy.

After the run, `state` holds the final conversation:

```cpp
const auto& messages = state.get<"messages">();  // your message + the model's reply
```

**Next:** [Step 3: Wire the ReAct tool loop](03-tools.md), where edges start making
decisions.
