# Orchestrating Agents, Step by Step

This guide builds an agent from nothing: first a raw model call, then a graph, then
tools, streaming, persistence, retrieval, multiple agents, and finally measurement.
Each step is self-contained, builds on the one before it, and ends with a pointer to
a complete, runnable example in `examples/`.

Before starting, build the project once (see the [README](../../README.md#quick-start)).
Every snippet runs offline against the mock backend by default, no API key needed,
and the code is identical for a live provider.

To keep the snippets readable, they assume the same using-directives the examples use:

```cpp
using namespace agents_framework::core;   // Result<T>, load_dotenv, ...
using namespace agents_framework::llm;    // Message, ChatRequest, backends, ...
using namespace agents_framework::graph;  // Channel, GraphBuilder, Executor, ...
using namespace agents_framework::tools;  // ToolDef, ToolRegistry, ...
using namespace agents_framework::store;  // Db, CheckpointStore, VectorStore, ...
```

## The steps

1. **[Talk to a model](01-backends.md)**: configure a backend, send a message, stream
   a reply.
2. **[Build your first graph](02-first-graph.md)**: declare typed state, add nodes and
   edges, compile, run.
3. **[Wire the ReAct tool loop](03-tools.md)**: register a tool and route the model to
   it until it answers.
4. **[Stream events from a run](04-streaming-and-events.md)**: watch super-steps and
   tokens as they happen.
5. **[Checkpoint, resume, and ask a human](05-persistence.md)**: persist every step to
   SQLite, pause for approval, resume, replay.
6. **[Add memory and retrieval](06-memory-and-rag.md)**: embed a corpus, store
   vectors, retrieve context into the prompt.
7. **[Compose agents as subgraphs](07-multi-agent.md)**: mount whole agents as nodes
   under a supervisor.
8. **[Measure the agent](08-evaluation.md)**: score it over a task suite with
   confidence intervals, record a baseline, export training data.

## After the guide

- [Architecture](../architecture.md): the reference view. Modules, the execution
  model, and conventions.
- [Examples walkthrough](../examples.md): what each demo in `examples/` shows.
- The [README's roadmap](../../README.md#roadmap): where the training features are
  headed.
