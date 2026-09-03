# Orchestrating Agents, Step by Step

This guide builds an agent from nothing: a raw model call, then a graph, tools,
streaming, persistence, retrieval, multiple agents, and finally measurement. Each
step builds on the last and ends with a runnable example in `examples/`.

Build the project first (see the [README](../../README.md#quick-start)). Every
snippet runs offline against the mock backend, no API key needed, and the code is
identical for a live provider.

Every example starts from one include, and the snippets assume the same:

```cpp
#include "agents_framework/prelude.hpp"
```

The prelude pulls in the whole library and opens every module namespace, so
`Executor`, `Message`, and `ToolRegistry` need no qualifier. It belongs in `.cpp`
files only: a using-directive in a header leaks into every file that includes it.
Library code, and any file that wants control over what is visible, includes the
module headers directly, or `agents_framework/agents_framework.hpp` for everything
without the using-directives.

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
- The [README's Incoming Features](../../README.md#incoming-features): where the training features are
  headed.
