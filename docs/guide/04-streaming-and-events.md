# Step 4: Stream Events from a Run

Once graphs loop and call tools, you want to watch them work: which nodes are
running, what tokens the model is producing, when a run pauses. The `EventBus`
carries all of it.

## 1. Create a bus and subscribe

```cpp
auto events = std::make_shared<EventBus>();

events->subscribe([](const ExecEvent& event) {
  if (const auto* step = std::get_if<StepStarted>(&event)) {
    std::cout << "[step " << step->step << "]";
    for (const auto& node : step->nodes) std::cout << " " << node;
    std::cout << "\n";
  } else if (const auto* token = std::get_if<TokenDelta>(&event)) {
    std::cout << token->text << std::flush;   // live token stream
  } else if (std::get_if<RunInterrupted>(&event)) {
    std::cout << "[run paused for approval]\n";
  }
});
```

`ExecEvent` is a variant; subscribe once and pattern-match on the cases you care
about. A subscriber that ignores an event type costs nothing.

## 2. Hand the bus to the run, and to the nodes

The executor publishes step-level events when the bus is in `RunOptions`; an LLM node
publishes its token deltas when the bus is in its options:

```cpp
LlmNodeOptions llm_options;
llm_options.events = events;       // token deltas from this node
llm_options.label  = "researcher"; // names this node's events

const RunOptions run_options{.events = events};  // step lifecycle from the executor

executor.run(*graph, state, run_options);
```

Labels matter once several LLM nodes share one bus: they tell you *which* agent is
speaking.

## What this gives you

- **Live UX**: stream tokens to a terminal or UI while the graph is mid-run.
- **Progress**: `StepStarted` shows the active node set of every super-step, which
  is also the clearest picture of the scheduler doing its work; parallel nodes appear
  together in one step.
- **Hooks**: the same events later drive tracing and observability; nothing about a
  subscriber is demo-specific.

## Complete example

[`examples/orchestration_demo.cpp`](../../examples/orchestration_demo.cpp) uses one
bus across a supervisor and two sub-agents (`watch_events`), so you can see steps,
streamed tokens, and an interrupt in a single run.

**Next:** [Step 5: Checkpoint, resume, and ask a human](05-persistence.md)
