# Step 7 — Compose Agents as Subgraphs

An agent in this framework is a compiled graph — which means an agent can be a
*node* in a bigger graph. Multi-agent systems are composition, not a new concept.

## 1. Keep each agent's state private

The supervisor and its workers have different schemas. The supervisor coordinates;
each worker keeps its own conversation:

```cpp
using SupervisorSchema = Schema<Task, Query, Results>;   // Results appends
using ResearchSchema   = Schema<Query, Docs, Messages>;  // the Step 6 agent
using MathSchema       = Schema<Messages>;
```

## 2. Translate state at the boundary

Mounting a subgraph takes two functions: an **enter** function that builds the
child's starting state from the parent's, and a **report** function that turns the
finished child state into an update for the parent:

```cpp
Result<State<ResearchSchema>> enter_research(StateView<SupervisorSchema> parent) {
  State<ResearchSchema> child;
  child.set<"query">(parent.get<"query">());
  return child;
}

Result<Update<SupervisorSchema>> report_research(const State<ResearchSchema>& child) {
  return Update<SupervisorSchema>{}.write<"results">(
      {"research: " + last_assistant_text(child.get<"messages">())});
}
```

This boundary is the whole multi-agent contract: the child never sees the parent's
state, and the parent only sees what the report function chooses to surface.

## 3. Mount the agents and route between them

```cpp
GraphBuilder<SupervisorSchema> builder;
builder
    .add_node("supervisor", start_task)  // copies the task into the query channel
    .add_node("research_agent",
              make_subgraph_node<SupervisorSchema, ResearchSchema>(
                  build_research_agent(...), enter_research, report_research))
    .add_node("math_agent",
              make_subgraph_node<SupervisorSchema, MathSchema>(
                  build_math_agent(...), enter_math, report_math))
    .add_node("report", write_report)    // folds the results channel into a summary
    .set_entry("supervisor")
    .add_conditional_edge("supervisor", pick_agent)  // any function of the state
    .add_edge("research_agent", "report")
    .add_edge("math_agent", "report")
    .set_finish("report");
```

`pick_agent` is a plain function returning the next node's name — route on keywords,
on a classifier, or on an LLM call; the graph does not care. Because `results` is an
appending channel, workers that run in the same super-step report concurrently
without clobbering each other — that is the reducer from Step 2 earning its keep.

## 4. Everything still composes

The mounted agents are ordinary graphs, so every capability from the earlier steps
holds for the whole ensemble: one event bus streams all of it (label the LLM nodes to
tell the speakers apart), one checkpointer persists it, and `interrupt_before` can
gate any node — including an entire sub-agent.

## Complete example

[`examples/orchestration_demo.cpp`](../../examples/orchestration_demo.cpp) is this
step end to end: a supervisor, the Step 6 research agent, a math agent, streaming,
checkpoints, and an approval pause in one run.

**Next:** [Step 8 — Measure the agent](08-evaluation.md)
