# Step 5: Checkpoint, Resume, and Ask a Human

With a checkpointer attached, the executor saves the whole graph state to SQLite
after every super-step. One mechanism buys three capabilities: resuming a run,
replaying its history, and pausing for human approval.

## 1. Open a store

```cpp
auto db = *Db::open_shared("agent.sqlite");  // or Db::open_memory_shared()
auto checkpoints = *CheckpointStore::open(db);
```

(Both calls return `Result`s; the examples wrap them in a small `need()` helper that
prints the error and exits. Do the same or handle them properly.)

## 2. Checkpoint every step

Give the run an id and the checkpointer:

```cpp
const RunOptions options{.run_id = "run-42",
                         .checkpointer = &checkpoints};

executor.run(*graph, state, options);
```

That is the entire integration. Each super-step now persists the serialized state;
the channel schema you declared in Step 2 already knows how to serialize itself.

## 3. Pause for approval

`interrupt_before` stops the run just before the named nodes execute:

```cpp
const RunOptions options{.run_id = "run-42",
                         .checkpointer = &checkpoints,
                         .interrupt_before = {"report"}};

const auto paused = executor.run(*graph, state, options);

if (paused->status == RunStatus::Interrupted) {
  // paused->pending_nodes tells you what is waiting; here: "report"
}
```

Nothing is lost while you wait: the state is already in SQLite, so the process can
exit and a different process can pick the run up later.

## 4. Approve and resume

Load the latest checkpoint, restore the typed state, and continue:

```cpp
const auto checkpoint = *checkpoints.latest("run-42");
auto restored = *State<AgentSchema>::deserialize(checkpoint.state);

executor.resume(*graph, restored, checkpoint, options);
```

The resumed run continues from exactly the barrier it stopped at: the pending nodes
run next, and checkpointing continues as before.

## 5. Time travel

Every step of every run is retained, so history is queryable and forkable:

```cpp
const auto history = *checkpoints.list("run-42");   // one entry per super-step
```

`CheckpointStore::fork` starts a new run from any past step. It is the debugging move
when you want to know "what would have happened if the state had been different at
step 3?", and the foundation of deterministic replay.

## Complete example

[`examples/orchestration_demo.cpp`](../../examples/orchestration_demo.cpp) runs with
`interrupt_before = {"report"}`, prints the pending node, then reloads the checkpoint
and resumes: a complete approval flow, minus the human.

**Next:** [Step 6: Add memory and retrieval](06-memory-and-rag.md)
