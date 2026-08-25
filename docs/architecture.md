# Architecture

This document is the reference view of the framework as it exists today: the
modules, the execution model, and the conventions the code follows. If you are new,
start with the step-by-step guide, [Orchestrating Agents, Step by
Step](guide/README.md), and come back here for the full picture. The forward-looking
design, including the training pipeline and the `finetune` / `distill` / `port`
features, lives in [PLAN.md](../PLAN.md).

## The big picture

The framework is organized as three tracks:

1. **The orchestration engine** (primary product). LLMs are embedded in structured,
   stateful execution graphs that coordinate reasoning steps, tool calls, memory, and
   multi-agent collaboration.
2. **The measurement stack** (the bridge). Every run can be captured as a structured
   trace, scored by a pluggable evaluator, aggregated by an eval harness with
   confidence intervals, and exported as training data.
3. **The training pipeline** (upcoming). LibTorch-based local inference and LoRA
   training behind three features: `finetune` (a curated dataset becomes an adapter),
   `distill` (a better model or an entire graph teaches a smaller one), and `port`
   (an existing task adaptation moves to a new base model).

One rule spans all three tracks: **a model is never trained on its own outputs.**
Supervision comes from a stronger model, gold labels, or a programmatic verifier.
The measurement stack exists to generate teacher datasets and to judge students
against baselines, not to close a self-improvement loop.

Tracks 1 and 2 are complete; track 3 is the current focus.

## Modules

Public headers live under `include/agents_framework/`, one directory per module, with
implementations mirrored under `src/`.

| Module | What it provides |
|---|---|
| `core` | `Result<T>` (`std::expected<T, Error>`), structured logging with secret redaction, config, IDs, seeded RNG, a thread pool, `.env` loading |
| `http` | a libcurl transport behind a small `Transport` seam (so retry/backoff is unit-testable offline), SSE parsing for streaming, secret handling |
| `llm` | the canonical chat types (messages, content blocks, tool definitions, requests/responses, stream events) and the backends that speak them: Anthropic Messages API, OpenAI-compatible (OpenAI, OpenRouter, vLLM, llama.cpp), and a deterministic mock; embedding backends behind the same pattern |
| `graph` | typed state channels, the graph builder and compiler, the super-step executor, checkpointing, the event bus, and the built-in node types (LLM, tool, retrieval, subgraph) |
| `tools` | native C++, subprocess, and HTTP tool kinds; a JSON-schema registry with argument validation; a tolerant text-protocol fallback for models without native function calling |
| `store` | the SQLite wrapper and migrations, the checkpoint store, and the vector store |
| `trace` | structured trace capture from executor runs, the SQLite trace store, the trace-to-example dataset builder, and a graph channel that accumulates verified training examples while a teacher graph executes |
| `eval` | the `TaskSuite` seam, built-in verifiers, pluggable evaluators, the eval harness with confidence intervals, the eval store for baselines, and the text-to-SQL reference suite |

## State: typed channels with reducers

Shared graph state is a compile-time schema of named channels:

```cpp
using Messages = Channel<"messages", std::vector<Message>, Append>;
using Query    = Channel<"query", std::string>;
using Schema_  = Schema<Messages, Query>;
```

Channel names are non-type template parameters, so `state.get<"messages">()` is a
typed, monomorphized accessor: no codegen step and no hot-path type lookups. Each
channel has a **reducer** that defines how concurrent node outputs merge (the default
is last-write; `Append` accumulates). A type-erased channel map underneath gives the
executor and the persistence layer a uniform handle, and every channel type knows how
to serialize itself for checkpointing.

Nodes never mutate state directly: they read through a `StateView` and return an
`Update` that the executor applies through the reducers.

## Execution: deterministic super-steps

The executor is a BSP (Pregel-style) scheduler. Within a super-step, all active nodes
run concurrently on a thread pool; their outputs are buffered and applied to shared
state **at a barrier, in a deterministic order**. Conditional edges and cycles are
evaluated at the barrier to select the next super-step's active set, and step budgets
bound loops.

The guarantee this buys: given the same inputs, seeds, and model/tool outputs, a run
is reproducible.

## Persistence, replay, and human-in-the-loop

With a checkpointer configured, the executor saves state to SQLite after every
super-step. That one mechanism provides:

- **Resume**: pick a run back up from its latest checkpoint.
- **Time travel**: `CheckpointStore::fork` replays a run from any past step.
- **Human-in-the-loop**: `interrupt_before` pauses a run before named nodes; the
  application inspects the checkpoint, gets approval, and resumes.

`orchestration_demo` exercises all three paths end to end.

## LLM backends

All backends exchange one canonical request/response/tool-call format, so graphs are
written once and run against any provider. Backend selection is environment-driven
(`AF_BACKEND`, `AF_MODEL`, `AF_BASE_URL`; see the README's configuration table) with
the mock backend as the default, which keeps every example and test runnable offline.
Tool calls use the provider's native function calling as the primary path, with a
tolerant text-protocol parser as the fallback for weaker or local models.

## Traces, task suites, and evaluation

The `TaskSuite` interface is the domain seam: it supplies task instances (with
train / held-out / retention splits) and a verifier that scores an agent's output.
Built-in verifier kinds cover most needs (exact match, numeric tolerance, result-set
comparison, subprocess exit code, LLM-judge); a custom verifier is a function.

The eval harness fans an agent across a suite concurrently, scores each instance, and
aggregates into **accuracy with a 95% confidence interval**, pinning everything needed
to reproduce the number: model, adapter version, seed, framework version, and git
commit. Reports persist in SQLite, a run can be recorded as the suite's **baseline**,
and two runs can be compared for a signed delta. Verified traces export directly as
JSONL training data; run a strong teacher over a suite and the accepted traces become
the dataset a smaller student is distilled on.

The shipped text-to-SQL suite (REF-A) verifies by executing predicted and gold SQL
against multiple database instances and comparing result sets, so a query that is
wrong but coincidentally matches on one dataset is still caught. Nothing in the
harness knows what SQL is; the third-domain test in `tests/eval/` exists specifically
to prove the machinery is domain-independent.

## Conventions

- **Errors**: recoverable failures return `Result<T>`; exceptions are reserved for
  programmer errors and broken invariants.
- **Secrets**: API keys come from the environment or `.env`, are never logged, and are
  redacted from error messages.
- **Testing**: everything is testable offline. Providers go through the mock backend,
  HTTP through the `Transport` seam, storage through in-memory SQLite. Live
  round-trip tests are hidden behind the `[live]` Catch2 tag and skip themselves
  unless a backend is configured.

## Repository layout

```
include/agents_framework/   public headers (the API surface), one directory per module
src/                        implementation, mirroring include/
examples/                   four runnable demos (see docs/examples.md)
tests/                      Catch2 suite, deterministic and offline by default
docs/                       this documentation
PLAN.md                     the full design document: scope, build order, acceptance criteria
```
