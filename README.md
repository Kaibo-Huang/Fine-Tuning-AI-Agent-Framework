# Fine-Tuning AI Agent Framework

[![CI](https://github.com/Kaibo-Huang/Fine-Tuning-AI-Agent-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaibo-Huang/Fine-Tuning-AI-Agent-Framework/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++23](https://img.shields.io/badge/C%2B%2B-23-informational)

A C++23 framework for **creating, orchestrating, and fine-tuning AI agents** — think
LangChain + LangGraph, but as a native systems library with direct control over
scheduling, state, persistence, and (soon) the training stack itself.

The core idea: **training and orchestration live in the same process, so fine-tuning
becomes a tool call.** An agent's own execution traces — verified tool results,
successful completions, failure/correction pairs — are the dataset, and the graph can
decide to train, evaluate against a baseline, and hot-swap or roll back an adapter as
ordinary graph operations.

## What works today

- **LLM backend abstraction** — one canonical request/response/tool-call format over
  the **Anthropic Messages API**, any **OpenAI-compatible** server (OpenAI, OpenRouter,
  vLLM, llama.cpp), and a deterministic **mock backend** for offline development and CI.
  Blocking and token-streaming paths, native function calling with a tolerant
  text-protocol fallback.
- **Typed state graphs** — shared state is a compile-time schema of named channels with
  reducers (`Channel<"messages", std::vector<Message>, Append>`). No codegen, no
  hot-path type lookups.
- **Deterministic concurrent execution** — a Pregel-style super-step scheduler: active
  nodes run in parallel on a thread pool, outputs merge at a barrier in a deterministic
  order. Conditional edges, cycles with step budgets, and reproducible runs given the
  same inputs and seeds.
- **Persistence & time travel** — every super-step checkpoints to embedded SQLite:
  resume, replay, fork any past step, and pause for human approval
  (`interrupt_before`) with the run resuming from the stored checkpoint.
- **Tools** — native C++, subprocess, and HTTP tools behind one JSON-schema registry
  with argument validation.
- **Memory & RAG** — an `EmbeddingBackend` seam, a SQLite-backed vector store, and a
  retrieval node type.
- **Multi-agent** — agents are graphs; mount one inside another as a subgraph node with
  typed state translation in and out. Supervisor/hand-off patterns are just edges.
- **Traces, datasets & evaluation** — structured trace capture into SQLite, a
  `TaskSuite` seam for pluggable domains, built-in verifiers (exact match, numeric
  tolerance, result-set comparison, subprocess exit code, LLM-judge), and an eval
  harness that fans an agent across a suite concurrently and reports
  **accuracy ± 95% confidence interval**, pinned to the model, seed, framework version,
  and commit that produced it. Verified traces export directly as JSONL training data.

## Roadmap

The training stack lands next, as three verbs a graph can call:

| Verb | What it does | Status |
|---|---|---|
| `finetune` | LoRA SFT on a dataset — external files or the agent's own traces | planned (Phase 5) |
| `distill` | a larger model — or an entire multi-node graph — teaches one small fast model | planned (Phase 5) |
| `port` | carry a task adaptation onto a **new base model** from a small calibration set, instead of retraining from scratch (a C++ port of [portallib](https://github.com/ramp-public/portallib)) | planned (Phase 6) |

Phase 4 (LibTorch inference, `safetensors` weight loading, LoRA, local + remote
training runners) is the current focus.

## Documentation

- **[Orchestrating Agents, Step by Step](docs/guide/README.md)** — the guide: start
  from a raw model call and build up through graphs, tools, streaming, checkpoints,
  retrieval, multi-agent composition, and evaluation, one step at a time
- **[Architecture](docs/architecture.md)** — the reference view: modules, the
  typed-channel state model, the deterministic super-step executor, persistence, and
  the conventions the code follows
- **[Examples walkthrough](docs/examples.md)** — what each demo shows, what to look at
  in the code, and how to point it at a live provider
- **[PLAN.md](PLAN.md)** — the full design document: scope, build order, and
  acceptance criteria for every phase

## Quick start

### Prerequisites

- CMake ≥ 3.23
- [vcpkg](https://github.com/microsoft/vcpkg), with the `VCPKG_ROOT` environment
  variable pointing at it (dependencies — libcurl, nlohmann-json, spdlog, sqlite3,
  Catch2 — install automatically on first configure)
- Windows: Visual Studio 2022 (MSVC x64) · Linux: GCC 13+ or Clang 17+, and Ninja

### Build and test

```bash
# Windows (MSVC x64)
cmake --preset windows-msvc
cmake --build --preset windows-msvc
ctest --preset windows-msvc

# Linux (Ninja)
cmake --preset linux
cmake --build --preset linux
ctest --preset linux
```

### Run the examples

Every example runs fully offline against the mock backend by default — no API key
needed. Binaries land in `build/windows-msvc/examples/Debug/` (Windows) or
`build/linux/examples/` (Linux). Each one is walked through in detail in
[docs/examples.md](docs/examples.md).

| Example | What it shows |
|---|---|
| `chat_demo` | the backend abstraction: a reply, a token stream, and a full tool round-trip |
| `react_demo` | a minimal ReAct agent — an LLM node and a tool node routed in a loop |
| `orchestration_demo` | a supervisor delegating to subgraph agents, with RAG retrieval, event streaming, SQLite checkpoints, and a human-in-the-loop pause/resume |
| `eval_demo` | the measurement stack: a text-to-SQL task suite scored with confidence intervals, baseline recording, and verified traces exported as training data |

To point the same code at a live provider, create a `.env` file (or set the
environment) as described below.

## A taste of the API

The whole ReAct loop from `react_demo`, minus the printing:

```cpp
using Messages   = Channel<"messages", std::vector<Message>, Append>;
using AgentSchema = Schema<Messages>;

GraphBuilder<AgentSchema> builder;
builder.add_node("agent", make_llm_node<AgentSchema>(backend, options))
    .add_node("tools", make_tool_node<AgentSchema>(registry))
    .set_entry("agent")
    .add_conditional_edge("agent", tools_router<AgentSchema>("tools"),
                          {"tools", std::string{kEnd}})
    .add_edge("tools", "agent");
auto graph = std::move(builder).compile();

State<AgentSchema> state;
state.set<"messages">({Message::user_text("What is 987654321 times 123456789?")});

Executor executor;
auto stats = executor.run(*graph, state, RunOptions{.max_steps = 10});
```

Recoverable failures return `Result<T>` (`std::expected<T, Error>`) throughout;
exceptions are reserved for broken invariants.

## Configuration

Configuration is environment-driven. A `.env` file in the working directory is loaded
automatically and never overrides variables already set in the environment.

| Variable | Values | Meaning |
|---|---|---|
| `AF_BACKEND` | `mock` (default) · `anthropic` · `openai` | which backend to use |
| `AF_MODEL` | a model id | override the backend's default model |
| `AF_BASE_URL` | a URL | point the OpenAI-compatible backend at OpenRouter, vLLM, llama.cpp, … |
| `ANTHROPIC_API_KEY` | key | required for `AF_BACKEND=anthropic` |
| `OPENAI_API_KEY` | key | required for `AF_BACKEND=openai` |

Secrets are redacted from logs and error messages.

## Testing

`ctest` runs the full offline suite; provider backends are exercised through the mock,
so no network or API key is ever required. Live round-trip tests exist but are hidden
behind the `[live]` tag and skip themselves unless a backend is configured:

```bash
AF_BACKEND=openai build/linux/tests/agents_framework_tests "[live]"
```

CI builds and tests every push on Windows (MSVC) and Linux (GCC). The live tests are
an opt-in toggle on the workflow's manual dispatch.

## License

[MIT](LICENSE) © Kaibo Huang
