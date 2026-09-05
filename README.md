# Fine-Tuning AI Agent Framework

[![CI](https://github.com/Kaibo-Huang/Fine-Tuning-AI-Agent-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaibo-Huang/Fine-Tuning-AI-Agent-Framework/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++23](https://img.shields.io/badge/C%2B%2B-23-informational)

A C++23 framework for creating, orchestrating, and fine-tuning AI agents. Think
LangChain + LangGraph as a native systems library, with the training stack in the
same process.

The core idea: **use better LLMs to train worse ones.** A model cannot train itself,
so every transfer moves capability from something stronger into a small, cheap
artifact:

| Verb | The transfer |
|---|---|
| `finetune` | a curated dataset becomes a LoRA adapter |
| `distill` | a better model, or an entire multi-node graph, teaches one small fast model |
| `port` | an existing adaptation moves to a new base model via a thin alignment refit (a C++ port of [portallib](https://github.com/ramp-public/portallib)) |

The orchestration engine makes those transfers cheap and measurable: teacher graphs
generate the data, the eval harness scores students against recorded baselines with
confidence intervals, and adapters hot-swap or roll back.

## What works today

- **LLM backends**: Anthropic, any OpenAI-compatible server (OpenAI, OpenRouter,
  vLLM, llama.cpp), and a deterministic offline mock, all behind one canonical
  format. Streaming, native tool calling, and a text-protocol fallback.
- **Typed state graphs**: compile-time channel schemas with reducers. No codegen, no
  hot-path lookups.
- **Deterministic concurrency**: a Pregel-style super-step scheduler. Parallel nodes,
  deterministic merges, reproducible runs.
- **Persistence**: every step checkpoints to SQLite. Resume, replay, fork any past
  step, pause for human approval.
- **Tools**: native C++, subprocess, and HTTP tools behind one JSON-schema registry.
- **Memory and RAG**: pluggable embeddings, a SQLite vector store, a retrieval node.
- **Multi-agent**: agents are graphs; mount one inside another with typed state
  translation.
- **Measurement**: task suites, verifiers, and an eval harness reporting accuracy
  with 95% confidence intervals, pinned for reproducibility. Verified teacher traces
  export as JSONL training data.

## Incoming Features

The orchestration engine and the measurement stack are done. Next, in order:

1. **The training pipeline**: LibTorch inference, `safetensors` loading, LoRA, local
   and remote training runners.
2. **`finetune` and `distill`**, gated by the eval harness against recorded
   baselines.
3. **`port`**: task latents and the hypernetwork stack.

## Documentation

- **[Orchestrating Agents, Step by Step](docs/guide/README.md)**: the guide, from a
  raw model call to a measured multi-agent system.
- **[Architecture](docs/architecture.md)**: modules, the execution model, and
  conventions.
- **[Examples walkthrough](docs/examples.md)**: what each demo shows and how to run
  it live.

## Quick start

### Prerequisites

- CMake 3.23 or newer
- [vcpkg](https://github.com/microsoft/vcpkg), with `VCPKG_ROOT` set (dependencies
  install automatically on first configure)
- Windows: Visual Studio 2022 (MSVC x64). Linux: GCC 13+ or Clang 17+, and Ninja.

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

Every example runs fully offline against the mock backend; no API key needed.
Binaries land in `build/windows-msvc/examples/Debug/` (Windows) or
`build/linux/examples/` (Linux). Details in [docs/examples.md](docs/examples.md).

| Example | What it shows |
|---|---|
| `chat_demo` | the backend abstraction: a reply, a token stream, a tool round-trip |
| `react_demo` | a ReAct agent from the prebuilt graph, three calls end to end |
| `orchestration_demo` | a supervisor with subgraph agents, RAG, streaming, checkpoints, and a human-in-the-loop pause |
| `eval_demo` | a task suite scored with confidence intervals, a recorded baseline, traces exported as training data |

## A taste of the API

The whole ReAct agent from `react_demo`:

```cpp
auto graph = make_react_agent(backend, registry,
                              {.system = "Use the calculator tool for arithmetic."});

auto state = chat_state("What is 987654321 times 123456789?");

Executor executor;
auto stats = executor.run(*graph, state, {.max_steps = 10});

std::cout << last_assistant_text(state.get<"messages">()) << "\n";
```

Every demo starts with one include, `agents_framework/prelude.hpp`, which pulls in
the whole library and opens its namespaces. Library code includes the module headers
it needs instead.

Under the hood that is an ordinary typed graph: two nodes, a conditional edge, an
appending messages channel. Custom agents use the same `GraphBuilder` the prebuilt
uses; the [guide](docs/guide/README.md) builds this exact loop by hand.

Recoverable failures return `Result<T>` (`std::expected<T, Error>`); exceptions are
reserved for broken invariants.

## Configuration

Environment-driven. A `.env` file in the working directory is loaded automatically
and never overrides real environment variables. Secrets are redacted from logs.

| Variable | Values | Meaning |
|---|---|---|
| `AF_BACKEND` | `mock` (default), `anthropic`, `openai` | which backend to use |
| `AF_MODEL` | a model id | override the backend's default model |
| `AF_BASE_URL` | a URL | point the OpenAI-compatible backend at OpenRouter, vLLM, llama.cpp |
| `ANTHROPIC_API_KEY` | key | required for `AF_BACKEND=anthropic` |
| `OPENAI_API_KEY` | key | required for `AF_BACKEND=openai` |

## Testing

`ctest` runs the full suite offline; providers are exercised through the mock, so no
key or network is required. Live round-trip tests hide behind the `[live]` tag and
skip themselves unless a backend is configured:

```bash
AF_BACKEND=openai build/linux/tests/agents_framework_tests "[live]"
```

CI builds and tests every push on Windows (MSVC) and Linux (GCC); the live tests are
an opt-in toggle on manual dispatch.

## License

[MIT](LICENSE) © Kaibo Huang
