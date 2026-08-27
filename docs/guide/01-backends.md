# Step 1 — Talk to a Model

Everything in the framework sits on one backend interface, `LLMBackend`, which
exchanges a canonical request/response format. The Anthropic backend, the
OpenAI-compatible backend, and the offline mock all implement it, so the code you
write in this step never changes when you switch providers.

## 1. Configure a backend

Backend selection is environment-driven. Create a `.env` file next to your binary
(or set real environment variables):

```ini
AF_BACKEND=anthropic          # or: openai, mock (the default)
ANTHROPIC_API_KEY=sk-ant-...
```

Then, in code, load the environment and construct whatever it names:

```cpp
(void)load_dotenv();  // reads .env; never overrides real environment variables

BackendOptions options;
options.max_tokens = 256;

auto opened = backend_from_env(std::move(options));
if (!opened) { /* opened.error() explains what's missing */ }
auto backend = *opened;  // unwrap the Result once; backend is a shared_ptr<LLMBackend>
```

With no configuration at all you get the mock backend, which is deterministic and
free — the right default while you build the graph around it.

## 2. Send a message

```cpp
ChatRequest request;
request.messages = {Message::user_text("In one sentence, what is a directed graph?")};
request.sampling.max_tokens = 128;

const auto reply = backend->generate(request);
if (reply) std::cout << reply->text() << "\n";
```

Recoverable failures — a missing key, a network error, a refusal to parse — come back
as `Result<T>` (`std::expected<T, Error>`), never as exceptions.

## 3. Stream the reply

The same request, delivered token by token. For the common case — you just want the
text — wrap a plain callback in `on_text`:

```cpp
const auto reply = backend->generate_stream(
    request, on_text([](std::string_view text) { std::cout << text << std::flush; }));
```

When you need more than text, pass a full `StreamCallback` instead: it receives every
`StreamEvent` — tool-call deltas, the stop reason and token usage, stream errors — as
a variant you match on.

## 4. Make the mock speak

For tests and demos you script the mock's side of the conversation with a handler,
which makes every run reproducible:

```cpp
BackendOptions options;
options.mock_handler = [](const ChatRequest&) -> Result<ChatResponse> {
  ChatResponse response;
  response.content.push_back(TextBlock{"A canned, deterministic reply."});
  return response;
};
```

This is the pattern the entire test suite uses to exercise provider behavior in CI
without a key or a network.

## Complete example

[`examples/chat_demo.cpp`](../../examples/chat_demo.cpp) runs all of the above plus a
full tool round-trip — which is where the next steps take over: instead of calling
`generate` by hand, you'll put the model inside a graph.

**Next:** [Step 2 — Build your first graph](02-first-graph.md)
