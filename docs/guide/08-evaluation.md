# Step 8: Measure the Agent

An agent you cannot measure is an agent you cannot improve. The eval harness runs an
agent over a task suite, scores every instance with a verifier, and reports accuracy
with a confidence interval: the number all later changes are judged against.

## 1. Get a task suite

A `TaskSuite` supplies instances (with train / held-out / retention splits) and a
verifier. Implement the interface for your own domain, or start with the shipped
text-to-SQL reference suite:

```cpp
auto suite = *TextToSqlSuite::create(default_text_to_sql_spec());
```

Its verifier executes the predicted and gold SQL against multiple database instances
and compares result sets, so a query that is wrong but happens to match on one
dataset is still caught.

## 2. Wrap your agent

The harness drives anything shaped `AgentFn`: task instance in, output (plus an
optional trace) out. Wrapping a graph or a single model call looks the same:

```cpp
AgentFn agent = [&](const TaskInstance& instance) -> Result<AgentOutput> {
  ChatRequest request;
  request.system = "...";  // your prompt
  request.messages.push_back(Message::user_text(instance.input.value("question", "")));
  AF_TRY(const auto response, backend->generate(request));

  AgentOutput output;
  output.output = response.text();
  return output;
};
```

Attach a `Trace` to the output and the harness persists it. That is what makes the
dataset export in step 5 possible.

## 3. Run the harness

```cpp
auto traces = *TraceStore::open(db);
auto evals  = *EvalStore::open(db);

EvalOptions options;
options.split   = Split::Train;
options.workers = 4;          // instances fan out on the thread pool
options.traces  = &traces;
options.store   = &evals;

const auto report = *run_eval(suite, agent, options);
```

The report is pinned to everything needed to reproduce it (seed, model, framework
version, git commit), and every per-instance result is persisted so two runs can be
diffed later:

```text
accuracy   0.833  [0.436, 0.970] 95% CI
pinned     seed=0 version=0.0.1 commit=0bb5473e4ca7 eval_id=01M1...
```

The confidence interval is load-bearing: on a small suite, a few lucky answers move
the mean by several points. Without the interval you cannot tell improvement from
noise.

## 4. Record a baseline and compare

```cpp
(void)evals.set_baseline(report.eval_id);

// later, after changing the agent:
const auto delta = *compare(baseline_report, candidate_report);
// delta.delta, delta.ci, delta.significant
```

`compare` returns a signed delta with its own interval: the honest answer to "did
that change help?".

## 5. Turn verified teacher runs into training data

Run a *strong* model through the harness and the traces the verifier accepted are
exactly the examples a smaller model should be trained on:

```cpp
TraceQuery query;
query.suite = report.suite;
query.verified = true;

const auto verified = *traces.select(query);
const auto dataset  = *build_dataset(verified, {});
(void)write_jsonl(dataset, "dataset.jsonl");
```

This is the bridge to the framework's core idea: better models train worse ones. The
upcoming `distill` feature consumes precisely this kind of dataset, with a teacher
authoring the examples and a verifier gating what gets in. A model is never trained
on its own outputs; the supervision always comes from something stronger.

## Complete example

[`examples/eval_demo.cpp`](../../examples/eval_demo.cpp) runs this whole chapter,
including two deliberately wrong answers, so you can watch the verifier catch a
near-miss that string comparison would have accepted.

**You're done.** For the reference view of everything the guide touched, read
[Architecture](../architecture.md); for where the project goes next, read
[PLAN.md](../../PLAN.md).
