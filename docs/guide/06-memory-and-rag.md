# Step 6 — Add Memory and Retrieval

Short-term memory is already solved: it is the conversation living on a state
channel. Long-term memory is a vector store plus a retrieval node — this step wires
both.

## 1. Choose an embedding backend

Embeddings sit behind their own seam, `EmbeddingBackend`, with the same
offline-first pattern as chat backends. The mock produces deterministic vectors of a
fixed dimension:

```cpp
constexpr std::size_t kDims = 64;
auto embedder = std::make_shared<MockEmbeddingBackend>(kDims);
```

Swap in the OpenAI-compatible embeddings backend for real vectors; nothing
downstream changes.

## 2. Open a vector store and index a corpus

The vector store lives in the same SQLite database as everything else:

```cpp
auto vectors = std::make_shared<VectorStore>(*VectorStore::open(db, "knowledge", kDims));

for (const auto& [id, content] : corpus) {
  const auto embedded = embedder->embed({"", {content}});
  if (embedded) (void)vectors->upsert({id, embedded->embeddings.front(), content, {}});
}
```

Each entry is an id, a vector, the original text, and optional metadata.

## 3. Add retrieval to the graph

The state needs two more channels — the query going in and the documents coming out:

```cpp
using Query = Channel<"query", std::string>;
using Docs  = Channel<"documents", std::vector<Document>>;
using ResearchSchema = Schema<Query, Docs, Messages>;
```

The retrieval node embeds the query and writes the top-k matches to `documents`; a
small compose node turns them into a prompt; the LLM node answers grounded in them:

```cpp
Update<ResearchSchema> compose_prompt(StateView<ResearchSchema> view) {
  const std::string context = render_documents(view.get<"documents">());
  return Update<ResearchSchema>{}.write<"messages">(
      {Message::user_text(context + "\nQuestion: " + view.get<"query">())});
}

GraphBuilder<ResearchSchema> builder;
builder
    .add_node("retrieve", make_retrieval_node<ResearchSchema>(embedder, vectors,
                                                              RetrievalOptions{.k = 2}))
    .add_node("compose", compose_prompt)
    .add_node("answer", make_llm_node<ResearchSchema>(backend, llm_options))
    .set_entry("retrieve")
    .add_edge("retrieve", "compose")
    .add_edge("compose", "answer")
    .set_finish("answer");
```

That three-node chain — retrieve, compose, answer — **is** RAG. Because it is an
ordinary graph, everything from the previous steps applies unchanged: it streams, it
checkpoints, it can pause for approval.

## Complete example

The research agent inside
[`examples/orchestration_demo.cpp`](../../examples/orchestration_demo.cpp) is exactly
this graph, seeded with a three-document corpus (one of them a decoy that retrieval
should rank last).

**Next:** [Step 7 — Compose agents as subgraphs](07-multi-agent.md)
