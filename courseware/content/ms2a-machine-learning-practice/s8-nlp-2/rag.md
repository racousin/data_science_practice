# RAG

Retrieval-Augmented Generation puts the facts in the prompt instead of in the
weights. It is the default answer to "the model does not know about our data",
and it is a search problem wearing an LLM costume.

<!-- notes: 30 minutes. Emphasise that most RAG failures are retrieval failures,
and that they are invisible unless retrieval is measured separately. The
chunk-size slide is the one they will actually use in the lab. -->

---

## What it fixes

![Retrieval-augmented generation: retrieve, then generate](assets/nlp/rag0.png)

| Problem | Why weights cannot fix it |
|---|---|
| Stale knowledge | the cutoff is baked in; retraining costs a run |
| Private data | it was never in the corpus, and must not be |
| No citations | a weight has no provenance |
| Hallucination | no way to say "not in my data" |

RAG fixes all four with one move: fetch relevant text at query time and put it
in the context window. Nothing is trained; the knowledge base updates by writing
a file.

---

## The pipeline

**Offline, per corpus change:** chunk the documents, embed each chunk, write
the vectors and their metadata to an index.

**Online, per query:** embed the query, retrieve the top-k chunks, rerank,
assemble a prompt, generate. The prompt states the grounding constraint, gives
an explicit escape hatch — reply `NOT_IN_CONTEXT` — and requires chunk-id
citations. Without an allowed way to fail, the model answers from its weights
and you cannot tell which answers came from your documents.

```python
chunks   = chunk(docs, size=400, overlap=50)
index    = build_index(embed(chunks))
hits     = index.search(embed(query), k=5)
answer   = llm(PROMPT.format(context=join(hits), question=query))
```

Six lines, five independent failure points — each measurable separately, which is
the discipline of this lesson.

---

## Chunking

![Splitting a document into overlapping chunks](assets/nlp/chunk.png)

```python
def chunk(text, size=400, overlap=50):
    assert overlap < size
    words = text.split()
    step, out = size - overlap, []
    for i in range(0, max(len(words) - overlap, 1), step):
        out.append(" ".join(words[i:i + size]))
    return out
```

The chunk is the unit of retrieval *and* the unit of context; those roles pull
in opposite directions, which is why no default always works. Split on structure
before falling back to a word count — a chunk beginning mid-sentence embeds
badly.

Two details in those five lines are the whole difference between a chunker and a
bug. `assert overlap < size` stops a negative step silently producing garbage.
And the loop stops at `len(words) - overlap`, not at `len(words)`: iterating to
the end emits a final chunk that is wholly contained in its predecessor — a
duplicate that goes straight into the index and competes with the real chunk for
the top-k slot.

---

## Size and overlap are a real trade-off

| Chunk size | Retrieval precision | Context quality |
|---|---|---|
| 100–200 tokens | high — one idea per vector | often too little to answer |
| 400–600 tokens | balanced; the usual default | usually enough |
| 1000+ | low — the topic vector is diluted | plenty, mostly irrelevant |

Overlap of 10–20% keeps a fact that straddles a boundary retrievable from both
sides; it costs storage linearly and is worth it.

> Tune chunk size against a question set, not against intuition. It is the
> highest-leverage parameter in the system, and the one most often left at
> whatever the tutorial used.

---

## Embedding

The retriever's ceiling is the embedding model. Two rules, both non-negotiable:

- **The same model must embed the corpus and the query.** Different models put
  vectors in unrelated spaces; the search still returns results, ranked by
  nothing.
- **Re-embed the whole corpus when you change the model.** Version the index by
  model name so a mismatch crashes instead of degrading.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")   # 384-dim, fast
vecs = model.encode(chunks, normalize_embeddings=True)
```

Start small and open, measure recall@k, then consider a hosted API.

---

## Vector search

![Semantic search over an embedding space](assets/nlp/vector-search.jpeg)

$$
sim(q, d) = \frac{q \cdot d}{|q|\, |d|} = \cos(\theta)
$$

Cosine similarity measures angle, not magnitude — what you want for text. On
normalised vectors it is a dot product, so retrieval is one matrix multiply.

```python
index = faiss.IndexFlatIP(384)     # exact, inner product on normalised vectors
index.add(vecs)
scores, ids = index.search(model.encode([query], normalize_embeddings=True), 5)
```

Exact search over 100k chunks is milliseconds on a laptop. Do not deploy a
vector database before you need one.

---

## ANN indexes

![Families of approximate nearest neighbour index](assets/nlp/indexing.png)

Above a few million vectors, scanning everything stops being free. Approximate
nearest neighbour indexes trade a little recall for a lot of speed.

- **IVF** — k-means the vectors, search only the nearest clusters; tuned by
  `nlist` and `nprobe`.
- **HNSW** — a navigable small-world graph walked from a coarse layer to a fine
  one. More memory, better recall at equal latency, no training step.

Default to HNSW, and measure recall against exact search on a sample: an ANN
index returning the wrong neighbours looks exactly like one that works.

---

## Hybrid search

Dense embeddings generalise across wording and fail on exact tokens: product
codes, error numbers, surnames, `ValueError`. BM25 is the opposite. Combine
them:

$$
score = \lambda \cdot BM25(q,d) + (1-\lambda) \cdot cos(q,d)
$$

Run both retrievers, take the top-k of each, and fuse the rankings — reciprocal
rank fusion avoids calibrating two incomparable scores.

Hybrid retrieval is the most reliable single upgrade to a mediocre RAG system.
If a query containing an identifier returns nothing useful, this is why.

---

## Reranking

Retrieval optimises recall over millions of chunks and cannot read each one
carefully. A cross-encoder can: it scores the query and one chunk *together*.

```python
from sentence_transformers import CrossEncoder
ranker = CrossEncoder("BAAI/bge-reranker-base")
ranked = sorted(zip(ranker.predict([(q, c) for c in top50]), top50),
                reverse=True)[:5]
```

Retrieve 50 cheaply, rerank to 5 accurately, generate from those — tens of
milliseconds, and it typically buys more than an embedding upgrade.

---

## Measure retrieval separately

Write 20–50 questions, recording which chunk answers each. Then:

$$
recall@k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|R_q \cap A_q^{k}|}{|R_q|}
$$

If recall@5 is 0.4, no prompt engineering and no larger model saves the answers:
60% of the time the evidence is not in the context at all.

Fix retrieval first, generation second. Debugging the generator while retrieval
is broken is the most common way to waste a week on a RAG system.

---

## Failure modes

| Symptom | Usual cause |
|---|---|
| Confident answer, wrong facts | chunk retrieved but self-contradictory |
| "I don't know" on covered topics | chunk too large, or embedding mismatch |
| Exact identifiers never found | dense-only retrieval; add BM25 |
| Answer ignores a retrieved chunk | evidence buried mid-context; rerank |
| Fine in the demo, fails in production | questions unlike the ones you tested |

Faithfulness — every claim supported by a retrieved chunk — is judged, not
computed: use the LLM-judge protocol from the previous lesson, same pinned
prompt. RAG does not eliminate hallucination; it makes it **detectable**,
because for the first time you hold the evidence the answer was meant to use.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   def chunk(text, size=400, overlap=50):
       assert overlap < size
       words = text.split()
       step, out = size - overlap, []
       for i in range(0, max(len(words) - overlap, 1), step):
           out.append(" ".join(words[i:i + size]))
       return out

   doc = " ".join(str(i) for i in range(360))
   print([len(c.split()) for c in chunk(doc, 400, 50)])   # -> [360]
   print([len(c.split()) for c in chunk(doc, 200, 50)])   # -> [200, 200, 60]

   cs = chunk(doc, 200, 50)
   print(cs[0].split()[-50:] == cs[1].split()[:50])       # -> True
   rebuilt = cs[0] + "".join(" " + " ".join(c.split()[50:]) for c in cs[1:])
   print(rebuilt == " ".join(doc.split()))                # -> True
   ```

   **Answer.** Consecutive chunks share exactly `overlap` words, and dropping the
   overlap from every chunk after the first rebuilds the document — *up to
   whitespace*, because `text.split()` has already thrown the paragraph breaks
   away. That caveat is why the Lab 8 test says "up to whitespace normalisation"
   and not "exactly".

2. recall@5 on your question set is 0.4. Someone proposes rewriting the answer
   prompt and switching to a larger generator. What will that buy you?

   **Answer.** Nothing. Sixty per cent of the time the evidence is not in the
   context at all, so no prompt and no generator can recover it. Fix retrieval
   first — chunking, hybrid search, reranking — and only then touch generation.

3. A query containing an error code or a product reference returns nothing
   useful, while paraphrased questions work fine. What is the diagnosis and the
   standard fix?

   **Answer.** Dense-only retrieval. Embeddings generalise across wording and are
   poor on exact tokens. Add BM25 and fuse the two rankings — hybrid retrieval is
   the most reliable single upgrade to a mediocre RAG system.

4. Why must the answer prompt give the model an explicit way to fail?

   **Answer.** Without an allowed escape hatch — `NOT_IN_CONTEXT` — the model
   answers from its weights when retrieval misses, and you can no longer tell
   which answers came from your documents. The escape hatch is what makes
   hallucination detectable.
