# Lab 8 — Build and Judge a RAG Pipeline

Build a small retrieval-augmented system over a corpus you choose, then measure
it twice: retrieval recall against a hand-written question set, and answer
faithfulness with a pinned LLM judge.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.

<!-- notes: The trap is spending 40 minutes on the pipeline and 5 on the
evaluation. Announce at minute 20 that retrieval must already be measurable.
The judge prompt as a versioned file is the habit worth the whole lab. -->

---

## Setup

Work on a branch in your project repository.

```text
src/rag/
    chunking.py     <- deterministic splitter
    index.py        <- embed + search
    generate.py     <- prompt assembly + call
    evaluate.py     <- recall@k and the judge
prompts/
    answer_v1.txt
    judge_faithfulness_v1.txt
data/corpus/        <- gitignored
questions.jsonl     <- committed, this is the test set
results/ablation.md
```

`questions.jsonl` is committed; the corpus is not. The evaluation set is the
part of this lab with lasting value.

---

## Part A — Corpus and questions (8 min)

Pick 20–200 documents you care about: your notes, a library's docs, papers, a
wiki export. Anything the model cannot already know is better.

Then write **at least 15 questions** by hand, each recording the document and
chunk that answers it:

```json
{"q": "What does apply_chat_template do?", "doc": "peft.md", "gold": "sec-3"}
```

Include three questions whose answer is **not** in the corpus. A system that
cannot say "not in context" is not finished.

---

## Part B — Chunk and index (10 min)

`chunking.py` exposes one pure function:

```python
def chunk(text: str, size: int, overlap: int) -> list[Chunk]: ...
```

- returns a `Chunk` carrying `text`, `doc_id`, `chunk_id`, `start`, `end`
- deterministic: same input, same output, no randomness, no dict ordering
- `overlap < size`, asserted, raising on violation
- splits on paragraph boundaries where possible, word counts otherwise

`index.py` embeds with one named model, stores that name beside the vectors, and
refuses to search if the query model differs. Exact search is fine at this scale
— `faiss.IndexFlatIP` on normalised vectors.

---

## Part C — Retrieval recall (8 min)

Before touching generation, measure retrieval alone:

```python
def recall_at_k(questions, index, k: int) -> float:
    hits = sum(q["gold"] in [c.chunk_id for c in index.search(q["q"], k)]
               for q in questions)
    return hits / len(questions)
```

Report recall@1, recall@3 and recall@5. Below 0.6 at k=5, fix the chunker or the
embedding model now — no prompt work recovers evidence that was never retrieved.

---

## Part D — Generate and judge (12 min)

`prompts/answer_v1.txt` instructs the model to answer only from the context, to
cite chunk ids, and to emit exactly `NOT_IN_CONTEXT` when the answer is absent.
`prompts/judge_faithfulness_v1.txt` is a separate file scoring one answer against
its retrieved context:

```text
Score 0-2: 0 = contains a claim absent from the context,
1 = supported but incomplete, 2 = fully supported and cited.
Reply as JSON: {"score": <int>, "reason": "<one sentence>"}
```

The judge call pins the model id, sets `temperature=0`, logs both, and parses
with `json.loads` so a malformed response raises. Report the mean and the count
of zeros.

---

## Part E — The ablation (7 min)

Change **one** parameter and rerun both evaluations — chunk size
(200 / 400 / 800) or reranking (on / off). Write `results/ablation.md`:

| Variant | recall@5 | Faithfulness | NOT_IN_CONTEXT correct |
|---|---|---|---|
| chunk 200 / 50 | 0.73 | 1.41 | 2/3 |
| chunk 400 / 50 | 0.87 | 1.62 | 3/3 |
| chunk 800 / 50 | 0.80 | 1.35 | 1/3 |

Three numbers and one sentence saying which you would ship. A table with no
conclusion is not a result.

---

## Required tests

```python
def test_chunker_is_deterministic():
    """chunk(text, 400, 50) twice returns identical chunk ids and text."""

def test_chunker_respects_overlap_and_loses_nothing():
    """Consecutive chunks overlap by `overlap` words; concatenating the
    non-overlapping spans reproduces the source text exactly."""

def test_known_chunk_is_retrieved_in_top_k():
    """A question written against a fixture chunk retrieves it within k=3."""

def test_judge_prompt_is_versioned():
    """The judge prompt loads from prompts/judge_faithfulness_v1.txt and its
    sha256 matches the value pinned in the test."""
```

The last one is the point of the lab: pin the hash, and any edit to the rubric
fails the suite until you bump the version and re-run the baseline.

---

## Pull request

The description states:

- what the corpus is, how many documents and chunks
- recall@1/3/5 and the mean faithfulness score
- the ablation table and which variant you would ship
- one question the system gets wrong, with the retrieved chunks, and your
  diagnosis of whether it is a retrieval or a generation failure

---

## Grading

| Criterion | Weight |
|---|---|
| Deterministic chunker with overlap and boundary handling | 15% |
| Index refuses a model mismatch; retrieval works end to end | 15% |
| 15+ hand-written questions incl. three unanswerable | 15% |
| recall@k computed and reported | 15% |
| Judge with a versioned prompt, pinned model, temperature 0 | 20% |
| Ablation table with a stated conclusion | 10% |
| Four tests passing | 10% |

---

## Automatic deductions

- the judge prompt inlined in a notebook or an f-string in code
- the corpus or the vector index committed to git
- an API key in the source or in the history
- a bare `except` around a judge call, or a parse failure scored as zero
- reporting faithfulness without reporting retrieval

---

## Carry it forward

You now own an evaluation harness with a pinned judge — the instrument that
scores the project's LLM-judged track. Point it at your submission before the
leaderboard does.

The questions file is the durable artefact. Models change monthly; a test set
you wrote by hand keeps telling you the truth.
