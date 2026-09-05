# Lab 8 — Build and Judge a RAG Pipeline

Build a small retrieval-augmented system over a corpus you choose, then measure
it twice: retrieval recall against a hand-written question set, and answer
faithfulness with a pinned LLM judge.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository,
and one scored run on the session's competition (Part F).

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

## Setup — pick your judge

The judge is the instrument of Parts D and E, and it is the one thing this lab
does not hand you. Choose before minute 20:

- **Local, and the default.** No account, no key, about a gigabyte of weights on
  CPU: `Qwen/Qwen2.5-0.5B-Instruct` with `do_sample=False` — the second snippet
  in *Evaluating Generation*, which runs as written. The whole lab is
  completable this way.
- **Hosted.** The course issues no API keys. If you already have one, export it
  and build the client with no arguments, so the key is never in the source and
  never in the history:

```bash
export ANTHROPIC_API_KEY=...          # never in a file git can see
uv pip install anthropic
```

Either way the model id goes into `results/ablation.md` beside every score. A
faithfulness number with no model id next to it is not a measurement, and the
ablation in Part E compares two runs of the *same* judge or it compares nothing.

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

The judge call pins the model id, switches sampling off, logs the id, and parses
with `json.loads` so a malformed response raises. "Sampling off" is
`do_sample=False` locally; the hosted Anthropic SDK no longer accepts a
`temperature` argument at all, which is why pinning an exact dated model id is
the whole of your reproducibility story there. Report the mean and the count of
zeros.

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

## Part F — Put it on the board (5 min here, the run at home)

The session's competition is **GSM8k**, `competition_id=165` — grade-school math
word problems against a private test set. It is the rarest thing in Session 8: a
generation task whose answer is exactly checkable.

At test time the environment draws **K = 5 disjoint subsets of 10 questions** and
sends each as one batch with a **60-second** timeout. Score is the count of
replies matching the gold answer to within `1e-6`, out of the **50** questions
sent. Higher is better, and the ceiling is 50.

```python
class Agent:
    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        ...    # (numeric answers, reasoning traces) — one entry per question,
               # in the input order. A bare list[float] is still accepted.
```

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.submit(competition_id=165, files=["agent.py"])
print(client.leaderboard(165).head())
```

---

## Three things the board will not tell you

**The served template does not run.** `agent_template.py` is truncated: its
`answer()` calls `self._solve_one(q)` and that method is missing, so an unedited
template raises `AttributeError: 'Agent' object has no attribute '_solve_one'` on
the first question. Write `_solve_one` yourself before you deploy.

**The floor is a crash, not a wrong answer.** The creator's reference row,
`__benchmark__`, scores **0.0 of 50** across 224 runs; the best of 866 entries
scores **18.0 of 50**. A zero almost always means a crash or a timeout, because
the first failed batch stops every later subset. Get one correct answer end to
end before you try for ten.

**The GPU may be down.** 165 runs on an external GPU VM. If your agent sits in
`queued`, check `client.competition(165)["engine"]["vm_health_ok"]` — `False`
means the machine is unreachable and nothing will run until it is back. It read
`False` on 2026-09-02; say so on the course channel rather than resubmitting.

The habits from this session transfer whole: the prompt is a versioned file, the
model id is pinned and logged, decoding is greedy, and a reply that will not
parse to a float raises rather than silently scoring zero.
`Qwen/Qwen2.5-0.5B-Instruct` — the local judge from *Evaluating Generation* — is
on the competition's pre-populated model cache, so it does not have to be
downloaded inside your batch budget.

---

## Required tests

```python
def test_chunker_is_deterministic():
    """chunk(text, 400, 50) twice returns identical chunk ids and text."""

def test_chunker_respects_overlap_and_loses_nothing():
    """Consecutive chunks overlap by `overlap` words, and concatenating the
    non-overlapping spans reproduces the source text up to whitespace
    normalisation — `text.split()` has already discarded the paragraph
    breaks, so exact reconstruction is not on offer. No chunk is a subset
    of its predecessor."""

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
- your competition 165 score and the agent name it is under, or the reason no
  run completed

---

## Grading

| Criterion | Weight |
|---|---|
| Deterministic chunker with overlap and boundary handling | 15% |
| Index refuses a model mismatch; retrieval works end to end | 15% |
| 15+ hand-written questions incl. three unanswerable | 15% |
| recall@k computed and reported | 15% |
| Judge with a versioned prompt, a pinned and logged model id, sampling off | 20% |
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

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] `questions.jsonl` is committed with 15 or more hand-written questions, each carrying `doc` and `gold`, and exactly three of them unanswerable from the corpus (Part A)
- [ ] `data/corpus/` and the vector index are gitignored, and `git log -p` contains no API key
- [ ] `index.py` raises when the query embedding model differs from the one stored beside the vectors (Part B)
- [ ] recall@1, recall@3 and recall@5 are printed by `evaluate.py` and recorded in the PR, and recall@5 is at or above 0.6 (Part C)
- [ ] Every judge score in `results/ablation.md` carries the model id it was produced with, and a malformed judge reply raises instead of scoring zero (Part D)
- [ ] `results/ablation.md` holds three rows differing in exactly one parameter, and one sentence naming the variant I would ship (Part E)
- [ ] All four required tests pass, `test_judge_prompt_is_versioned` included, with the sha256 pinned in the test
- [ ] My submission is on the leaderboard of GSM8k (#165) — `client.leaderboard(165)` lists my agent name
- [ ] My score beats the reference: **more than 0 of 50 correct**, the `__benchmark__` row. If nothing ran, `client.competition(165)["engine"]["vm_health_ok"]` is `False` and the PR says so.

If the last two are not ticked you have not finished the lab, however good the
code is.
