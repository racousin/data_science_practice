# The Brief

The project is half the grade of this course and it is one challenge:
**DS-Harness**. You build an AI system around a small language model so that
it answers data-science objectives it could never answer alone.

<!-- notes: 20 minutes. Open the challenge page (https://ml-arena.com/viewchallenge/194)
on the projector and show the leaderboard while you talk. The ladder slide is the
one to dwell on: the same 1.5B model goes from 10.6 to 46.1 on dev.json through the
system around it, not through a bigger model. Teams of two are declared by 2026-10-15. -->

---

## Why a system around the model

A model of 1.5 billion parameters reads an objective about a spreadsheet well.
It cannot compute a standard deviation over 300 rows it never sees, its
arithmetic slips, and its answer drifts from the format you asked for.

The **system** is the program around the model. It decides:

- how the objective and its files are read
- what kind of task this is
- what reaches the model
- which tools exist and how the model calls them
- how the answer is parsed and checked
- how the 40 seconds of a call are spent

**An efficient system matters more than the model.** This is how most
production LLM systems are built.

---

## The challenge

Each task is an objective in English, sometimes with CSV files, and its
answer is **one number**.

| Family | Example |
|---|---|
| No files | the sample standard deviation of 8 numbers; compound interest; the day of the week of a date |
| Tables | the median amount billed of the stays of the cardiology department, from a messy CSV and its data dictionary |
| Fits | fit a logistic regression on `train.csv`, give the predicted probability for one row of `test.csv` |

The objective states the rounding, and how to read the files: separators,
missing values, units, duplicates, a column to leave out.

---

## The score

- inside the stated rounding the task scores 1, otherwise 0
- **score = 100 × the mean over the 120 private tasks**
- the board also shows the mean without files, on tables and on fits

Public: `dev.json`, 180 tasks with their files, answers and types. Private:
120 tasks, other draws, and table types `dev.json` does not have.

---

## What you build

One file, `agent.py`, plus the modules it imports:

```python
class Agent:
    def __init__(self):            # load the model: 60 s at most
        ...
    def solve(self, tasks):        # 8 tasks: {"id", "objective", "files"}
        return [{"id": t["id"], "answer": 41.07} for t in tasks]
```

It runs offline on one RTX 4090 with one of five models (0.5B to 1.7B
parameters). 15 calls of 8 tasks, **40 s per call**. A call that raises or
runs out of time ends the run: **no score**.

---

## The ladder, measured on dev.json

| System | Model | Score | No files | Tables | Fits |
|---|---|---|---|---|---|
| stage 1: answer marked `ANSWER:`, parsed with a fallback | Qwen2.5-1.5B | **10.6** | 26.2 | 2.5 | 0.0 |
| stage 1 | Qwen3-1.7B | **16.7** | 44.6 | 0.0 | 2.9 |
| stage 2: + a calculator in a loop | Qwen2.5-1.5B | **13.3** | 35.4 | 1.2 | 0.0 |
| stage 2 | Qwen3-1.7B | **22.2** | 60.0 | 1.2 | 0.0 |
| stage 2 with routing: the loop without files, stage 1 with files | Qwen2.5-1.5B | **13.3** | 35.4 | 1.2 | 0.0 |
| reference: files read by the system, data view, code, one repair | Qwen2.5-1.5B | **46.1** | 86.2 | 28.8 | 11.4 |
| reference | Qwen2.5-Coder-1.5B | **53.3** | 81.5 | 47.5 | 14.3 |
| reference | Qwen3-1.7B | **62.2** | 80.0 | 62.5 | 28.6 |

Same model, 4 times the score. The ceiling is 100.

---

## What you are given

- **The repository** [github.com/racousin/ds-harness](https://github.com/racousin/ds-harness):
  the scorer the leaderboard runs, `localtest.py`, `schema.md`, the kit
- **The kit** `dsh.py`: model loading, answer parsers, a calculator, a
  tool-call parser, a code runner, a clock
- **Two stages**: `stage1_direct.py` (the model, a structured answer, a
  parser) and `stage2_tool_loop.py` (a tool and a loop, the loop body to write)
- **The starter notebook** on Colab: the data, stage 1 in detail, stage 2 as a
  skeleton, the directions

Everything above the kit is yours.

---

## How you are graded, in one slide

| Part | Share of the project |
|---|---|
| Leaderboard, against fixed reference systems | 25 % |
| Oral, both members | 75 % |

The project is **50 %** of the course grade. The leaderboard mark uses a final
private set, regenerated after the freeze. Details in *Deliverables and
Grading*.

---

## Pairs, and the rules that matter

- Teams of **two**, created on the challenge page, declared by **2026-10-15**
- **5 deployments per person per rolling 24 h**, across every ML-Arena
  challenge, failed ones included: test locally first
- Choose the **PyTorch** runtime when you submit
- Do not log or store task content from platform runs

---

## Timeline

| Date | Milestone |
|---|---|
| now | leaderboard open |
| 2026-10-15 | teams declared |
| **2026-11-03 23:59** | **freeze**: last deployment queued, slides (PDF), repository tag |
| 2026-11-04 | final runs on the regenerated private set, then orals |

Next lesson: *Getting Started*, your first hour, step by step.

---

## Check yourself

1. Stage 1 and the reference system run the same 1.5B model. Where does the
   difference come from?

   **Answer.** From the system: it reads the files itself, gives the model a
   clean view of the data, lets it compute with code instead of in its head,
   parses and checks the answer, and spends the 40 s on what pays.

2. Your agent answers 110 tasks well, then one `solve` call raises. What is your
   score?

   **Answer.** None. A call that raises or runs out of time ends the run and
   the deployment fails. Catch errors per task and answer a number anyway.

3. Why can a system that scores well on `dev.json` lose points on the private
   set?

   **Answer.** The private set has other draws and table types that
   `dev.json` does not have. A system keyed on dev wordings fails there; one
   that reads the objective and the files carries over.
