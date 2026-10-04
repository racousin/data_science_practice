# The Brief

The project is half the grade of this course and it is one challenge:
**DS-Harness**. You build the program around a small language model so that it
solves data-science tasks it could never solve alone.

<!-- notes: 20 minutes. Open the challenge page (https://ml-arena.com/viewchallenge/194)
on the projector and show the leaderboard while you talk. The ladder slide is the
one to dwell on: a model under 2B parameters goes from 0.6 to about 61, mostly
through the code around it. Teams of two are declared by 2026-10-15. -->

---

## Why a harness

A model of 1 to 3 billion parameters reads a question about a spreadsheet well.
It cannot compute a standard deviation over 600 rows in its head, fit a model or
forecast a series, and its arithmetic slips.

A **harness** is the program around the model:

- it routes each task to the right approach
- it gives the model tools: code execution, a calculator, your own helpers
- it captures the result, checks it, retries when it fails

Small model, big toolbox. This is how most production LLM systems are built.

---

## The challenge

Each task is a question in English, sometimes with CSV files, and one answer.

| Level | Example | Scored |
|---|---|---|
| L1 | a unit conversion, a percentage, a column statistic | exact, tolerance |
| L2 | a word problem, a probability, filter → group → rank | exact, tolerance |
| L3 | forecast a series, predict a target for test rows | 0 baseline → 1 ref |

**Score = 100 × (0.3·L1 + 0.4·L2 + 0.3·L3)**, each level a mean over its tasks.

Public: `dev.json`, 178 tasks with answers. Private: 119 tasks, with other
wordings, other data domains, and families you have never seen.

---

## What you build

One file, `agent.py`, plus the modules it imports:

```python
class Agent:
    def __init__(self):          # load the model: 60 s at most
        ...
    def solve(self, tasks):      # a batch of tasks
        return [{"id": t["id"], "answer": ..., "trace": "..."}
                for t in tasks]
```

It runs offline on one 24 GB GPU, with only the models of the platform's
cache, and about **330 s** to answer all 119 tasks. A call that raises or runs
out of time ends the run: **no score**.

---

## The ladder, measured on the private set

| Agent | Score |
|---|---|
| A0: the model answers directly (Qwen2.5-1.5B) | 0.6 |
| A1: the kit, the model writes Python, one repair | 13.9 |
| the kit with Qwen3-1.7B instead (one line) | 27.6 |
| A2: instructor harness: routing, numpy tools, checks | ≈ 61 |

Same size of model, a hundred times the score. The ceiling is 100.

On the *Unseen families* column every agent measured stays between 8 and 21.
That is where the top of the board is decided.

---

## What you are given

- **The repository** [github.com/racousin/ds-harness](https://github.com/racousin/ds-harness):
  the scorer the leaderboard runs, local test tools, `schema.md`, the kit
- **The kit** `dsh.py`: model loading, code sandbox, calculator, answer
  parsing, a stopwatch. A0 and A1 are built on it
- **The starter notebook** on Colab: from the first evaluation to a
  first submission
- **The challenge page**: rules, models, and the live leaderboard

Everything above the kit is yours.

---

## How you are graded, in one slide

| Part | Share of the project |
|---|---|
| Leaderboard, against fixed anchors (A0, A1, A2) | 25 % |
| Oral, both members, 10 min + 10 min questions | 75 % |

The project is **50 %** of the course grade.

The leaderboard mark uses a final private set, regenerated after the freeze.
The oral weighs your evaluation method most. Details in *Deliverables and
Grading*.

---

## Pairs, and the rules that matter

- Teams of **two**, created on the challenge page, declared by **2026-10-15**
- **5 deployments per person per rolling 24 h**, across every ML-Arena
  challenge, failed ones included: test locally first
- Choose the **PyTorch** runtime when you submit
- No scikit-learn, scipy or statsmodels on the platform
- Do not log or store task content from platform runs

---

## Timeline

| Date | Milestone |
|---|---|
| now | leaderboard open |
| 2026-10-15 | teams declared |
| 2026-11-16 | checkpoint: 1-page design + ablation draft (feedback) |
| **2026-11-20 23:59** | **freeze**: last deployment, slides, repo tag |
| 2026-11-21 → 11-23 | final runs on the regenerated private set |
| 2026-11-24 → 11-27 | orals |

Next lesson: *Getting Started*, your first hour, step by step.

---

## Check yourself

1. A0 and A2 both run a model under 2B parameters. Where does the difference
   of about 60 points come from?

   **Answer.** Mostly from the program around the model: routing, tools the
   model calls instead of computing in its head, checks, and the use of the
   time budget. A better model alone takes the kit to 27.6, not 61.

2. Your agent answers 110 tasks well, then one `solve` call raises. What is your
   score?

   **Answer.** None. A call that raises or misses its timeout ends the run and
   the deployment fails. Catch errors per task and return a placeholder.

3. Why does the *Unseen families* column matter more than the public tasks?

   **Answer.** The private set has families absent from `dev.json`. A harness
   keyed on the public wordings fails there; one that reads the prompt and the
   files carries over. The final set is regenerated, so only general harnesses
   keep their score.
