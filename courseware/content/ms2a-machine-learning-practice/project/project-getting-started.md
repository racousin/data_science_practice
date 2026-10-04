# Getting Started

From zero to a first scored submission in about an hour, then how to climb:
change one thing, measure it locally, keep it only if it pays.

<!-- notes: 30 minutes. Do the first hour live in Colab if the room has GPUs,
otherwise show the notebook with its outputs. Insist on two things: the PyTorch
runtime (a wrong pick costs one of their two daily deployments) and the
try/except per task (a raised exception means no score at all). The second half
is the ablation habit: it is 30 % of the oral. -->

---

## Before you start

- A partner: the team of two is created on the
  [challenge page](https://ml-arena.com/viewchallenge/194)
- Your API key: Profile page on ML-Arena, it starts with `mlk_user_`
- A Google account for Colab, with a **T4 GPU** runtime
- Challenge id: **194**

Budget the hour: about 20 minutes of model download and local runs, about
10 minutes for the submission to be scored.

---

## Step 1: the notebook and the data

Open the [starter notebook in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-project-ds-harness.ipynb).
It clones [racousin/ds-harness](https://github.com/racousin/ds-harness) and
downloads the public tasks:

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.download_dataset(194, ".")      # writes dev.json
```

`dev.json`: 178 tasks with their answers.

---

## Step 2: read before you code

Print three tasks, one per level, and read them in full. Then read `schema.md`.

- Which columns does the task need? Where does the prompt say how to clean
  them, how to round, which format to answer in?
- At level 3, which method does the prompt ask for, and what goes in the
  answer?

Most lost points are a right value in a wrong format, or a rule in the
prompt that the harness ignored.

---

## Step 3: run the kit locally

```bash
python local_eval.py agent_naive.py --limit 30 --slowdown 3
python local_eval.py agent_kit_baseline.py --limit 30 --slowdown 3
```

30 tasks spread over the levels. You see a score per level and the seconds
per task. `--slowdown 3` gives each call three times its timeout, because a
T4 is slower than the platform's GPU. Look at the traces of a few failures.

---

## Step 4: the dress rehearsal

The notebook builds `private_like.json`: 119 dev tasks with the private mix
(37 / 67 / 15). Run it under the platform's rules:

```bash
python local_eval.py agent_kit_baseline.py \
    --dev private_like.json \
    --budget-s 330 --out run.json
```

No `--slowdown` here: these are the platform's own timeouts. The last line
says how many tasks were answered before the budget ran out. `--budget-s 330`
is the budget of the 119-task private mix; on another task file the runner
scales it (about 575 s for the full `dev.json`).

---

## Step 5: the first submission

```python
import shutil
shutil.copy("agent_kit_baseline.py", "agent.py")
client.submit(challenge_id=194, files=["agent.py", "dsh.py"],
              runtime={"language": "python", "framework": "torch"},
              submission_name="kit baseline",
              wait=True, timeout_sec=1800)
```

From the console: upload the same two files and pick the **PyTorch**
runtime. The kit scores 13.9 (L1 32.4, L2 10.4, L3 0) and answers 103 of 119 tasks before time runs out. One job takes 7 to 9
minutes plus the queue.

---

## What makes a run fail

A run with no score costs a deployment and teaches you nothing.

- `solve` raises: wrap each task in `try/except`, return a placeholder
- `solve` is too slow: stop at `t["time_budget_s"]`, answer what you have
- an import of `sklearn`, `scipy` or `statsmodels`: Colab has them, the
  platform does not
- a model not in the cache, or `from_pretrained` without `revision=`: use
  `dsh.load_llm`
- `__init__` longer than 60 s

---

## How to climb: the first levers

| Lever | What it is | Measured |
|---|---|---|
| Model | change one line in the kit | 13.9 → 27.6 with Qwen3-1.7B |
| Level 3 | a numpy fit / forecast tool, ~40 lines | kit: 0 at L3 |
| Format | answer exactly as `schema.md` says | exact match |
| Time | the kit answers 103 / 119 in time: make it faster | +16 tasks |

The instructor harness (≈ 61) combines these with routing and checks. Level 3
rewards reading the spec, not heavy modelling.

---

## Time is a lever, in both directions

The kit already uses most of the budget. When we added a second repair round
to an earlier version of it, the score **dropped** from 19.5 to 14.7: each task got slower, and the
tasks at the end of the run went unanswered.

- measure seconds per task, per level, before anything else
- give the time where the points are: a level-3 task is worth 2.0 points,
  a level-2 task 0.6
- keep a margin: a batch that does not fit before the deadline scores 0

---

## The ablation habit

One change, one measurement, one row in a table.

| Run | Change | L1 | L2 | L3 | Score | Time |
|---|---|---|---|---|---|---|
| r0 | kit as shipped | | | | | |
| r1 | + Qwen3-1.7B | | | | | |
| r2 | + numpy forecast tool | | | | | |
| r3 | − second repair | | | | | |

Same task file, same budget, same seed for every row. Keep the losers in the
table: the oral asks about them.

---

## Measure locally, not on the board

Five deployments a day is too few to learn from. The board confirms; your own
set decides.

- `local_eval.py ... --out run.json` keeps per-task scores, timings, traces
- sort the failures by cause: reading, reasoning, code, format, time
- build your own validation tasks with new wordings and new data, so the
  score does not reward memorising `dev.json`
- when local and board scores differ, explain the gap

---

## Rules to keep in mind

- **5 deployments per person per rolling 24 h**, every ML-Arena challenge,
  failed ones included
- Do not log or store prompts or files from platform runs
- Do not edit `scoring.py` or `env.py`: the leaderboard runs its own copies
- Commit every run that you report, so the repository can reproduce it

---

## Check yourself

1. Your local score on `dev.json` is 8 points above your leaderboard score.
   Is the leaderboard wrong?

   **Answer.** No. You tuned on the dev wordings, and the private set has
   other wordings, other data and unseen families. Dev scores run higher.
   Your own validation set gives the honest estimate.

2. You add a self-check round that fixes 5 % of the answers it sees, and the
   leaderboard score drops. What happened?

   **Answer.** Probably time: the extra round made each task slower and the
   last batches were not sent, so they scored 0. Measure seconds per task and
   the number of tasks answered, not just accuracy.

3. Your first submission failed at deployment. Name three likely causes you
   could have caught locally.

   **Answer.** An import of scikit-learn, scipy or statsmodels; an exception
   escaping `solve`; a model outside the cache or loaded without `revision=`.
   The wrong runtime is the fourth, caught by reading the submit form.
