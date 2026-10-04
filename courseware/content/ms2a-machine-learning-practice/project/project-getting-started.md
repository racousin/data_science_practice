# Getting Started

From zero to a first scored submission in about an hour, then how to climb:
change one thing, measure it locally, keep it only if it pays.

<!-- notes: 30 minutes. Do the first hour live in Colab if the room has GPUs,
otherwise show the notebook with its outputs. Insist on two things: the PyTorch
runtime (a wrong pick costs a deployment) and the try/except per task (a raised
exception means no score at all). The second half is the ablation habit: it is
the largest part of the oral. -->

---

## Before you start

- A partner: the team of two is created on the
  [challenge page](https://ml-arena.com/viewchallenge/194)
- Your API key: Profile page on ML-Arena, it starts with `mlk_user_`
- A Google account for Colab, with a **T4 GPU** runtime
- Challenge id: **194**

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

`dev.json`: 180 tasks with their files, answers and types.

---

## Step 2: read before you code

Print one task of each family and read it in full, files included. Then read
`schema.md`.

- What does the agent receive? Only the objective and the file paths.
- Which columns does a table task need? The objective names them by their
  description in `data_dictionary.csv`, not by their header.
- Where does the objective say how to read the file, and how to round?

---

## Step 3: stage 1, measured

`stage1_direct.py`: the model reasons, ends with `ANSWER: <number>`, and the
parser reads that line, else the last number written.

```bash
python localtest.py stage1_direct.py --test-run --timeout 120
```

On the platform's GPU it scores **10.6** on `dev.json`: 26.2 without files,
2.5 on tables, 0.0 on fits. The notebook shows why each design
choice matters: an example answer gets copied, JSON breaks the reasoning, the
fallback parser is worth more than any prompt change.

`--timeout 120`: a T4 is about three times slower than the platform's RTX 4090.

---

## Step 4: stage 2, your first loop

`stage2_tool_loop.py` gives the model a calculator: a tool spec in the system
prompt, a call format, a parser, the execution, the observation sent back, a
stop rule. Its loop body is marked `TODO`: write it (about ten lines), then

```bash
python localtest.py stage2_tool_loop.py --type calc.functions --type calc.growth --timeout 120
```

Solved, it takes the tasks without files from 26.2 to 35.4. The tables and
fits do not move: the model still never reads the data.

---

## Step 5: the first submission

```python
import shutil
shutil.copy("stage1_direct.py", "agent.py")
client.submit(challenge_id=194, files=["agent.py", "dsh.py"],
              runtime={"language": "python", "framework": "torch"},
              submission_name="stage 1", wait=True, timeout_sec=1800)
```

From the console: upload the same two files and pick the **PyTorch** runtime.
A deployment is a test run (16 dev tasks), then the scored run (120 tasks):
about 10 minutes plus the queue.

---

## What makes a run fail

A run with no score costs a deployment and teaches you nothing.

- `solve` raises: wrap the work in `try/except`, answer `0` for a task you
  cannot finish
- a call takes more than 40 s: bound generation with `max_time`, keep a margin
  for the code you run after it (`dsh.Clock`)
- more than 3 GiB of RAM: the agent is killed
- a model outside the five, or `from_pretrained` without `revision=`: use
  `dsh.load_llm`
- `__init__` longer than 60 s
- the upload scan: `class Agent` anywhere but `agent.py`, or an `exec(...)` (run generated
  code with `dsh.run_python`)

---

## How to climb: the directions

| Direction | What it is |
|---|---|
| Routing | recognise the kind of task from the objective, send it to the right prompt and tools |
| Reading | parse each file yourself with the stated rules; give the model a clean view linked to the dictionary |
| Tools | data tools with arguments, or code the model writes and you run: measure both |
| Fits | a fitting tool (least squares, a few Newton steps for the logistic) beats code written from scratch |
| Checks | range, rounding, integer when asked; failed code sent back once with its error |
| Time | two batched model rounds fit in 40 s, a third may not |

The reference system (46.1 on `dev.json`) combines reading, a data view and
code execution with one repair round.

---

## The ablation habit

One change, one measurement, one row in a table.

| Run | Change | No files | Tables | Fits | Score | s / call |
|---|---|---|---|---|---|---|
| r0 | stage 1 | | | | | |
| r1 | + calculator loop | | | | | |
| r2 | + files read by the harness | | | | | |
| r3 | + fitting tool | | | | | |

Same task file, same model, every row. Keep the losers in the table: the oral
asks about them.

---

## Measure locally, not on the board

Five deployments a day is too few to learn from. The board confirms; your own
set decides.

- `localtest.py ... --out run.json` keeps each task's answer, gold, score and error
- sort the failures by cause: reading, column, computation, rounding, format, time
- write a generator for the types you work on, so you measure on hundreds of
  tasks, not on the 7 to 18 per type of `dev.json`
- when local and board scores differ, explain the gap

---

## Rules to keep in mind

- **5 deployments per person per rolling 24 h**, every ML-Arena challenge,
  failed ones included
- Do not log or store objectives or files from platform runs
- Do not edit `scoring.py` or `env.py`: the leaderboard runs its own copies
- Commit every run that you report, so the repository can reproduce it

---

## Check yourself

1. Your local score on `dev.json` is 8 points above your leaderboard score.
   Is the leaderboard wrong?

   **Answer.** No. You tuned on the dev draws, and the private set has other
   draws and table types that `dev.json` does not have. Your own generated
   validation set gives the honest estimate.

2. You add a third model round that fixes some answers, and the deployment
   fails. What happened?

   **Answer.** A call went over 40 s: the run ended. Measure seconds per call
   with the slowest tasks, bound generation with `max_time`, and keep a margin
   for the code you run after it.

3. Stage 2 gains points without files and none on tables. Why?

   **Answer.** The calculator fixes the arithmetic; on a table task the model
   never sees the data. Reading the files, and deciding what reaches the model,
   is the next step.
