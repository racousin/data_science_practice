# DS-Harness: build an AI system around a small model

Each task is an **objective** in English, sometimes with CSV files, and its answer is **one
number**: a computation stated in the text, a statistic over messy tables, or a regression
fitted on a training file. A language model of 1.5 billion parameters, asked directly, gets the
arithmetic wrong, never sees the data and drifts from the format. You build the **system** around
it:

- how the objective and its files are read;
- how the system decides what kind of task it faces;
- what reaches the model;
- which tools exist and how the model calls them;
- how the answer is parsed and checked;
- how the 40 seconds of each call are spent.

**An efficient system matters more than the model.** Measured on `dev.json` with the same 1.5B
model: asked directly with a well-parsed answer, **10.6**; with a calculator in a loop, **13.3**;
with a system that reads the files itself, shows the model a clean view of them and runs the code
it writes, **46.1**.

## The tasks

| Family | What | Share of the private set |
|---|---|---|
| No files | arithmetic, statistics of numbers given in the text, percentages, growth, logarithms and roots, dates, integers, word problems, probability | 45 tasks |
| Tables | a statistic, a filter, a group ranking, a join, a derived column — over one or two CSV files with a `data_dictionary.csv` | 50 tasks |
| Fits | a linear or logistic regression on `train.csv`: a prediction for a row of `test.csv`, a coefficient, R², a count | 25 tasks |

The files are messy, and the objective says how: separators and decimal commas, missing-value
tokens, a `-1` that means *not recorded*, units glued to values, duplicated rows, day-first dates,
a column to leave out. Headers are short codes that change from task to task; the objective names
columns by their dictionary **description**. The agent receives only the objective and the file
paths: no type, no format. The private set (120 tasks) has other draws, and table types that
`dev.json` does not have.

## The score

The objective states the rounding (*n decimals* → tolerance `1.5 × 10⁻ⁿ`, *an integer* →
`1e-6`). A task scores 1 inside the tolerance, else 0. **Score = 100 × the mean over the 120
private tasks.** The board also shows the mean without files, on tables and on fits.

## The rules

- **Contract.** `agent.py` defines `class Agent`: `__init__` loads the model (**60 s at most**);
  `solve(tasks)` gets 8 tasks `{"id", "objective", "files": [paths]}` and returns
  `[{"id", "answer": <a number>}]`. Formats: `schema.md` in the repository.
- **Time.** 15 calls of 8 tasks, **40 s per call**. **A call that raises or takes longer ends the
  run and the deployment fails, with no score.** Catch errors per task, keep a margin, answer a
  number anyway.
- **Runtime: choose PyTorch** when you submit. torch, transformers, accelerate, numpy, pandas,
  sympy. One RTX 4090, 3 CPUs, **3 GiB of RAM** (over it, the agent is killed: a failure), 128 MB
  of `/tmp`, no network.
- **Models.** Only these five are mounted: `Qwen/Qwen2.5-0.5B-Instruct`,
  `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`,
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen/Qwen3-1.7B`. Load them with a pinned
  `revision=` (`dsh.load_llm` does it).
- **Uploads.** `agent.py` and the modules it imports (e.g. `dsh.py`): up to 10 files, scanned.
- **Quota.** 5 deployments per person per rolling 24 h, across ML-Arena, failed ones included. A
  deployment is a test run (16 dev tasks), then the scored run: about 10 min plus the queue.
  Test locally first.
- **Teams of two.** Create the team on this page before your first submission.
- **Freeze: 2026-11-03 23:59** (a deployment queued before it counts). Orals on 2026-11-04.
- **Privacy.** Do not log or store the objectives or files of platform runs.

## Start here

1. **Dataset.** Download `dev.json` from this challenge: 180 tasks with their files, answers and
   types.
2. **Repository.** [github.com/racousin/ds-harness](https://github.com/racousin/ds-harness): the
   scorer the leaderboard runs, `localtest.py`, `schema.md`, and the kit (`dsh.py`,
   `stage1_direct.py`, `stage2_tool_loop.py`).
3. **Notebook.** [Open the starter notebook in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-project-ds-harness.ipynb)
   on a T4 GPU: the data, the first solution in detail, a tool and a loop, the directions.
4. **First submission.** `stage1_direct.py` as `agent.py`, with `dsh.py`:

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=194, files=["agent.py", "dsh.py"],
              runtime={"language": "python", "framework": "torch"},
              wait=True, timeout_sec=1800)
```

## How it is graded

The project is half of the course grade: the leaderboard (on a private set regenerated after the
freeze, marked against fixed reference systems) and an oral. Details and calendar: the course's
[Project module](https://ml-arena.com/courses/ms2a-machine-learning-practice/mlp-project).
