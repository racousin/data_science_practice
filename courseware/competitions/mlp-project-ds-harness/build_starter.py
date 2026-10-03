#!/usr/bin/env python3
"""Write the Colab starter of the DS-Harness project challenge (challenge 194).

    python build_starter.py

Output: website/public/modules/ms2a-machine-learning-practice/challenges/
mlp-project-ds-harness.ipynb (opened from GitHub in Colab, like the other
challenge notebooks). Never edit the .ipynb by hand: change this file and run it.

The notebook is guided, not a solution: setup, load dev.json, run the two kit
agents on a few tasks, see where the time goes, analyse the errors, build a
validation set, submit the kit baseline, then how to progress.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = (HERE.parents[2] / "website" / "public" / "modules" / "ms2a-machine-learning-practice"
       / "challenges" / "mlp-project-ds-harness.ipynb")

CELLS = [
("md", r"""# DS-Harness starter

**A small language model, inside a harness, solving data-science tasks under a time budget.**

Each task is a question in English, sometimes with CSV files: a unit conversion, a word problem,
a chain of filters and aggregations over messy tables, a forecast or a prediction for a test
file. A 1.5B model asked directly scores 0.6 / 100. Your project is the program around the
model, the **harness**: it routes each task, prompts the model, runs the code the model writes,
checks the result and recovers from errors, all **within a few seconds per task**.

This notebook runs end to end on a free Colab T4: get the code and the data, look at the tasks,
run the two kit agents, see where the time goes, analyse the errors, build a validation set and
submit the kit baseline. The last section says how to go further. Challenge page:
[ml-arena.com/viewchallenge/194](https://ml-arena.com/viewchallenge/194)."""),

("md", r"""---

## 0. Setup

**Use a GPU runtime:** Runtime → Change runtime type → **T4 GPU**. On a CPU a 1.5B model takes
tens of seconds per task.

The ML-Arena client is published as **`mlarena-sdk`** and imports as `mlarena` (do not
`pip install mlarena`, an unrelated package). Colab already has torch, transformers, pandas and
numpy."""),

("code", r"""!pip install -q mlarena-sdk accelerate"""),

("md", r"""---

## 1. Get the code and the data

The code is the public repository [racousin/ds-harness](https://github.com/racousin/ds-harness):
the scorer the leaderboard runs, two local runners and the starter kit.

| file | what it is |
|---|---|
| `scoring.py` | the scorer and the delivery loop (`run_agent`) the leaderboard runs |
| `schema.md` | the task and answer formats, the scoring, the delivery |
| `localtest.py` | runs an agent file on a task file as the platform does |
| `local_eval.py` | the same, plus timings, full traces (`--out`) and `--slowdown` |
| `dsh.py` | the starter kit: model loading, code sandbox, calculator, parsing, stopwatch |
| `agent_naive.py` | the model answers directly |
| `agent_kit_baseline.py` | the model writes Python, the kit runs it, one repair round |
| `kit_README.md` | the kit's documentation: read it |"""),

("code", r"""!git clone -q https://github.com/racousin/ds-harness"""),

("code", r"""import os
os.chdir("ds-harness")
print(sorted(os.listdir(".")))"""),

("md", r"""Your personal API key is on your ML-Arena **Profile** page (it starts with `mlk_user_`). The
cell asks for it, so it is not saved in the notebook. `download_dataset` writes `dev.json`: the
178 public tasks, with their answers."""),

("code", r"""from getpass import getpass

import mlarena

CHALLENGE_ID = 194   # https://ml-arena.com/viewchallenge/194
client = mlarena.connect(api_key=getpass("ML-Arena API key (mlk_user_...): "))
print(client.download_dataset(CHALLENGE_ID, "."))"""),

("md", r"""Do not edit `scoring.py`: the leaderboard runs its own copy, so a change here only makes your
local numbers wrong. `dsh.py` is yours to copy and change: you upload the version your agent
imports."""),

("code", r"""import json
import random
import time

import pandas as pd

import dsh
import scoring

dev = json.load(open("dev.json"))
print(len(dev), "tasks")"""),

("md", r"""---

## 2. The tasks

Each task has the keys your agent receives (`id`, `prompt`, `files`, `answer_type`) and keys only
you see in `dev.json` (`family`, `level`, `answer`, `scoring`, `heldout_family`)."""),

("code", r"""tasks = pd.DataFrame([{"id": t["id"], "level": t["level"], "family": t["family"],
                       "answer_type": t["answer_type"], "n_files": len(t["files"])} for t in dev])
tasks.groupby(["level", "family", "answer_type"]).size().rename("tasks").reset_index()"""),

("code", r"""def show(task, n_chars=400):
    print(f"{task['id']}  level {task['level']}  {task['family']}  -> {task['answer_type']}")
    print(task["prompt"], "\n")
    for name, text in task["files"].items():
        print(f"--- {name} ({text.count(chr(10))} lines)")
        print(text[:n_chars])
    answer = task["answer"]
    print("\nanswer:", answer if not isinstance(answer, list) else f"{answer[:5]} ... ({len(answer)} values)")
    print("scoring:", task["scoring"])

for level in (1, 2, 3):
    show(next(t for t in dev if t["level"] == level))
    print("\n" + "=" * 100 + "\n")"""),

("md", r"""**Score = 100 × (0.3·L1 + 0.4·L2 + 0.3·L3)**, each level being the mean over its tasks. Level 1
and 2 are exact answers with a tolerance. A level-3 task scores between 0 (a trivial baseline)
and 1 (a reference model): the prompt states the method and the format, so reading it carefully
pays more than heavy modelling. `schema.md` gives every format and tolerance.

**Questions.** Which columns does the level-2 task need, and how are its missing values written?
In the level-3 task, which column must not be used? What must a model get right, in order, to
answer each of the three?"""),

("md", r"""---

## 3. The time budget

The private set has **119 tasks: 37 at level 1, 67 at level 2, 15 at level 3**. They arrive in
**21 `solve` calls**: 8 level-1/2 tasks or 2 level-3 tasks per call.

- **Each call** has a timeout of 2.5 × (2 s per level-1/2 task + 8 s per level-3 task): 40 s for
  a full batch.
- **The job** ends 399 s after it starts, model loading included: about 380 s of answering after
  a ~17 s load. **Plan for 330 s of answering**, about 2 s per level-1/2 task and 8 s per level-3
  task.
- Every call gets its full timeout. When the full timeout of the next batch no longer fits
  before the deadline, that batch is not sent and its tasks score 0.
- **A `solve` call that raises an exception or misses its timeout ends the run, and the
  deployment fails: no score at all.** Catch errors per task, answer a placeholder rather than
  raise, and stop at the task's `time_budget_s`.

The per-call timeouts add up to far more than the job has: the binding limit is the total. The
cell below recomputes these numbers with the scorer's own functions, on 119 dev tasks drawn with
the private set's level mix (`private_like`)."""),

("code", r"""PRIVATE_MIX = {1: 37, 2: 67, 3: 15}
TOTAL_S = 330   # seconds of answering to plan for

def sample_like_private(tasks, n_total, seed=0):
    # n_total tasks of `tasks` with the private set's level proportions
    rng = random.Random(seed)
    out = []
    for level, n in PRIVATE_MIX.items():
        pool = [t for t in tasks if t["level"] == level]
        out += rng.sample(pool, round(n * n_total / sum(PRIVATE_MIX.values())))
    return out

def planned_seconds(tasks):
    # the scorer's planning pace: 2 s per level-1/2 task, 8 s per level-3 task
    return sum(scoring.EST_TASK_S[t["level"]] for t in tasks)

private_like = sample_like_private(dev, 119)
batches = scoring.make_batches(private_like)
timeouts = [scoring.batch_budget(b) for b in batches]
print(f"{len(private_like)} tasks in {len(batches)} calls")
print(f"sum of the per-call timeouts: {sum(timeouts):.0f} s; plan for {TOTAL_S} s of answering")
print(f"at 2 s / 8 s per task: {planned_seconds(private_like):.0f} s")
for b, t in list(zip(batches, timeouts))[:4]:
    print(f"  call of {len(b)} level-{b[0]['level']} tasks: timeout {t:.0f} s")"""),

("md", r"""Each task also carries `time_budget_s`, its share of **its call's** timeout: 5 s for a
level-1/2 task, 20 s for a level-3 task. That is the limit that keeps a call alive, not the pace
to plan for: a harness that spends 5 s on every level-1/2 task answers well under half of the
private set. `dsh.Budget.for_tasks(tasks)` turns `time_budget_s` into a stopwatch.

What costs time on a GPU: loading the model (once, 60 s at most); **generating tokens**, by far
the most (one batched `generate` for a whole call is much cheaper than one per task); running
code (`dsh.run_python` starts a fresh Python process, about 0.5–1.5 s); any second attempt."""),

("md", r"""---

## 4. Run the two kit agents

`local_eval.py` runs an agent file through `scoring.run_agent`, the platform's own delivery and
scoring, and records each call's duration. The model is set by `DSH_MODEL` (default
`Qwen/Qwen2.5-1.5B-Instruct`); the challenge page lists the models the platform has.

40 tasks with the private level mix are enough to compare agents in a few minutes (the first run
also downloads the model). A T4 is slower than the platform's RTX 4090, so `--slowdown 3`
multiplies every timeout by 3: this run measures what the agent **can** do. Section 5 checks
whether it is fast enough."""),

("code", r"""os.environ["DSH_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"

sample = sample_like_private(dev, 40)
json.dump(sample, open("sample40.json", "w"))
json.dump(private_like, open("private_like.json", "w"))
print({lv: sum(t["level"] == lv for t in sample) for lv in (1, 2, 3)})"""),

("code", r"""!python local_eval.py agent_naive.py --dev sample40.json --slowdown 3 --out run_naive.json | tail -20"""),

("code", r"""!python local_eval.py agent_kit_baseline.py --dev sample40.json --slowdown 3 --out run_kit.json | tail -20"""),

("code", r"""def load_run(path):
    run = json.load(open(path))
    df = pd.DataFrame(run["details"])
    df["level"] = df["id"].map({t["id"]: t["level"] for t in dev})
    return run, df

runs = {name: load_run(f"run_{name}.json") for name in ("naive", "kit")}
pd.DataFrame({name: {"score": run["score"], **{f"L{k}": round(v, 1) for k, v in run["level_scores"].items()},
                     "format errors": run["metrics_detail"]["format_errors"],
                     "model load (s)": round(run["init_s"], 1)}
              for name, (run, _) in runs.items()})"""),

("md", r"""---

## 5. Where the time goes

`call_seconds` is the duration of the call a task was in, so a task costs its call's duration
divided by the tasks in that call. Projected onto the 119 private tasks, it says whether the
harness answers everything in 330 s."""),

("code", r"""def timing(df):
    per_call = df.groupby("call_index")["id"].transform("count")   # tasks in the same call
    df = df.assign(task_s=df["call_seconds"] / per_call)
    by_level = df.groupby("level")["task_s"].mean()
    projected = sum(PRIVATE_MIX[lv] * by_level.get(lv, 0.0) for lv in PRIVATE_MIX)
    return by_level.round(2), projected

for name, (run, df) in runs.items():
    by_level, projected = timing(df)
    verdict = "fits" if projected <= TOTAL_S else "DOES NOT FIT"
    print(f"{name:6s} s per task by level {by_level.to_dict()} -> 119 tasks in {projected:.0f} s: "
          f"{verdict} in {TOTAL_S} s (model load {run['init_s']:.0f} s, limit 60 s)")"""),

("md", r"""**The rehearsal.** Run the agent at the platform's own timeouts (no `--slowdown`) with the
330 s total (`--budget-s 330`), on `private_like.json`: the 119 tasks with the private mix, so
the run looks like the platform's. `--budget-s` is the budget of the private mix: on another
task file the runner scales it to the tasks you run (about 575 s for the full `dev.json`).

The last line of the output says how many tasks were sent and answered. "RUN ENDED EARLY" means a
call raised or missed its timeout: on the platform, that deployment fails with no score. The T4
is usually slower than the platform's GPU, so a rehearsal that fits here has a margin there."""),

("code", r"""!python local_eval.py agent_kit_baseline.py --dev private_like.json --budget-s 330 --out run_kit_rehearsal.json | tail -3"""),

("md", r"""**Questions.** How much of a call is generation, how much code execution (time
`dsh.run_python` alone)? How does the time per task change with `max_new_tokens`, with the batch
size, with a repair round? Which tasks could skip the model, or the sandbox?"""),

("md", r"""---

## 6. Error analysis

Per-family scores say where to work; traces say why a task failed: the reading of the prompt (a
wrong column, a missed cleaning rule), the reasoning, the code, the answer format, or time."""),

("code", r"""run, df = runs["kit"]
fam = df.groupby(["level", "family"]).agg(tasks=("id", "count"), score=("score", "mean"),
                                         format_errors=("error", lambda e: e.notna().sum()))
fam.round(2)"""),

("code", r"""failed = df[df["score"] < 1].sort_values(["level", "family"])
print(len(failed), "tasks below full score")
failed[["id", "level", "family", "score", "error"]].head(20)"""),

("code", r"""def trace(task_id, run_df=df):
    row = run_df.set_index("id").loc[task_id]
    task = next(t for t in dev if t["id"] == task_id)
    print(task["prompt"], "\n")
    print("gold:", task["answer"] if not isinstance(task["answer"], list) else task["answer"][:8])
    print("got: ", row["answer"] if not isinstance(row["answer"], list) else row["answer"][:8])
    print("score", row["score"], "|", row["error"], "\n")
    print(row["trace"])

trace(failed["id"].iloc[0])"""),

("md", r"""Keep a table of failures by cause (reading, reasoning, code, format, time). The oral asks for a
per-family error analysis and a few traced failures with what fixed them."""),

("md", r"""---

## 7. Build a validation set

`dev.json` is what you tune on, so it overestimates your private score: the private set uses
other wordings, other data domains and task families that are not in `dev.json`. A validation set
you never tune on is the honest estimate, and the oral asks how you built it.

One way is a generator: a function that draws random data, writes the prompt and computes the
answer with code. The example makes level-2 tasks: a share over the rows whose value is known."""),

("code", r"""import numpy as np

def make_share_task(rng, task_id):
    n = int(rng.integers(40, 120))
    missing = rng.choice(["n/a", "-", "unknown"])
    city = rng.choice(["Lyon", "Lille", "Nantes", "Rennes"], size=n)
    price = np.round(rng.lognormal(3.0, 0.5, size=n), 2).astype(object)
    price[rng.random(n) < 0.1] = missing
    df = pd.DataFrame({"shop": [f"S{i:03d}" for i in range(n)], "city": city, "price_eur": price})
    threshold = int(rng.integers(15, 30))
    target = str(rng.choice(sorted(set(city))))

    known = df[df["price_eur"] != missing]
    rows = known[known["city"] == target]
    answer = round(100 * (rows["price_eur"].astype(float) > threshold).mean(), 1)
    prompt = (f"shops.csv lists one product price per shop, in euros; a missing price is written "
              f"'{missing}'. Among the shops of {target} whose price is known, what percentage "
              f"charge more than {threshold} euros? Round to 1 decimal place.")
    return {"id": task_id, "prompt": prompt, "files": {"shops.csv": df.to_csv(index=False)},
            "answer_type": "number", "family": "mine.share", "level": 2,
            "answer": float(answer), "scoring": {"kind": "numeric", "abs_tol": 0.051, "rel_tol": 1e-6},
            "heldout_family": False}

rng = np.random.default_rng(0)
mine = [make_share_task(rng, f"m_{k:03d}") for k in range(16)]
for t in mine:
    scoring.validate_task(t)   # the scorer's own format check
json.dump(mine, open("mine.json", "w"))
show(mine[0], n_chars=200)"""),

("md", r"""`local_eval.py` needs all three levels in a task file, so a one-family set is scored task by
task: call the agent's `solve` directly, then the scorer's own `score_answer`. This loads the
model in the notebook itself."""),

("code", r"""import importlib

agent = importlib.import_module("agent_kit_baseline").Agent()
payload = [dict({k: t[k] for k in scoring.AGENT_KEYS}, time_budget_s=15.0) for t in mine]
t0 = time.monotonic()
replies = {r["id"]: r for r in agent.solve(payload)}
seconds = time.monotonic() - t0
scores = [scoring.score_answer(t, replies[t["id"]]["answer"])[0] for t in mine]
print(f"{sum(scores)}/{len(mine)} correct, {seconds / len(mine):.1f} s per task")"""),

("md", r"""**Questions.** Do your generated tasks score like the `dev.json` family they resemble? How
many wordings, column names and domains does a generator need before the score stops moving?
Which other questions is a data scientist asked every day? Write generators for them, keep one
set aside, never tune on it, and report it at the oral."""),

("md", r"""---

## 8. Submit

**Match the platform first.** Its Python has torch, transformers, accelerate, pandas, numpy,
sympy and matplotlib, and **no scikit-learn, scipy or statsmodels**. Colab has them, so code
that imports them runs here and fails there. Put this guard in front of the code your harness
runs locally (`dsh.run_python`) to make Colab behave like the platform:"""),

("code", r"""PLATFORM_GUARD = '''
import sys
class _Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sklearn", "scipy", "statsmodels"):
            raise ImportError(f"{name} is not installed on the platform")
sys.meta_path.insert(0, _Blocked())
'''

res = dsh.run_python(PLATFORM_GUARD + "import sklearn\nRESULT = 1", {})
print(res.ok, res.error.strip().splitlines()[-1])"""),

("md", r"""**What a submission is.** `agent.py` (it defines `class Agent`) plus the modules it imports,
here `dsh.py`: up to 10 files, 100 MB. The runtime is **PyTorch**: the cell below picks it, and
in the console choose it yourself (the default is not torch, and a wrong pick costs a
deployment).

**Quota: 2 deployments per person per rolling 24 h, counted across every ML-Arena challenge,
failed ones included.** Rehearse locally first (section 5). You work in pairs: create your team
on the challenge page before your first submission.

Each deployment is a short test run on a few dev tasks, then a scored run on the private set.
One job takes about 7–9 minutes, plus the queue (jobs run one at a time). The cell prints the
progress and ends when the deployment has settled."""),

("code", r"""import shutil

shutil.copy("agent_kit_baseline.py", "agent.py")   # replace with your own agent
result = client.submit(challenge_id=CHALLENGE_ID, files=["agent.py", "dsh.py"],
                       runtime={"language": "python", "framework": "torch"},
                       submission_name="kit baseline")
submission_id = result["submission_id"]
for line in client.tail_logs(CHALLENGE_ID, submission_id, timeout_sec=3600):
    print(line)"""),

("code", r"""final = client.status()
print("status:", final["status"], "|", final["last_status_message"])
if final["status"] == "deploy_failed":
    print("no score:", final["latest_deploy"]["failure_message"])
for run in final["run_info"]["results"]:
    mine_row = next(r for r in run["submission_results"] if r["submission_id"] == submission_id)
    kind = "test run  " if run["is_test"] else "scored run"
    print(kind, run["job_status"], "| score", mine_row["score"], "|", mine_row["info_message"])"""),

("md", r"""If the wait times out, the deployment goes on: run the last cell again later, or look at the
challenge page. The scored run's message says how many tasks were sent and answered."""),

("md", r"""---

## 9. How to progress

Measured on the private set:

| Agent | Score | L1 / L2 / L3 | What changed |
|---|---|---|---|
| `agent_naive.py` (Qwen2.5-1.5B, answers directly) | 0.6 | 0 / 1.5 / 0 | — |
| `agent_kit_baseline.py` (writes Python, one repair) | 13.9 | 32.4 / 10.4 / 0 | the kit as shipped: 103 of 119 answered in time |
| an earlier kit with `Qwen/Qwen3-1.7B` | ≈ 31 | 57 / 34 / 0 | one line |
| the instructor's harness (not published) | ≈ 61 | 65 / 40 / 85 | routing, numpy fit / forecast tools, checks |

The ceiling is 100. The *Unseen families* column stays at 8–21 for every agent measured: that
is where the top of the board is decided.

The first levers that pay:
- **The model.** Try the platform's models (`DSH_MODEL`); measure accuracy against seconds per
  task.
- **Level 3.** The kit scores 0 there. A small numpy fit / forecast tool that the model calls
  with the arguments the prompt states is about 40 lines.
- **Answer format.** Check the type, the length, the rounding before you reply.
- **Time.** The kit already uses most of the budget: adding a repair round to an earlier
  version of the kit lowered its score from 19.5 to 14.7, because tasks went unanswered.

Measure every change on your own validation set (section 7), with its time cost (section 5), and
keep the table: it is the ablation study of your oral.

**Questions.** Which tasks need the model at all, which need code, which a calculator? What does
the model see of each file? Which checks catch a wrong answer before it is sent? When is a
second attempt worth its time? What in your harness is specific to the public families?"""),
]


def to_source(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build() -> dict:
    cells = []
    for kind, text in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code",
                "metadata": {}, "source": to_source(text)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(CELLS)} cells)")
