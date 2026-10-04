#!/usr/bin/env python3
"""Write the Colab starter of the DS-Harness project challenge (challenge 194).

    python build_starter.py

Output: website/public/modules/ms2a-machine-learning-practice/challenges/
mlp-project-ds-harness.ipynb (opened from GitHub in Colab, like the other
challenge notebooks). Never edit the .ipynb by hand: change this file and run it.

Sections (plan §6): setup; the data; stage 1 in detail (model + structured answer
+ parser); stage 2 as a skeleton (a tool and a loop); directions; measure; submit.
The measured numbers quoted come from M below (GPU VM runs on dev.json, the
platform's agent image and GPU).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = (HERE.parents[2] / "website" / "public" / "modules" / "ms2a-machine-learning-practice"
       / "challenges" / "mlp-project-ds-harness.ipynb")

# Measured on dev.json (180 tasks), RTX 4090, the platform's torch agent image (anchors2/README.md).
M = json.loads((HERE / "anchors2" / "measured.json").read_text())

CELLS = [
("md", r"""# DS-Harness starter

**Build an AI system around a small language model.**

Each task is an objective in English, sometimes with CSV files, and its answer is **one number**:
a computation stated in the text, a statistic over messy tables, or a regression fitted on a
training file. A model of 1.5 billion parameters asked directly gets the files wrong, the
arithmetic wrong and the format wrong. Your project is the system around it:

- how the objective and its files are read;
- how the system decides what kind of task it faces;
- what reaches the model;
- which tools exist and how the model calls them;
- how the answer is parsed and checked;
- how the 40 seconds of each call are spent.

**An efficient system matters more than the model.** On `dev.json`, the same 1.5B model goes from
%(s1_q15)s / 100 (asked directly, answer well parsed) to %(a2)s with a system that reads the files
itself, shows the model a clean view of them and runs the code it writes.

This notebook runs on a free Colab **T4 GPU**: the data, the first solution in detail, a tool and
a loop as a skeleton, the directions, how to measure, how to submit. Challenge page:
[ml-arena.com/viewchallenge/194](https://ml-arena.com/viewchallenge/194)."""),

("md", r"""---

## 0. Setup

**Runtime → Change runtime type → T4 GPU.** The ML-Arena client is the package **`mlarena-sdk`**
(imported as `mlarena`; `pip install mlarena` is an unrelated package). Colab already has torch,
transformers, pandas and numpy.

The code is the public repository [racousin/ds-harness](https://github.com/racousin/ds-harness):

| file | what it is |
|---|---|
| `scoring.py` | the scorer and the delivery loop (`run_agent`) the leaderboard runs |
| `schema.md` | formats, score, timing, the machine: read it once |
| `localtest.py` | runs an agent file on `dev.json` the way the leaderboard does |
| `dsh.py` | the kit: model loading, answer parsers, calculator, tool-call parser, code runner |
| `stage1_direct.py` | the first solution (section 2) |
| `stage2_tool_loop.py` | a tool and a loop, with the loop body to write (section 3) |"""),

("code", r"""!pip install -q mlarena-sdk accelerate
!git clone -q https://github.com/racousin/ds-harness
import os
os.chdir("ds-harness")
print(sorted(os.listdir(".")))"""),

("md", r"""Your API key is on your ML-Arena **Profile** page (`mlk_user_...`). `download_dataset` writes
`dev.json`: 180 public tasks with their files, answers and types."""),

("code", r"""from getpass import getpass
import mlarena

CHALLENGE_ID = 194
client = mlarena.connect(api_key=getpass("ML-Arena API key (mlk_user_...): "))
print(client.download_dataset(CHALLENGE_ID, "."))"""),

("code", r"""import collections, json, re, tempfile, time
import pandas as pd
import dsh, scoring

dev = json.load(open("dev.json"))
print(len(dev), "tasks")"""),

("md", r"""---

## 1. The data

### What your agent receives

Your `Agent.solve(tasks)` is called with 8 tasks at a time. A task is **only** its `id`, its
`objective` and the **paths** of its files. The platform writes the files of a call to disk just
before the call and deletes them after. `scoring.write_files` does exactly that here:"""),

("code", r"""no_file = next(t for t in dev if not t["files"])
with_files = next(t for t in dev if t["type"] == "table.filter")
root = tempfile.mkdtemp()
payload = scoring.write_files([no_file, with_files], root)
for p in payload:
    print(json.dumps(p, indent=1), "\n")"""),

("code", r"""for path in payload[1]["files"]:
    print(f"--- {os.path.basename(path)}")
    print("".join(open(path).readlines()[:8]))"""),

("md", r"""Read the objective of the second task again, then the files. The headers are short codes that
change from task to task; the objective names a column by its **description** in
`data_dictionary.csv`. The objective also states how to read the file: the separator, the decimal
mark, how missing values are written, values that carry their unit, duplicated rows, a value
that means *not recorded*. Getting all of this right is the system's job, not the model's.

### What only you see

`dev.json` also carries, for each task, its `answer`, its tolerance `tol` and its `type`. The
agent never gets them: your system must recognise the kind of task from the objective."""),

("code", r"""df = pd.DataFrame([{"type": t["type"], "family": t["type"].split(".")[0], "files": len(t["files"])}
                   for t in dev])
df.groupby(["family", "type"]).size().rename("tasks").reset_index()"""),

("code", r"""for ty in ("calc.dates", "table.join", "fit.logistic"):
    t = next(t for t in dev if t["type"] == ty)
    print(f"[{ty}]  answer {t['answer']}  tol {t['tol']:g}  files {list(t['files'])}")
    print(t["objective"], "\n")"""),

("md", r"""### Score and timing

- **One number per task.** The objective states the rounding: *n decimals* → tolerance
  `1.5 × 10⁻ⁿ`; *an integer* → `1e-6`. Inside it the task scores 1, otherwise 0. An exact,
  unrounded value is always inside.
- **Score = 100 × the mean over all tasks.** The private set has 120 tasks: 45 without files,
  50 tables, 25 fits, with other draws and a few table types that `dev.json` does not have.
- **Time.** `Agent()` has 60 s to load its model, then 15 calls of 8 tasks, **40 s per call**.
  A call that raises or takes longer ends the run and the deployment fails."""),

("code", r"""for answer in (41.07, "41.07", 41.069, 41.08, "41,07", None):
    print(repr(answer), "->", scoring.score_answer({"answer": 41.07, "tol": 0.015}, answer))"""),

("md", r"""---

## 2. Stage 1: the model, a structured answer, a parser

### One call

`dsh.load_llm` loads a model of the platform's list; `chat` takes a list of conversations and
generates all the replies in one batch (much cheaper than one call per task)."""),

("code", r"""llm = dsh.load_llm("Qwen/Qwen2.5-1.5B-Instruct")
reply = llm.chat([[{"role": "user", "content": no_file["objective"]}]], max_new_tokens=300)[0]
print(no_file["objective"], "\n\n---\n", reply, "\n\ngold:", no_file["answer"])"""),

("md", r"""The reply is prose. The scorer wants one number. Stage 1 is three decisions: **what the
prompt asks for**, **how the answer is marked**, and **how your code reads it back**. We measure
them on 24 tasks without files (the cell takes a few minutes on a T4)."""),

("code", r"""calc = [t for t in dev if not t["files"]]
sample = calc[:24]

def measure(name, system, parse, max_new_tokens=512):
    convs = [[{"role": "system", "content": system}, {"role": "user", "content": t["objective"]}]
             for t in sample]
    t0 = time.time()
    replies = llm.chat(convs, max_new_tokens=max_new_tokens)
    secs = time.time() - t0
    answers = [parse(r) for r in replies]
    score = sum(scoring.score_answer(t, a if a is not None else 0)[0] for t, a in zip(sample, answers))
    print(f"{name:34s} {100 * score / len(sample):5.1f} / 100   {secs / len(sample):.2f} s per task")
    return replies"""),

("md", r"""**An example last line.** Show the expected last line with an example value and read
the number after `####`."""),

("code", r"""copy = measure("example last line '#### 42.5'",
               "Answer the question. End with a line like: #### 42.5",
               lambda r: dsh.last_number(r.split("####")[-1]))
print(copy[0][:300])"""),

("md", r"""**Three designs.** Reason first, then mark the answer:

1. a last line `ANSWER: <number>`, read by a strict parser — or with a **fallback** to the last
   number written;
2. a JSON object `{"reasoning": ..., "answer": ...}`;
3. two calls: reason freely, then ask for the number only."""),

("code", r"""TAG = ("Solve the problem. Reason step by step, writing each calculation. "
       "Then write a last line of the form ANSWER: <number>, with the number only.")
tag = measure("ANSWER line, strict", TAG, dsh.tagged_number)
_ = measure("ANSWER line, else last number", TAG, lambda r: dsh.tagged_number(r) or dsh.last_number(r))"""),

("code", r"""def from_json(r):
    try:
        return dsh.to_number(json.loads(re.sub(r"^```(json)?|```$", "", r.strip()).strip())["answer"])
    except Exception:
        return dsh.last_number(r)

_ = measure("JSON {reasoning, answer}",
            'Reply with JSON only: {"reasoning": "<short working>", "answer": <number>}', from_json)"""),

("code", r"""convs = [[{"role": "system", "content": "Solve the problem. Reason step by step."},
          {"role": "user", "content": t["objective"]}] for t in sample]
first = llm.chat(convs, max_new_tokens=512)
second = llm.chat([c + [{"role": "assistant", "content": r},
                        {"role": "user", "content": "Write only the final answer as a number, rounded as asked."}]
                   for c, r in zip(convs, first)], max_new_tokens=16)
score = sum(scoring.score_answer(t, dsh.last_number(r) or 0)[0] for t, r in zip(sample, second))
print(f"two calls: {100 * score / len(sample):.1f} / 100")"""),

("md", r"""Measured on the 65 dev tasks without files on the platform's GPU (Qwen2.5-1.5B):
%(s1_table)s
- **the parser is a design decision**: the same replies score %(strict)s with a strict `ANSWER:`
  parser and %(fallback)s with a fallback to the last number;
- **the best-obeyed format is not the most accurate**: JSON is followed most often and scores
  least — writing JSON takes the place of the reasoning;
- **separating the reasoning from the answer pays**: two calls score best, for one more short
  generation.

`stage1_direct.py` uses the `ANSWER:` line with the fallback: simple, one call. Two calls is
your first measured improvement.

### Where it fails, by type"""),

("code", r"""answers = [dsh.tagged_number(r) or dsh.last_number(r) for r in tag]
by_type = collections.defaultdict(list)
for t, a in zip(sample, answers):
    by_type[t["type"]].append(scoring.score_answer(t, a or 0)[0])
for ty, v in sorted(by_type.items()):
    print(f"{ty:20s} {100 * sum(v) / len(v):5.1f}  ({len(v)} tasks)")"""),

("md", r"""Products of large numbers, standard deviations, compound interest, logarithms, days of the
week: the model reasons correctly and computes wrongly. That is what a tool fixes (section 3).

### Never let a task end the run

One exception inside `solve` ends the run and fails the deployment. `stage1_direct.py` wraps the
generation in `try/except` and answers `0` for a task it cannot read: a wrong answer costs one
task, an exception costs all of them.

### With files: the wall

`stage1_direct.py` pastes the first lines of each file into the prompt. Run it on the platform's
test run (16 dev tasks, 2 calls) and on the table types. A T4 is about three times slower than the
platform's RTX 4090, so give each call 120 s here instead of 40."""),

("code", r"""!python localtest.py stage1_direct.py --test-run --timeout 120"""),

("code", r"""!python localtest.py stage1_direct.py --type table.stat --type table.filter --limit 16 --timeout 120"""),

("md", r"""The model never sees the data: a statistic over 300 rows cannot come from 6 lines. Measured on the
full `dev.json` on the platform's GPU, stage 1 scores **%(s1_q15)s** (no files %(s1_q15_calc)s,
tables %(s1_q15_table)s, fits %(s1_q15_fit)s). `stage1_direct.py` is your first submission
(section 6).

---

## 3. Stage 2: a tool and a loop (a skeleton)

A tool is a function your code runs on the model's behalf. The model asks for it in a fixed
format, your code runs it and sends the result back, and the model continues. Six parts:

| part | in `stage2_tool_loop.py` |
|---|---|
| tool spec | the system prompt says what the calculator accepts and how to call it |
| call format | `<call tool="calculator">EXPRESSION</call>`, shown once in a worked exchange (`DEMO`); generation stops at `</call>` |
| call parser | `dsh.parse_tool_call(reply)` → `("calculator", "15.18 - 15.9")` or `None` |
| execution | `observe(call)`: `dsh.calculator`, an error returned as text, never raised |
| observation | a user message `Result: -0.72` (or `Error: ...`) appended to the conversation |
| stop rule | a reply without a call, `MAX_ROUNDS` rounds, or the clock |

The worked exchange matters: with the format only described in the system prompt, the 1.5B model
writes its own (`CALL: ...`) and guesses the result; shown once, it calls the tool (measured on the
65 tasks without files: %(loop_nodemo)s without the exchange, %(loop_demo)s with it).

One round, by hand:"""),

("code", r"""import importlib, stage2_tool_loop as s2
importlib.reload(s2)
t = next(t for t in dev if t["type"] == "calc.functions")
conv = [{"role": "system", "content": s2.SYSTEM}] + s2.DEMO + [{"role": "user", "content": t["objective"]}]
r = llm.chat([conv], max_new_tokens=256, stop=["</call>"])[0]
print(r)
call = dsh.parse_tool_call(r)
print("\ncall:", call, "\nobservation:", s2.observe(call) if call else None)"""),

("code", r"""if call:
    conv += [{"role": "assistant", "content": r}, {"role": "user", "content": s2.observe(call)}]
    print(llm.chat([conv], max_new_tokens=256, stop=["</call>"])[0])
print("\ngold:", t["answer"])"""),

("md", r"""**Your turn.** Write the loop body marked `TODO` in `stage2_tool_loop.py` (about ten lines:
keep the reply, parse a call, append the reply and the observation, keep the conversation
active). Then compare it with stage 1 on the computation types:"""),

("code", r"""CALC = " ".join(f"--type {ty}" for ty in sorted({t["type"] for t in dev if not t["files"]}))
!python localtest.py stage1_direct.py {CALC} --timeout 120 | tail -14
!python localtest.py stage2_tool_loop.py {CALC} --timeout 120 | tail -14"""),

("md", r"""On the platform's GPU, the solved loop takes the tasks without files from %(s1_q15_calc)s to
%(s2_q15_calc)s with the same model. It does nothing for the tables and the fits: the model still
never reads the data.

---

## 4. Directions

No solution here: these are the design decisions your project is about.

- **Routing.** Recognise the kind of task from the objective (files or not, `train.csv`, words like
  *percentile*, *join*, *probability*) and send it to the right prompt and tools. The private set
  has table types `dev.json` does not: a router keyed on dev wordings fails there.
- **Read the files once, yourself.** Parse each CSV with the rules the objective states, link each
  column to its dictionary description, and give the model a compact, clean view: names,
  descriptions, units, types, a few values. The model then reasons about meaning, not about bytes.
- **Data tools or code execution.** Either a few tools with arguments (`stat(column, filter, ...)`),
  easy to call and to check, or the model writes pandas code that `dsh.run_python` runs
  (general, but a 1.5B model's code fails often). Measure both.
- **A fitting tool for `fit.*`.** Least squares is `numpy.linalg.lstsq` on `[1, X]`; an
  unregularised logistic regression is a few Newton steps. A tool the model calls with the target,
  the features and the rows beats code it writes from scratch.
- **Checks, repair, pacing.** Is the number in a plausible range, rounded as asked, an integer when
  one is asked? Send failed code back once with its error. Two batched model rounds fit in 40 s;
  a third may not.
- **The model.** Five are mounted: `Qwen2.5-0.5B/1.5B-Instruct`, `Qwen2.5-Coder-1.5B-Instruct`,
  `DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen3-1.7B`. Change it last, and measure.

Measured on `dev.json` (platform GPU), the ladder these directions lead to:

%(ladder)s

---

## 5. Measure

`localtest.py --out run.json` writes every task's answer, gold, score and error. Group the
failures before you change anything."""),

("code", r"""!python localtest.py stage1_direct.py --test-run --timeout 120 --out run.json > /dev/null
run = json.load(open("run.json"))
d = pd.DataFrame(run["details"])
print(d.groupby("type")["score"].mean().mul(100).round(1))
d[d.score == 0][["id", "type", "answer", "gold", "error"]].head(10)"""),

("md", r"""- **A ladder.** Keep one table: system version, score by family, seconds per call. One change
  per line.
- **By type.** `dev.json` has 7 to 18 tasks per type: a difference of one task is 6–14 points on a
  type. Look at families and at the whole score.
- **Error causes.** Wrong reading of a file? Wrong column? Right code, wrong rounding? A format
  error? Each cause has its own fix.
- **Your own validation set.** `dev.json` is small, and the private set has other draws: write a
  generator for the types you work on (a few lines of Python with its own gold) and measure on
  hundreds of tasks.
- **Time.** Measure seconds per call on the platform (the run's log) and keep a margin: the T4 is
  slower, the RTX 4090 faster, a long objective slower.

---

## 6. Submit

Upload your agent file **as `agent.py`**, with `dsh.py`, and choose the **PyTorch** runtime. A
deployment runs the test run (16 dev tasks), then the scored run (120 private tasks)."""),

("code", r"""import shutil
shutil.copy("stage1_direct.py", "agent.py")
result = client.submit(challenge_id=CHALLENGE_ID, files=["agent.py", "dsh.py"],
                       submission_name="stage 1",
                       runtime={"language": "python", "framework": "torch"},
                       wait=True, timeout_sec=1800)
print(result["status"]["status"], result["status"]["last_status_message"])"""),

("md", r"""The leaderboard shows the score and its three family means. Deployments are limited per day:
test locally first, submit when a change is measured."""),
]


def fmt_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    subs = dict(M["scalars"])
    subs["s1_table"] = "\n\n" + fmt_table(M["stage1_designs"], ["design", "score (/100)", "format obeyed"]) + "\n"
    subs["loop_nodemo"], subs["loop_demo"] = M["loop"]["no demo"], M["loop"]["one demo"]
    d = {row[0]: row[1] for row in M["stage1_designs"]}
    subs["strict"], subs["fallback"] = d["ANSWER line, strict parser"], d["ANSWER line, else the last number"]
    subs["ladder"] = fmt_table(M["ladder"], ["system", "model", "score", "no files", "tables", "fits"])
    cells = []
    for kind, src in CELLS:
        src = src % subs if "%(" in src else src
        lines = src.split("\n")
        body = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                          "source": body})
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
