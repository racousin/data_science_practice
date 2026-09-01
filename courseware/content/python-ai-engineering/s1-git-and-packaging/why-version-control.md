# Why Version Control

Every project you will hand in this year lives in a Git repository. Not because
the syllabus says so — because the alternative does not survive contact with a
second person, a second machine, or a bad afternoon.

<!-- notes: Keep this short — 10 minutes. The goal is motivation, not history.
Ask the room who has ever emailed themselves a zip of their code. -->

---

## The problem, concretely

Without version control, a project accumulates this:

```text
model.py
model_v2.py
model_v2_FINAL.py
model_v2_FINAL_marie.py
model_v2_FINAL_marie_works.py
```

Three questions you cannot answer from that directory:

- What changed between `FINAL` and `FINAL_marie`?
- Which one produced the result in the report?
- If Marie and you both edited, how do you combine the work?

---

## What Git actually is

A Git repository is a **directed graph of snapshots**. Each snapshot (a
*commit*) records the full state of your project plus a pointer to its parent.

That single idea gives you all of the following for free:

- **History** — every state the project has ever been in, recoverable exactly.
- **Attribution** — who changed what line, and when.
- **Branching** — several lines of work in the same directory, isolated.
- **Distribution** — every clone is a complete copy; there is no single point of failure.

---

## The three places a file can be

This is the mental model to hold for the rest of the session. Everything else is
commands that move files between these three places.

| Place | What it holds | Command that fills it |
|---|---|---|
| **Working directory** | The files you edit | your editor |
| **Staging area (index)** | Changes selected for the next snapshot | `git add` |
| **Repository (.git)** | Committed snapshots, permanently | `git commit` |

![The three areas of a Git repository](assets/git/commit-main.png)

*Source: [Visual Git Guide](https://marklodato.github.io/visual-git-guide/index-en.html)*

---

## Why staging exists

Beginners find the staging area redundant. It is not. It lets you commit *part*
of your work.

You fixed a bug and, on the way, renamed a variable in an unrelated file. Those
are two different commits. Staging is what makes that possible without undoing
anything.

> A good commit is one change, explainable in one sentence. Staging is the tool
> that makes your commits look like your intentions rather than like your
> afternoon.

---

## What we are building today

By the end of Session 1 you will have a repository that contains:

- a Python package installable with `pip install -e .`
- a test suite that runs with `pytest`
- a pinned, reproducible environment
- a history of small, readable commits on a feature branch
- a pull request, reviewed and merged

That repository is the starting point for Sessions 2, 3 and 4 — and the second
half of your project grade is the quality of exactly this.
