# Session Plan — Shell, Notebooks & Colab

> **Draft.** This is the authoring brief for the session, not the session. It is
> unpublished: it states what will be built and why, so the next pass writes
> lessons against a fixed target instead of re-deciding the scope.

Session 2 was Agentic Coding; those six lessons now sit at the end of Session 1.
This slot is free, and the material it should carry is the one thing the course
assumes everywhere and teaches nowhere: **the shell, the notebook, and Colab.**

---

## Why this session exists

Session 1 opens with `git init`, `uv venv`, `uv run pytest`. Sessions 3 and 4
are built out of notebook-shaped work — load a dataframe, plot it, train a
model, read a traceback — and the MS2A competitions hand students a Colab link
as the starting point. Three tools, used from the first hour, taught in none.

The failure mode is not that students cannot learn them alone. It is that the
ones who cannot spend Session 1 debugging their terminal instead of learning
git, and the gap is invisible from the front of the room: a student who cannot
read `command not found` looks identical to one who disagrees with the design.

The three tools are also a progression, not a list:

| | Where the code runs | What persists | Taught for |
|---|---|---|---|
| **Shell** | your machine | the filesystem | running anything the course asks you to run |
| **Notebook** | your machine, one kernel | the kernel's memory | exploration, plots, iteration |
| **Colab** | Google's machine, with a GPU | nothing, unless you mount it | Sessions 3-4 and every MS2A competition |

The last column is the argument. Each is introduced because something later
requires it, not for completeness.

---

## Block 1 — The Shell

**Target:** a student can navigate to a repository, inspect it, run a command,
and read the error when it fails.

- Paths — absolute vs relative, `.`, `..`, `~`, and why `cd` is the source of
  half of all "the file is not there" reports.
- `ls`, `cd`, `pwd`, `cat`, `less`, `mkdir`, `rm`, `mv`, `cp`. Ten commands, no
  more.
- Tab completion and history. Taught as the two habits, not as trivia.
- `$PATH` and `which` — the mechanism behind `command not found` and behind
  "it works in the terminal but not in VS Code".
- Exit codes, `stdout` vs `stderr`, and the pipe. Enough to understand
  `uv run pytest -q | tail -20`.
- Environment variables — `export MLARENA_API_KEY=…` is a lab step in this
  course; students should know what it does and why it dies with the shell.

**Deliberately excluded:** `sed`/`awk`/`grep` beyond a literal search, shell
scripting, job control, `vim`. They are a reference lesson at most.

**Windows.** Decide once and state it on the page: WSL, or Git Bash. The rest of
the session assumes it. Undecided — see open questions.

---

## Block 2 — Notebooks

**Target:** a student can run a notebook, explain why it broke, and know when to
stop using one.

- The kernel model — cells are not a program; a notebook is a REPL with a
  scrollback. This is the whole lesson and everything else follows from it.
- Out-of-order execution. The demo is the lesson: run cells 1, 3, 2 and produce
  a result that no fresh run reproduces. Then *Restart & Run All* as the only
  honest check.
- `!` and `%` — shell escape and magics, which is where Block 1 pays off.
- Where the file lives and what `git diff` shows for it: a JSON blob with
  outputs, which is why the packaging lesson said "notebooks are not the
  deliverable".
- The handoff, and the through-line back to Session 1: explore in the notebook,
  ship the tested module. Not one or the other.

**Deliberately excluded:** widgets, `nbconvert`, Jupyter server configuration.

---

## Block 3 — Colab

**Target:** a student can open the competition's starter notebook, get a GPU,
and get their data and results in and out.

- What it is: a hosted notebook on someone else's machine, and every consequence
  of that word *hosted*.
- Runtime types and the GPU toggle — the reason Sessions 3-4 use it at all.
- The ephemeral filesystem. Uploads, `!wget`, Drive mount; the runtime dies and
  takes everything with it. Save the weights or lose them.
- `!pip install` in a session, and why the first cell of every course notebook
  is an install cell.
- Submitting from Colab to ML-Arena — the API key, and the fact that pasting one
  into a shared notebook publishes it.

**Deliberately excluded:** Colab Pro, TPUs, local runtime connection.

---

## Lab

One lab, three parts, one artifact — the same shape as Labs 1 and 2:

1. **Shell** — from a fresh terminal: clone the Session 1 repository, create the
   environment, run the tests, and capture the output of a failing run into a
   file. Graded on the file, which cannot be produced without the pipe.
2. **Notebook** — open the provided notebook, which is broken by out-of-order
   execution. Diagnose it, fix it, and prove it with a clean *Restart & Run All*.
3. **Colab** — open the same notebook in Colab, enable the GPU, print the device,
   and submit a result to the session competition from the notebook itself.

The lab ends where every lab in this course ends: a submission on a leaderboard.

---

## Competition

Session 2 currently carries **#180 (PAIE S2 — Flesch reading-ease)**, which
belongs to Lab 2 and moves to Session 1 with it. This session needs its own.

The requirement it has to meet is the one Session 1's does not: it must be
submittable **from a Colab notebook in the last twenty minutes of class**. That
argues for a scorer with a trivial floor and a one-cell reference solution —
closer to `s4-mnist-warmup` than to `s3-adult-income` — so the thing being
tested is the submission path, not the model.

Per the house style in `competitions/s3-adult-income/overview.md:50-63`, its
page must state the measured trivial floor, the reference score, and the
direction. Numbers, not "beat the baseline".

---

## Open questions

1. **Where does this session go?** It is written as Session 2, but the argument
   in *Why this session exists* is an argument for it being **Session 0** —
   everything Session 1 does assumes it. Renumbering costs four module slugs,
   which are immutable, so decide before the next publish.
2. **Windows policy.** WSL or Git Bash. Affects every command on every page.
3. **Session 1 is now 12 lessons / ~5 hours** against a 3-hour slot. It holds
   two sessions' material and has to be split; this plan does not depend on how.
4. **Does the notebook block duplicate Session 3?** Session 3 already opens in a
   notebook. This block should teach the kernel model and hand Session 3 a
   student who has it — not teach pandas twice.
