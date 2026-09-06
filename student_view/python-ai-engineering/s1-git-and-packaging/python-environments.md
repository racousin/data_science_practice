# Python Environments

An environment is the answer to "it works on my machine". Getting this right
costs ten minutes now and saves a days in the future.

<!-- notes: 30 minutes. uv is the recommendation; venv is the fallback everyone
has. Show both, do not pretend venv is obsolete. -->

---

## The problem

Two projects on your laptop. One needs `numpy 1.26`, the other needs
`numpy 2.1`. Python installs one `numpy` per interpreter.

Install globally and the projects fight. The fight is silent: your code imports
successfully, then behaves differently from your teammate's.

![Environment isolation](/api/academic_courses/assets/lessons/31/env-isolation.png)

---

## The solution

One isolated interpreter and package set **per project**, living inside the
project directory, never committed.

```text
my_project/
├── .venv/            <- the environment (gitignored)
├── src/
├── tests/
├── pyproject.toml    <- what the project needs (you write this)
└── uv.lock           <- what you actually ran (generated)
```

The environment is disposable — delete `.venv` and rebuild it in seconds. The
two files next to it are what you commit.

---

## venv — always available

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Your prompt changes to show the active environment. Verify you are where you
think you are:

```bash
which python      # should print .../my_project/.venv/bin/python
```

Deactivate with `deactivate`.

This works everywhere and is worth knowing. But it manages only the
environment — the declaration and the lock are still your problem.

---

## uv — what we will use

`uv` is a Rust reimplementation of the packaging toolchain. Same concepts, one
to two orders of magnitude faster, and it manages Python versions too.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv init my_project           # pyproject.toml + .python-version + src/
cd my_project
uv add numpy                 # declare, resolve, lock, install — one step
uv run python train.py       # run inside the environment, without activating
```

You never create or activate `.venv` yourself. `uv add` and `uv run` build it
on demand and keep it matching `uv.lock`.

`uv run` is the habit worth forming: no activation state to get wrong, and no
way to accidentally run against the system interpreter.

<!-- notes: Watch for people typing `source .venv/bin/activate` out of habit.
It is not wrong, but it is not needed, and it is where the drift starts. -->

---

## Installing packages

```bash
uv add "numpy>=1.26"
uv add -r requirements.txt        # imports an old project into pyproject
uv add --dev pytest               # → [dependency-groups] dev
uv add --optional viz matplotlib  # → [project.optional-dependencies]
uv add --editable ./libs/foo
uv add "foo @ git+https://github.com/o/foo"
uv remove numpy
uv tree                           # what is installed, and what pulled it in
```

Each command edits `pyproject.toml`, updates `uv.lock`, and syncs `.venv` —
in that order. The three never drift apart.

---

## One trap: `uv pip`

`uv` also ships a pip-compatible interface:

```bash
uv pip install numpy      # installs into .venv, records nothing
```

It installs into the environment and writes to *neither* `pyproject.toml` nor
`uv.lock`. The next `uv run` or `uv sync` resyncs `.venv` from the lock and
silently removes what you installed.

**In this course: use `uv add`, never `uv pip`.**

`uv pip` exists for environments with no `pyproject.toml` — a scratch venv, a
CI job that still speaks `requirements.txt`. Outside a project it is a fast pip.
Inside one it is a footgun.

---

## Version specifiers

| Specifier | Means |
|---|---|
| `numpy` | any version — avoid |
| `numpy==2.1.0` | exactly this |
| `numpy>=1.26` | this or newer |
| `numpy>=1.26,<2` | a range — the usual choice |
| `numpy~=1.26.0` | `>=1.26.0, <1.27.0` — patch updates only |

Ranges belong in `pyproject.toml`. Exact versions belong in the lock, and you
do not type them by hand.

---

## Declaring dependencies

Two files, two different jobs. Confusing them is the most common mistake.

| File | Contains | Answers | Written by |
|---|---|---|---|
| `pyproject.toml` | ranges you *support* | "what does this project need?" | you |
| `uv.lock` / `requirements.txt` | exact pinned versions | "what exactly did I run?" | the tool |

Never hand-edit the lock.

---

## Declaration and lock, side by side

![Declaration versus lock](/api/academic_courses/assets/lessons/31/declaration-vs-lock.png)

Commit both. The first is intent; the second is reproducibility. The CI lesson
shows what happens when you commit only the first.

---

## pyproject.toml

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.5",
]
```

`uv add` writes these lines for you. Editing them by hand is fine too — that is
what `uv lock` is for.

---

## Locking

```bash
uv lock                 # re-resolve after editing pyproject.toml by hand
uv sync                 # make .venv match uv.lock exactly
uv lock --upgrade       # deliberately move to newer versions
```

`uv add` already does all three. You call them directly when you edited
`pyproject.toml` yourself, or after pulling a teammate's changes.

With plain pip, the closest equivalent is:

```bash
pip freeze > requirements.txt
```

`pip freeze` is blunter — it records everything installed, including things you
did not ask for, and it cannot tell a direct dependency from a transitive one.
But it works everywhere, and is the fallback when a grader cannot use `uv`.

---

## Reproducing an environment

Someone clones your repo. This is all they should have to do:

```bash
git clone <url> && cd my_project
uv sync
uv run pytest
```

If that sequence does not work on a clean machine, your project is not
reproducible, whatever the README claims.

<!-- notes: Have them actually test this — clone their own repo into /tmp and
run the three commands. A surprising number will fail on an uncommitted file. -->

---

## Python versions

`uv` will fetch an interpreter for you — no system Python involved:

```bash
uv python install 3.12
uv python pin 3.12       # writes .python-version, commit it
```

Also set `requires-python` in `pyproject.toml`, so a mismatch fails at install
time rather than at line 300 of your training script.
