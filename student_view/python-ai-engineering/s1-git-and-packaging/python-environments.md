# Python Environments

An environment is the answer to "it works on my machine". Getting this right
costs ten minutes now and saves a day in Session 4.

<!-- notes: 30 minutes. uv is the recommendation; venv is the fallback everyone
has. Show both, do not pretend venv is obsolete. -->

---

## The problem

Two projects on your laptop. One needs `numpy 1.26`, the other needs
`numpy 2.1`. Python installs one `numpy` per interpreter.

Install globally and the projects fight. The fight is silent: your code imports
successfully and then behaves differently from your teammate's.

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
└── pyproject.toml    <- the declaration of what belongs in it
```

The environment is disposable. The *declaration* is what you commit.

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

---

## uv — what we will use

`uv` is a Rust reimplementation of the packaging toolchain. Same concepts,
one to two orders of magnitude faster, and it manages Python versions too.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv                      # create .venv
uv pip install numpy         # install into it
uv run python train.py       # run in it, without activating
```

`uv run` is the habit worth forming: it guarantees the command runs in the
project environment, with no activation state to get wrong.

---

## Installing packages

```bash
uv pip install "numpy>=1.26"
uv pip install -r requirements.txt
uv pip install -e .            # your own project, editable
uv pip list
uv pip uninstall numpy
```

The `pip` interface is deliberately identical. Everything you already know
transfers.

---

## Version specifiers

| Specifier | Means |
|---|---|
| `numpy` | any version — avoid |
| `numpy==2.1.0` | exactly this |
| `numpy>=1.26` | this or newer |
| `numpy>=1.26,<2` | a range — the usual choice for libraries |
| `numpy~=1.26.0` | `>=1.26.0, <1.27.0` — patch updates only |

---

## Declaring dependencies

Two files, two different jobs. Confusing them is the most common mistake.

| File | Contains | Answers |
|---|---|---|
| `pyproject.toml` | ranges you *support* | "what does this project need?" |
| `uv.lock` / `requirements.txt` | exact pinned versions | "what exactly did I run?" |


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

---

## Locking

```bash
uv lock                 # resolve and write uv.lock
uv sync                 # make .venv match uv.lock exactly
```

With plain pip, the equivalent is:

```bash
pip freeze > requirements.txt
```

`pip freeze` is blunter — it records everything installed, including things you
did not ask for — but it works everywhere and is the fallback when a grader
cannot use `uv`.

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

`uv` will fetch an interpreter for you:

```bash
uv python install 3.12
uv venv --python 3.12
```

Pin the version in `pyproject.toml` (`requires-python`) so a mismatch fails at
install time rather than at line 300 of your training script.

---

## What never goes in Git

```text
.venv/
__pycache__/
*.egg-info/
.env
```

The environment is regenerated from the lockfile. Committing it bloats the repo
and breaks on any other operating system.

---

## Checklist

- [ ] `.venv/` exists and is gitignored
- [ ] `which python` points inside the project
- [ ] `pyproject.toml` declares dependencies with ranges
- [ ] a lockfile is committed
- [ ] `uv sync` on a fresh clone reproduces the environment
