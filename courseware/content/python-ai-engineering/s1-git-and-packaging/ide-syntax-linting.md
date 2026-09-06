# Code Quality: Linting, Formatting & Types

Three tools that catch a class of mistake each, automatically, before a human
has to. Configure once per project; never argue about any of it again.

<!-- notes: 25 minutes. The framing that lands is the cost curve — the same bug
costs seconds in the editor and a day in week six. Run ruff live on a file with
an unused import and a mutable default. -->

---

## The point

![Five gates](assets/s1-git-and-packaging/ide-syntax-linting/quality-gates.png)

The same defect costs a different amount depending on which gate catches it.
Everything in this lesson is about moving defects left.

---

## Formatter, linter, type checker

Three different jobs, routinely confused:

| Tool | Answers | Example |
|---|---|---|
| **formatter** | "does it look like the rest of the code?" | line length, quotes, blank lines |
| **linter** | "is this a mistake?" | unused import, mutable default, undefined name |
| **type checker** | "do these pieces fit together?" | passing `str` where `int` was declared |

Only the second and third find bugs. The first exists so that the second and
third are readable, and so that formatting never appears in a diff.

---

## Ruff — formatter and linter in one

```bash
uv add --dev ruff
```

```bash
uv run ruff format src/ tests/      # rewrite to the standard style
uv run ruff check src/ tests/       # report problems
uv run ruff check --fix src/        # report and fix the mechanical ones
```

It is fast enough — milliseconds on a project this size — to run on every save
and in every commit hook, which is the only reason anyone actually runs it.

---

## Configuring it, in `pyproject.toml`

Configuration lives with the project, so every contributor and the CI runner get
the same answer.

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

| Code | What it catches |
|---|---|
| `E` | pycodestyle — spacing, indentation |
| `F` | pyflakes — **unused imports, undefined names** |
| `I` | import order |
| `UP` | syntax that a newer Python does better |
| `B` | bugbear — **likely bugs** |
| `SIM` | needlessly complicated constructs |

`F` and `B` are the two that repay the setup. The rest is tidiness.

---

## Two lint findings that are real bugs

```python
def add_reading(readings=[]):     # B006: mutable default argument
    readings.append(1)
    return readings
```

The list is created **once**, when the function is defined. Call it twice and
the second call sees the first call's data. This is a genuine, hard-to-find bug
and the linter finds it in a millisecond.

```python
from mypackage.utils import normalise   # F401: imported but unused
```

Harmless here. But an unused import is usually the fossil of a deleted approach,
and sometimes it is the import you meant to use two lines below.

---

## Style is not the point

Arguments about quotes and line length are a waste of review attention. Pick a
formatter, run it on save, and the argument is over.

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff"
}
```

The measurable win: a formatted-on-save codebase has **no formatting in its
diffs**, so every line a reviewer reads is a line that changed on purpose.

---

## Type hints

```python
def accuracy(pred: list[int], true: list[int]) -> float:
    ...
```

They are not enforced at runtime. They buy you three things:

1. **The editor catches a mistake before you run anything.**
2. **The signature documents itself** — no docstring needed to say what goes in.
3. **A type checker can prove things** about code you have not executed.

For this course: hints on **public functions**, nothing else. Do not annotate
every local variable; it is noise.

---

## mypy, if you want the check

```bash
uv add --dev mypy
uv run mypy src/
```

```text
src/textstats/core.py:14: error: Argument 1 to "len" has incompatible type
"int"; expected "Sized"  [arg-type]
```

Start permissive and tighten later. A project that turns on `--strict` on day
one produces 400 errors and gets its type checking deleted the same afternoon.

---

## Pre-commit — the gate that runs itself

Running the tools is a habit. Habits fail. A git hook does not.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

```bash
uv tool install pre-commit
pre-commit install
```

Now `git commit` runs them first and refuses a commit that fails. Fail fast,
applied to your own workflow.

To run it over the whole repository once, the first time:

```bash
pre-commit run --all-files
```

<!-- notes: Warn them: the first run rewrites a lot of files. Do it in its own
commit, titled "Apply ruff format", so it never mixes with real work. -->

---

## The one hook everyone should have

Committing a notebook commits its outputs — including, sometimes, a printed API
key and a 40 MB embedded image.

```yaml
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

It strips outputs from `.ipynb` files on the way into a commit. Your notebooks
keep their outputs on your disk; the repository stores the code.

---

## Editors

| Editor | Notes |
|---|---|
| **VS Code** | free; Python + Jupyter + Ruff extensions; agent extensions |
| **PyCharm** | strongest refactoring; Community edition is free |
| **Neovim** | steep, fast, entirely yours |
| **Jupyter / Colab** | exploration only — not where a project lives |

Any of them is fine. Running `python script.py` with no editor support, and
finding out about your typo at runtime, is not.

Three things to configure, once, whichever you pick:

1. **Interpreter** — the project's `.venv`, not the system Python. Nearly every
   "it imports in the terminal but not in the editor" is this.
2. **Format on save.**
3. **Linter inline** — problems underlined as you type.

---

## What goes in the project

By the end of this session your repository contains:

```text
pyproject.toml           [tool.ruff] section
.pre-commit-config.yaml  ruff + ruff-format + nbstripout
```

and the CI workflow in the next lesson runs the same `ruff check` that your
editor runs — same version, same configuration, same answer.

That last property is the one that matters. A linter that disagrees with CI is
worse than no linter.
