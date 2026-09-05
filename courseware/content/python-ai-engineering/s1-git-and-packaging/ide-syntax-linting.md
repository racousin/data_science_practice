# IDE, Syntax & Linting

Reference only — not covered in class. Setting up an editor and automating code
style so nobody argues about it in review.

---

## Editors

| Editor | Notes |
|---|---|
| **VS Code** | free, the Python + Pylance extensions, Claude Code extension |
| **PyCharm** | strongest refactoring; Community edition is free |
| **Neovim** | steep, fast, configurable |
| **Jupyter / Colab** | exploration only — not where a project lives |

Any of them is fine. Not having one, and running `python script.py` blind, is
not.

---

## What to configure

Three things, once:

1. **Interpreter** — point it at the project's `.venv`, not the system Python.
   Nearly every "the import works in the terminal but not in the editor" problem
   is this.
2. **Format on save** — so formatting never appears in a diff.
3. **Linter inline** — errors underlined as you type, not at runtime.

---

## Ruff

One tool for both linting and formatting, fast enough to run on save.

```bash
uv pip install ruff
```

```bash
uv run ruff check src/ tests/          # lint
uv run ruff check --fix src/           # lint and autofix
uv run ruff format src/ tests/         # format
```

---

## Configuration

In `pyproject.toml`, so it is shared and versioned:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

| Code | Rules |
|---|---|
| `E` | pycodestyle errors |
| `F` | pyflakes — unused imports, undefined names |
| `I` | import sorting |
| `UP` | modernise old syntax |
| `B` | bugbear — likely bugs, not just style |

`F` and `B` are the ones that catch actual defects.

---

## Style is not the point

Formatting arguments are a waste of review attention. Pick a formatter, run it
automatically, and never discuss it again — that is the entire value.

The **linter** is different: `F401` (unused import) and `B006` (mutable default
argument) are real bugs, and worth reading.

---

## Type hints

```python
def accuracy(pred: list[int], true: list[int]) -> float:
    ...
```

They document the interface, and they let the editor catch a mistake before you
run anything.

Static checking, if you want it:

```bash
uv run mypy src/
```

For this course, hints on public functions are enough. Do not annotate every
local variable.

---

## Pre-commit

Run the checks before a commit exists, rather than discovering them in CI:

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

Now `git commit` runs the hooks and refuses a commit that fails. Fail fast,
applied to your own workflow.
