# Lab 1 — Ship a Package

Work in pairs. One repository per pair. Both of you must appear in the commit
history — this is graded on the history, not only on the final state.

**Time:** 60 minutes. **Deliverable:** a GitHub repository URL.

---

## Part A — Repository (10 min)

1. One of you creates a **private** GitHub repository named
   `paie-lab1-<lastname1>-<lastname2>`.
2. Add your partner and `racousin` as collaborators.
3. Both clone it.
4. Commit a `.gitignore` covering `.venv/`, `__pycache__/`, `.env`, `data/`.

**Check:** `git log --oneline` shows at least one commit on `main`.

---

## Part B — Package skeleton (15 min)

Build this, on a branch called `feature/skeleton`:

```text
pyproject.toml
src/textstats/__init__.py
src/textstats/core.py
tests/test_core.py
```

`pyproject.toml` must declare the project name, a Python version, and `pytest`
as a dev extra.

Implement in `core.py`:

```python
def word_count(text: str) -> int:
    """Number of whitespace-separated tokens in `text`."""


def char_frequencies(text: str) -> dict[str, int]:
    """Count of each character, ignoring whitespace and case."""


def longest_word(text: str) -> str:
    """The longest token. Raises ValueError on empty input."""
```

**Check:** `uv pip install -e ".[dev]"` succeeds, and
`python -c "import textstats"` works from your home directory.

---

## Part C — Tests (15 min)

Write at least **eight** tests. They must include:

- one `@pytest.mark.parametrize` covering three or more cases
- one `pytest.raises` test for `longest_word("")`
- one fixture used by two different tests

Follow the fail-fast principle: `longest_word("")` must raise, not return `""`.

**Check:** `uv run pytest -v` is green.

---

## Part D — The pull request (15 min)

1. Push `feature/skeleton`.
2. Open a pull request against `main`. Write a description that says what the
   package does and how to run the tests.
3. Your partner reviews it and leaves **at least one substantive comment** —
   not "LGTM".
4. Address the comment with a new commit on the same branch.
5. Merge, then delete the branch.

---

## Part E — Deliberate conflict (5 min)

Both of you, simultaneously:

1. `git switch main && git pull`
2. Each create a branch and edit **the same line** of `README.md`.
3. Both push. First one merges. Second one resolves the conflict.

Resolve it so the final `README.md` contains *both* contributions, sensibly
merged. Commit the resolution.

**Check:** `git log --graph --oneline` shows the merge commit.

---

## Grading

| Criterion | Weight |
|---|---|
| `uv sync && uv run pytest` green on a fresh clone | 30% |
| Commit history: small, readable, both authors present | 25% |
| Tests: edge cases covered, failure paths asserted | 25% |
| Pull request: description + real review exchange | 20% |

---

## Common failures

- **`.venv/` committed** — the repo is 200 MB. Check `git status` before every add.
- **One author** — pair-programming on one laptop still needs both accounts in the history. Swap the driver, or use `Co-Authored-By:` trailers.
- **`ModuleNotFoundError` on a fresh clone** — you forgot to commit `pyproject.toml`, or the package is outside `src/`.
- **Tests that assert nothing** — `assert word_count("a b") ` with no comparison passes on any truthy value.

---

## If you finish early

Add a `[project.scripts]` entry point so `textstats <file>` prints the word
count from the command line. Push it as a second pull request.
