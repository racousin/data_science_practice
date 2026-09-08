# Lab 1 — Ship a Package to GitHub

Build a Python package, prove it with tests, make GitHub run those tests for
you, and use the result from a Colab notebook you have never opened before.

**Work in groups of 2 or 3.** One repository for the group; each of you owns one
or more of its functions end to end, and the rest of the group reviews and
merges your work.

**Deliverable:** a GitHub repository URL, with every member of the group in its
commit history.

<!-- notes: Form the groups before you start — 3 is the shape the lab is written
for, 2 works, 4 leaves someone with nothing to own. Walk the room during Part B:
the src/ layout and the editable install are where people stall, and the split
only holds if everyone is on their own branch. -->

---

## What you are building

A package called `textstats`: three functions over text, tested, installable,
and importable from anywhere.

```text
textstats/                     ← the repository
├── .github/workflows/tests.yml
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── src/textstats/
│   ├── __init__.py
│   └── core.py
└── tests/
    └── test_core.py
```

Eight files. By the end they will all be on GitHub, and the badge in the README
will be green.

---

## How the group splits it

Three functions, one repository, one owner each:

| In a group of 3 | In a group of 2 |
|---|---|
| `word_count` | `word_count` **and** `char_frequencies` |
| `char_frequencies` | `longest_word` |
| `longest_word` | |

Your function is yours end to end: the implementation, its tests, its export in
`__init__.py`, its branch, its pull request. **Nobody pushes to `main` and
nobody merges their own pull request** — someone else in the group reads it and
clicks the button.

The shared parts — the repository, the skeleton, the workflow, the tag, the
notebook — are done **once**, by whoever says they are doing them.

---

## Part A — Create the repository
**One person** does this; everyone else clones what it produces.

1. On GitHub, **New repository** → name it `textstats` → **Public** → tick *Add
   a README*. Public, because CI minutes are free on public repositories and
   branch protection is not restricted. There is nothing secret in this.
2. *Settings → Collaborators → Add people* — add the rest of the group. They
   accept from their email or from `github.com/notifications`.

---

## Part A — Everyone clones it

```bash
git clone git@github.com:<owner>/textstats.git
cd textstats
```

One of you commits a `.gitignore`, before anything exists to be ignored:

```text
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.env
.ipynb_checkpoints/
.DS_Store
```

```bash
git add .gitignore && git commit -m "Ignore environments and build artefacts"
git push
```

**Check:** `git log --oneline` shows two commits, `git status` is clean, and
every member of the group has the clone on their own machine.

---

## Part B — Write the package
The skeleton is **one branch, merged before anyone starts a function**. If all
three of you run `uv init` on your own branch, all three of you write a
different `pyproject.toml` and every merge after the first one conflicts.

```bash
git switch -c feature/skeleton
uv init --lib --name textstats .    # or create the files by hand
uv add --dev pytest ruff
```

`--dev` puts them in `[dependency-groups]`, which `uv sync` installs. Do **not**
use `[project.optional-dependencies]` — `uv sync` skips extras, and a fresh
clone is how everyone else in the group runs your code.

---

## Part B — The files it needs

`pyproject.toml` must end up with at least:

```toml
[project]
name = "textstats"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
```

Then write `src/textstats/core.py` — the three signatures and docstrings, and
nothing else, on the skeleton branch:

```python
def word_count(text: str) -> int:
    """Number of whitespace-separated tokens in `text`."""

def char_frequencies(text: str) -> dict[str, int]:
    """Count of each character, ignoring whitespace and case."""

def longest_word(text: str) -> str:
    """The longest token. Raises ValueError on empty input."""
```

---

## Part B — Then everyone branches

Merge the skeleton, then each of you starts from it:

```bash
git switch main && git pull
git switch -c feature/<your-function>
```

Filling in your own stub touches lines nobody else is touching. That is the
point of splitting by function rather than by file.

---

## Part B — The specification

The docstrings do not pin these down, and two defensible readings disagree. This
is the specification — the group's contract, so that three people implementing
separately implement the same thing. The tests in Part C are written against it.

- **Tokens are whitespace-separated and keep their punctuation.**
  `"the end."` is two tokens and the second is `end.` — four characters.
- **`char_frequencies` folds case and drops whitespace.** Punctuation and digits
  *are* characters and do count: `"Aa b"` → `{"a": 2, "b": 1}`.
- **`longest_word` breaks ties to the left**: `longest_word("aaaa bbbb")` is
  `"aaaa"`.
- **`longest_word("")` raises `ValueError`** — it does not return `""`.

---

## Part B — Exports, and the check

Re-export the three from `src/textstats/__init__.py` so `from textstats import
word_count` works:

```python
from textstats.core import char_frequencies, longest_word, word_count

__all__ = ["word_count", "char_frequencies", "longest_word"]
```

Add **your** name to those two lines on your own branch. It is the one file in
the repository that all of you edit, so the second and third pull requests will
need `git switch main && git pull` and `git merge main` before they merge
cleanly. Expect it; it is not a mistake.

**Check**, from your **home directory**, not from the project:

```bash
uv run --project ~/textstats python -c "import textstats; print(textstats.__file__)"
```

It must print a path under your `src/`. A bare `python -c "import textstats"`
from home will not work — that is a different interpreter, which is the whole
point of the environment.

---

## Part C — Write the tests
Write at least **eight** tests in `tests/test_core.py`, three or more per
function, each written by the person who owns that function. Between you they
must include:

- one `@pytest.mark.parametrize` with three or more cases
- one `pytest.raises(ValueError)` for `longest_word("")`
- one fixture used by two different tests — so it is used by two different
  *people*, which means agreeing on it rather than writing one each
- one test for each specification rule: punctuation kept inside a token, digits
  counted as characters, a tie taken from the left

---

## Part C — What a test looks like

```python
import pytest
from textstats import longest_word, word_count


def test_longest_word_rejects_empty():
    with pytest.raises(ValueError):
        longest_word("")


@pytest.mark.parametrize("text,expected", [
    ("", 0), ("one", 1), ("the end.", 2),
])
def test_word_count(text, expected):
    assert word_count(text) == expected
```

**Check:**

```bash
uv run pytest -v
```

Eight or more passing tests once every branch has merged. If a test passes
before you wrote the implementation, it is not testing anything.

---

## Part D — Make GitHub run the tests
This goes on the **skeleton** branch, before the function branches — then every
pull request the group opens is checked before anybody reviews it.

Create `.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync
      - run: uv run pytest -q
```

---

## Part E — Push, open the pull request, get it reviewed

Commit `uv.lock` — without it, `uv sync` on the runner resolves different
versions from yours, and CI stops being evidence about *your* code.

```bash
git add -A && git commit -m "Add word_count and its tests"
git push -u origin feature/word-count
```

Open the pull request and say in one sentence what it does and how to check it.

---

## Part E — Someone else merges it

Another member of the group opens the *Files changed* tab, reads the diff, and
leaves at least one comment that names a line and a consequence. "LGTM" is not
a review. Then, and only when the check run is **green**, they merge it and
delete the branch.

You do not merge your own pull request. Nobody pushes to `main`.

Skeleton and workflow first, then one function at a time.

---

## Part E — Install it in Colab
Each of you opens a **new** Colab notebook — not one you have used before. Two
cells:

```python
!pip install -q git+https://github.com/<owner>/textstats.git@v0.1.0
```

```python
import textstats
print(textstats.word_count("the quick brown fox"))       # 4
print(textstats.longest_word("the quick brown fox"))     # quick
print(textstats.char_frequencies("Aa b"))                # {'a': 2, 'b': 1}
```

This is the moment the packaging was for. Nothing was copied and nothing was
pasted: Colab installed the exact code the group pushed — three people's
functions, called from one import.

Then one of you does *File → Save a copy in GitHub*, into the `textstats`
repository as `notebooks/demo.ipynb`. Commit message: `Add Colab demo`.

**Check:** the notebook is in your repository, and its install cell names the
`v0.1.0` tag.

---

## Common failures

- **Everyone starting from the skeleton branch at once** — three `pyproject.toml`
  files, three `uv.lock` files, and a conflict on every merge but the first.
- **Merging your own pull request** — then nobody read it, and the review half of
  the lab did not happen.
- **One person writing all three functions** — `git log --format='%an'` prints
  one name and the group did the lab once, not three times.
- **`.venv/` committed** — the repository is 200 MB. `git status` before every `add`.
- **`uv.lock` not committed** — CI resolves different versions and fails only there.
- **pytest in `[project.optional-dependencies]`** — `uv sync` skips it and CI dies
  with `Failed to spawn: pytest`.
- **`ModuleNotFoundError` on a fresh clone** — the package is not under `src/`, or
  `pyproject.toml` is not at the repository root.
- **Tests that assert nothing** — `assert word_count("a b")` with no comparison
  passes on any non-zero value.
- **Colab installing from `@main`** — works today, unpinned tomorrow.
