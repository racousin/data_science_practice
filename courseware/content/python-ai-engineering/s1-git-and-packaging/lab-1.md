# Lab 1 — Ship a Package to GitHub

Build a Python package, prove it with tests, make GitHub run those tests for
you, and use the result from a Colab notebook you have never opened before.

**Time:** 60 minutes. **Deliverable:** a GitHub repository URL. Everything is
graded from that URL — there is nothing to upload anywhere else.

<!-- notes: Everyone works solo here; Lab 2 is the one that needs a partner.
Walk the room during Part B — the src/ layout and the editable install are where
people stall. Put the four verification commands on the board. -->

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

## Part A — The repository (8 min)

1. On GitHub, **New repository** → name it `textstats` → **Public** → tick *Add
   a README*.
   Public, because CI minutes are free on public repositories and branch
   protection is not restricted. There is nothing secret in this.
2. Clone it and enter it:

```bash
git clone git@github.com:<you>/textstats.git
cd textstats
```

3. Commit a `.gitignore` before anything else exists to be ignored:

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

**Check:** `git log --oneline` shows two commits; `git status` is clean.

---

## Part B — The package (15 min)

Work on a branch, not on `main`:

```bash
git switch -c feature/skeleton
uv init --lib --name textstats .    # or create the files by hand
uv add --dev pytest ruff
```

`--dev` puts them in `[dependency-groups]`, which `uv sync` installs. Do **not**
use `[project.optional-dependencies]` — `uv sync` skips extras, and the
fresh-clone check below is what your grade is read from.


---

## Part B — the files

`pyproject.toml` must end up with at least:

```toml
[project]
name = "textstats"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
```

Then write `src/textstats/core.py`:

```python
def word_count(text: str) -> int:
    """Number of whitespace-separated tokens in `text`."""

def char_frequencies(text: str) -> dict[str, int]:
    """Count of each character, ignoring whitespace and case."""

def longest_word(text: str) -> str:
    """The longest token. Raises ValueError on empty input."""
```

---

## Part B — the specification

The docstrings do not pin these down, and two defensible readings disagree. This
is the specification; the tests in Part C are graded against it.

- **Tokens are whitespace-separated and keep their punctuation.**
  `"the end."` is two tokens and the second is `end.` — four characters.
- **`char_frequencies` folds case and drops whitespace.** Punctuation and digits
  *are* characters and do count: `"Aa b"` → `{"a": 2, "b": 1}`.
- **`longest_word` breaks ties to the left**: `longest_word("aaaa bbbb")` is
  `"aaaa"`.
- **`longest_word("")` raises `ValueError`** — it does not return `""`.


---

## Part B — exports and the check

Re-export the three from `src/textstats/__init__.py` so `from textstats import
word_count` works:

```python
from textstats.core import char_frequencies, longest_word, word_count

__all__ = ["word_count", "char_frequencies", "longest_word"]
```

**Check**, from your **home directory**, not from the project:

```bash
uv run --project ~/textstats python -c "import textstats; print(textstats.__file__)"
```

It must print a path under your `src/`. A bare `python -c "import textstats"`
from home will not work — that is a different interpreter, which is the whole
point of the environment.

---

## Part C — Tests (15 min)

Write at least **eight** tests in `tests/test_core.py`. They must include:

- one `@pytest.mark.parametrize` with three or more cases
- one `pytest.raises(ValueError)` for `longest_word("")`
- one fixture used by two different tests
- one test for each specification rule: punctuation kept inside a token, digits
  counted as characters, a tie taken from the left


---

## Part C — what a test looks like

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

Eight or more passing tests. If a test passes before you wrote the
implementation, it is not testing anything.

---

## Part D — Quality gate (5 min)

Add ruff configuration to `pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

```bash
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
```

Commit the formatting on its own, so it never mixes with real work:

```bash
git add -A && git commit -m "Apply ruff format and fix lint findings"
```

---

## Part E — Continuous integration (7 min)

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
      - run: uv run ruff check src/ tests/
      - run: uv run pytest -q
```


---

## Part E — commit, push, watch it run

Commit `uv.lock` — without it, `uv sync` on the runner resolves different
versions from yours, and CI stops being evidence about *your* code.

```bash
git add -A && git commit -m "Add CI: ruff and pytest on every push"
git push -u origin feature/skeleton
```

Open the pull request. Watch the check run. **Merge only when it is green.**

Then add the badge to `README.md` on `main`:

```markdown
![tests](https://github.com/<you>/textstats/actions/workflows/tests.yml/badge.svg)
```

**Check:** the Actions tab shows a successful run, and the badge is green.

---

## Part F — Release it (3 min)

A tag is what makes the next part reproducible.

```bash
git switch main && git pull
git tag v0.1.0
git push origin v0.1.0
```

---

## Part G — Use it in Colab (7 min)

Open a **new** Colab notebook — not one you have used before. Two cells:

```python
!pip install -q git+https://github.com/<you>/textstats.git@v0.1.0
```

```python
import textstats
print(textstats.word_count("the quick brown fox"))       # 4
print(textstats.longest_word("the quick brown fox"))     # quick
print(textstats.char_frequencies("Aa b"))                # {'a': 2, 'b': 1}
```

This is the moment the packaging was for. Nothing was copied and nothing was
pasted: Colab installed the exact code you pushed.

Then, *File → Save a copy in GitHub*, into your `textstats` repository as
`notebooks/demo.ipynb`. Commit message: `Add Colab demo`.

**Check:** the notebook is in your repository, and its install cell names the
`v0.1.0` tag.

---

## Part H — Optional: put it on a leaderboard

Not required, and not graded — the deliverable is the repository. But if you
have an ML-Arena account, **PAIE S1 — textstats** (competition `179`) is a free
outside opinion: it runs your three functions over twenty hidden texts, three
checks each, and scores exact matches.

Your tests prove the code does what *you* expect. This proves it does what the
Part B specification says.


---

## Part H — the submission shape

```python
# agent.py, next to a copy of core.py
from core import char_frequencies, longest_word, word_count


class Agent:
    def __init__(self):
        pass

    def word_count(self, text):
        return word_count(text)

    def char_frequencies(self, text):
        return char_frequencies(text)

    def longest_word(self, text):
        return longest_word(text)
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")     # from your Profile page
client.submit(competition_id=179, files=["agent.py", "core.py"])
```

Your class must define **every method the starter template declares, including
`__init__`** — upload validation rejects the submission before anything runs
otherwise.


---

## Part H — the numbers

A starter that raises scores **0.000**. `word_count` alone
scores **0.333**. An implementation that read the docstrings but not the Part B
specification — letters only in `char_frequencies`, the *last* of the tied
tokens in `longest_word` — scores **0.817**. The bar is **1.000**, sixty of
sixty, and the reference implementation reaches it. Anything below means one
specification rule is not implemented, and the per-function breakdown says
which.

---

## The four commands your grade is read from

Run them yourself before you say you are finished. From a directory that is not
your project:

```bash
git clone https://github.com/<you>/textstats.git /tmp/lab1-check
cd /tmp/lab1-check
uv sync
uv run pytest -q
```

Green output, from a clone that has never seen your laptop's state.

---

## Grading

| Criterion | Weight |
|---|---|
| `uv sync && uv run pytest` green on a fresh clone | 25% |
| Tests: eight or more, edge cases and failure paths asserted | 25% |
| CI workflow present, green, and required on the PR | 15% |
| Commit history: small, readable, one idea each | 15% |
| Colab notebook installs from the tag and runs | 15% |
| README: what it is, how to install, how to test | 5% |

---

## Common failures

- **`.venv/` committed** — the repository is 200 MB. `git status` before every `add`.
- **`uv.lock` not committed** — CI resolves different versions and fails only there.
- **pytest in `[project.optional-dependencies]`** — `uv sync` skips it and CI dies
  with `Failed to spawn: pytest`.
- **`ModuleNotFoundError` on a fresh clone** — the package is not under `src/`, or
  `pyproject.toml` is not at the repository root.
- **Tests that assert nothing** — `assert word_count("a b")` with no comparison
  passes on any non-zero value.
- **Colab installing from `@main`** — works today, unpinned tomorrow.

---

## If you finish early

- Add a `[project.scripts]` entry point so `textstats <file>` prints the word
  count from the command line, and open it as a second pull request.
- Add a matrix to the workflow for Python 3.11, 3.12 and 3.13.
- Add `pre-commit` with ruff and `nbstripout`.

---

## Did you validate this lab?

- [ ] `git clone <url> /tmp/lab1-check && cd /tmp/lab1-check && uv sync && uv run pytest`
      is green
- [ ] `git ls-files` lists `uv.lock` and `pyproject.toml`, and lists nothing
      under `.venv/`, `__pycache__/` or `*.egg-info/`
- [ ] From my home directory,
      `uv run --project ~/textstats python -c "import textstats; print(textstats.__file__)"`
      prints a path under `src/`
- [ ] `uv run pytest -v` reports ≥ 8 passing tests, including a `parametrize`
      with ≥ 3 cases, a `pytest.raises` on `longest_word("")`, and a fixture
      named by two different tests
- [ ] The Actions tab shows a green run on `main`, and the README badge is green
- [ ] `git tag --list` shows `v0.1.0` and it is pushed
- [ ] A brand-new Colab notebook installs from `@v0.1.0` and prints `4`,
      `quick` and `{'a': 2, 'b': 1}`
- [ ] That notebook is committed to the repository

The last three are the ones that distinguish a package from a folder of Python
files. If they are not ticked, the lab is not finished.
