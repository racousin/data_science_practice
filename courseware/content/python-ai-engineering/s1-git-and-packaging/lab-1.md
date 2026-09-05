# Lab 1 — Ship a Package

Work in pairs. One repository per pair. Both of you must appear in the commit
history — this is graded on the history, not only on the final state.

**Time:** 70 minutes. **Deliverable:** a GitHub repository URL, and a scored
submission on competition 179.

---

## Part A — Repository (10 min)

1. One of you creates a **private** GitHub repository named
   `paie-lab1-<lastname1>-<lastname2>`.
2. Add your partner and `racousin` as collaborators.
3. Both clone it.
4. Commit a `.gitignore` covering `.venv/`, `__pycache__/`, `*.egg-info/`,
   `.env`, `data/`. The editable install writes `src/<name>.egg-info/`, so
   without that rule you commit build artefacts on your first install.
5. Commit `uv.lock` once it exists — it is what makes the fresh-clone check
   in the grading table reproducible.

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
in a `[dependency-groups]` block named `dev` — not in
`[project.optional-dependencies]`. `uv sync` installs dependency groups and
does not install extras, and `uv sync` is what the grading table runs.

Implement in `core.py`:

```python
def word_count(text: str) -> int:
    """Number of whitespace-separated tokens in `text`."""


def char_frequencies(text: str) -> dict[str, int]:
    """Count of each character, ignoring whitespace and case."""


def longest_word(text: str) -> str:
    """The longest token. Raises ValueError on empty input."""
```

**Specification.** Tokens are whitespace-separated and keep their punctuation —
`"the end."` is two tokens and the second one is `end.`, four characters.
`char_frequencies` folds case and drops whitespace; punctuation and digits are
characters and do count, so `"Aa b"` gives `{"a": 2, "b": 1}`. `longest_word`
breaks ties by taking the **first** token of maximal length, so
`longest_word("aaaa bbbb") == "aaaa"`. These three rules are what Part F is
graded against. The docstrings alone do not pin them down: two defensible
readings of `char_frequencies` and `longest_word` disagree, and a test suite
written against the wrong reading is green *and* wrong.

**Check:** `uv sync` succeeds, and from *outside* the project directory

```bash
uv run --project ~/paie-lab1-<names> python -c "import textstats; print(textstats.__file__)"
```

prints a path under your `src/`. A bare `python -c "import textstats"` from your
home directory will *not* work — that is a different interpreter, which is
exactly what the environment is for.

---

## Part C — Tests (15 min)

Write at least **eight** tests. They must include:

- one `@pytest.mark.parametrize` covering three or more cases
- one `pytest.raises` test for `longest_word("")`
- one fixture used by two different tests
- one test for each of the three specification rules in Part B: punctuation
  kept inside a token, digits counted as characters, a tie taken from the left

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

## Part F — Put it on the leaderboard (10 min)

**PAIE S1 — textstats** (competition `179`) runs your three functions over
twenty hidden texts, three checks each — sixty checks, scored on exact match.
It is the outside opinion on Part B: your tests prove the code does what *you*
expect, this proves it does what the *specification* says.

Write an `agent.py` that exposes the three functions as methods, and upload
`core.py` next to it. The agent directory is on `sys.path`, so a plain
`from core import ...` works — which means the leaderboard grades the code you
actually shipped rather than a copy that has since drifted.

Your class must define **every method the starter template declares, including
`__init__`**. Upload validation compares your class against the template and
rejects the whole submission if one is missing — `Missing method
'__init__(self)' - required by template` — before anything runs.

```python
# agent.py
from core import word_count, char_frequencies, longest_word


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

```bash
uv pip install mlarena-sdk
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.submit(competition_id=179, files=["agent.py", "core.py"])
print(client.leaderboard(179, top=5))
```

**The numbers.** The starter agent as handed out raises on the first call and
scores **pass_rate = 0.000**. Getting `word_count` right and nothing else scores
**0.333**. An implementation that reads the docstrings but not the Part B
specification — `char_frequencies` counting letters only, `longest_word`
returning the *last* of the tied tokens — scores **0.817**. The bar here is
**pass_rate = 1.000**, sixty checks out of sixty, and it is reachable: the
reference implementation scores exactly that. Higher is better, and anything
below 1.000 means at least one specification rule is not implemented — the
leaderboard breaks the score down per function so you can see which one.

Two things about how the run works. `longest_word("")` is never called: every
hidden text is non-empty, and the platform records a raised exception as a
crash, so a competition that *required* a raise would mark every correct
submission broken. That path is graded by your Part C suite instead. And the
first exception ends the run: an agent that starts raising on the ninth text
keeps only what it had already passed — in one measured case sixteen checks of
sixty, **0.267**. So if your score is far below 1.000, read the result message
before you re-read your code: it names the case and the method that crashed.

---

## Grading

| Criterion | Weight |
|---|---|
| `uv sync && uv run pytest` green on a fresh clone | 30% |
| Tests: edge cases covered, failure paths asserted | 25% |
| Commit history: small, readable, both authors present | 20% |
| Pull request: description + real review exchange | 15% |
| textstats leaderboard entry on competition 179 | 10% |

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

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone —
    `git clone <url> /tmp/lab1-check && cd /tmp/lab1-check && uv sync && uv run pytest`
- [ ] Part A — `git ls-files` lists `uv.lock`, and lists nothing under `.venv/`,
    `__pycache__/` or `*.egg-info/`
- [ ] Part B — run from your home directory,
    `uv run --project <your-repo> python -c "import textstats; print(textstats.__file__)"`
    prints a path under `src/`
- [ ] Part C — `uv run pytest -v` reports at least 8 passing tests, among them a
    `parametrize` with three or more cases, a `pytest.raises` on
    `longest_word("")`, and a fixture named by two different tests
- [ ] Part D — the `feature/skeleton` pull request is merged, the branch is
    deleted, and the PR carries a review comment that is not "LGTM"
- [ ] Part E — `git log --graph --oneline` shows the `README.md` merge commit,
    and `git log --format='%an' | sort -u` prints both partners
- [ ] My submission is on the leaderboard of **PAIE S1 — textstats** (#179)
- [ ] My score beats the baseline: **pass_rate = 1.000** — sixty checks of
    sixty, against 0.817 for an implementation that skipped the Part B
    specification

If the last two are not ticked you have not finished the lab, however good the
code is.
