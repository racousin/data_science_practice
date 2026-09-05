# Packaging & Tests

Turning a folder of scripts into something installable, importable, and
verifiable.

<!-- notes: 35 minutes. The src/ layout argument is worth making properly —
it is the one that prevents the "works until I cd somewhere else" bug. -->

---

## Why package at all

You have `train.py`, `data.py`, `metrics.py` in one folder. It works. Then:

- a notebook in `notebooks/` cannot import them
- a teammate runs from a different directory and gets `ModuleNotFoundError`
- you want to reuse `metrics.py` in the next project

Packaging fixes all three by making your code *importable by name*, from
anywhere, once installed.

---

## Layout

```text
my_project/
├── pyproject.toml
├── README.md
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── data.py
│       └── metrics.py
└── tests/
    ├── test_data.py
    └── test_metrics.py
```

`__init__.py` is what marks a directory as a package. It can be empty.

---

## Why `src/`

Without it, your project root is on `sys.path` and `import my_project` picks up
the *source directory* whether or not the package is correctly installed.

With `src/`, the only way to import your code is to install it. So your tests
exercise the same thing your users get — and a broken `pyproject.toml` fails
immediately rather than three weeks later on someone else's machine.

---

## pyproject.toml

One file, and it is a standard (PEP 621) rather than a tool-specific format.

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "my-project"
version = "0.1.0"
description = "Session 1 deliverable"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "pandas>=2.2"]

[dependency-groups]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6"]
```

Note the two names: the **distribution** name (`my-project`, with a hyphen) and
the **import** name (`my_project`, with an underscore). They differ by
convention and that is fine.

`[dependency-groups]` (PEP 735) is where development tools go — they are needed
to *work on* the project, not to *use* it. `uv sync` installs them by default,
which is what makes `uv sync && uv run pytest` work on a fresh clone. The older
`[project.optional-dependencies]` spelling is for extras your users opt into,
and `uv sync` does **not** install those. Put pytest there and the fresh-clone
check either dies with `error: Failed to spawn: pytest` or — worse — picks up
some other `pytest` that happens to be on the machine's PATH and cannot import
your package.

---

## Editable install

```bash
uv pip install -e .
uv pip install -e . --group dev   # with the dev tools
```

`-e` (editable) links the installed package to your source directory. Edit
`src/my_project/data.py`, and the change is live — no reinstall.

Check it worked:

```bash
uv run python -c "import my_project; print(my_project.__file__)"
```

It must print a path under your `src/`. Use `uv run`, not a bare `python`: a
bare `python` is whatever interpreter is on your PATH, which is not the
project environment you just installed into.

---

## A console entry point

Turn a function into a command your users can type:

```toml
[project.scripts]
my-train = "my_project.cli:main"
```

After reinstalling, `my-train` runs `main()` from `src/my_project/cli.py`. This
is how `pytest`, `ruff` and `uv` all work.

---

## Why test

Two reasons, in order of how much they will matter to you:

1. **You can change code without fear.** A test suite is what makes refactoring
   a decision rather than a gamble.
2. **It is the contract with a coding agent.** In Session 2 the agent writes
   code; the tests are how you find out whether it works. An agent with no
   tests is a very fast way to produce plausible, wrong code.

---

## pytest

Test files are `test_*.py`. Test functions are `test_*`. Assertions are plain
`assert`.

```python
# tests/test_metrics.py
from my_project.metrics import accuracy


def test_accuracy_all_correct():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0


def test_accuracy_none_correct():
    assert accuracy([1, 1, 1], [0, 0, 0]) == 0.0
```

```bash
uv run pytest
uv run pytest -v                  # one line per test
uv run pytest tests/test_data.py  # one file
uv run pytest -k accuracy         # tests matching a name
```

---

## Testing failure, not just success

The interesting half of a test suite is what happens on bad input. Per the
fail-fast principle: bad input should *crash clearly*, not return something
plausible.

```python
import pytest
from my_project.metrics import accuracy


def test_accuracy_rejects_length_mismatch():
    with pytest.raises(ValueError):
        accuracy([1, 0], [1, 0, 1])
```

If that test fails because your function silently returned `0.66`, you have
found a real bug.

---

## Fixtures

Shared setup, without copy-paste:

```python
import pytest
import pandas as pd


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})


def test_row_count(sample_df):
    assert len(sample_df) == 3
```

Any test that names `sample_df` as a parameter gets a fresh one.

---

## Parametrising

One test body, many cases:

```python
@pytest.mark.parametrize("pred,true,expected", [
    ([1, 1], [1, 1], 1.0),
    ([1, 0], [0, 1], 0.0),
    ([1, 0], [1, 1], 0.5),
])
def test_accuracy(pred, true, expected):
    assert accuracy(pred, true) == expected
```

Each tuple is reported as a separate test, so a failure tells you *which* case
broke.

---

## Unit vs integration

| | Unit | Integration |
|---|---|---|
| Scope | one function | several components together |
| Speed | milliseconds | seconds |
| Fails when | that function is wrong | the wiring is wrong |
| How many | many | a few |

You want a lot of the first and a handful of the second. A pipeline that loads
a small CSV, trains, and asserts the score beats a constant baseline is an
excellent integration test.

---

## Coverage — and its limit

```bash
uv run pytest --cov=my_project
```

Coverage tells you which lines *ran*. It does not tell you whether the
assertions were meaningful. A suite with 100% coverage and no assertions on
edge cases is 100% decorative.

Use it to find untested files, not as a target to hit.

---

## The finished project

```text
my_project/
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
├── src/my_project/{__init__,data,metrics}.py
└── tests/{test_data,test_metrics}.py
```

```bash
git clone <url> && cd my_project
uv sync
uv run pytest
```

Three commands, green output. That is the deliverable.
