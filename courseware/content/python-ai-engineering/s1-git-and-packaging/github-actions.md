# GitHub Actions

Reference only — not covered in class. Continuous integration: run your tests
automatically on every push.

---

## Why it matters here

Your project is graded partly on "does it work from a clean clone". CI answers
that question every time you push, instead of the night before the deadline.

It also removes the "works on my machine" argument entirely: the runner is not
your machine.

---

## The minimum useful workflow

`.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.12

      - name: Install project
        run: uv sync

      - name: Run tests
        run: uv run pytest -v
```

Commit that file, push, and the Actions tab shows a run.

---

## The anatomy

| Key | Meaning |
|---|---|
| `on` | what triggers it — push, PR, schedule, manual |
| `jobs` | independent units, run in parallel by default |
| `runs-on` | the runner image |
| `steps` | sequential commands within a job |
| `uses` | a prebuilt action from the marketplace |
| `run` | a shell command |

---

## Testing several Python versions

```yaml
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv python install ${{ matrix.python-version }}
      - run: uv sync && uv run pytest
```

The job runs once per entry. This is how a library finds out it broke on one
version.

---

## Adding the linter

```yaml
      - name: Lint
        run: uv run ruff check src/ tests/
```

A separate step, so a failure tells you *which* check failed without reading the
log.

---

## Branch protection

In *Settings → Branches*, require the check to pass before merging. Now a red
test suite blocks the merge instead of merely being visible.

This is the mechanism that makes "the tests are the contract" (Session 2) real
rather than aspirational.

---

## Secrets

Never put a token in the YAML. Store it in *Settings → Secrets and variables →
Actions*, then:

```yaml
      - run: uv run python deploy.py
        env:
          MLARENA_API_KEY: ${{ secrets.MLARENA_API_KEY }}
```

Secrets are masked in logs. Note that anything a workflow can read, a pull
request from a fork may also be able to reach — do not put production
credentials in a public repository's CI.

---

## Cost

Free for public repositories. Private repositories get a monthly allowance of
runner minutes; a test suite of a few seconds will not come close to it.

---

## Check yourself

1. Save the workflow above as `.github/workflows/tests.yml`, then run this. You
   should get exactly the output shown.

   ```bash
   uv run --with pyyaml python -c "
   import yaml
   wf = yaml.safe_load(open('.github/workflows/tests.yml'))
   print([s.get('uses') or s.get('run') for s in wf['jobs']['test']['steps']])
   "
   # -> ['actions/checkout@v4', 'astral-sh/setup-uv@v3', 'uv python install 3.12', 'uv sync', 'uv run pytest -v']
   ```

2. A red check on a pull request is visible but not binding. What turns it into a
   merge gate, and why does this page say that matters?

   **Answer.** Branch protection — *Settings → Branches*, require the check to
   pass before merging. It is the mechanism that makes "the tests are the
   contract" real rather than aspirational.

3. Your workflow needs `MLARENA_API_KEY`. Where does it go, and what is the
   caveat for a public repository?

   **Answer.** Into *Settings → Secrets and variables → Actions*, referenced as
   `${{ secrets.MLARENA_API_KEY }}` — never written in the YAML. Secrets are
   masked in logs, but anything a workflow can read a fork's pull request may be
   able to reach, so keep production credentials out of a public repository's CI.
