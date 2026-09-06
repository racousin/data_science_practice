# Continuous Integration with GitHub Actions

"I ran the tests" is a claim. A green tick on a pull request is evidence,
produced on a machine that has never met your laptop.

<!-- notes: 25 minutes. Commit a workflow live and watch the run go green in the
Actions tab. Then break a test, push, and watch it go red — the red run is what
makes the point. -->

---

## What happens after you push

![The CI pipeline](assets/s1-git-and-packaging/github-actions/ci-pipeline.png)

GitHub looks in `.github/workflows/` for YAML files, starts a fresh virtual
machine per job, and runs the steps you wrote. The result is a status attached
to the commit — the tick or the cross you see next to it.


---

## The runner is not your machine

That is the entire value. It has no
`.venv` you forgot to commit, no environment variable you set in March, and no
file that exists only in your Downloads folder.

---

## CI and CD

![CI and CD](assets/s1-git-and-packaging/github-actions/ci-vs-cd.png)


---

## The two halves

- **Continuous integration** — every change is built, linted and tested,
  automatically, on a clean machine.
- **Continuous delivery** — a green `main` is automatically published: a
  release, a Docker image, a deployed service, a registered model.

This course does CI properly and CD once, at the end of this lesson, as a
GitHub release that Colab can `pip install`.

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
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync
      - run: uv run pytest -q
```

Commit it, push, and the *Actions* tab shows a run. That is the whole thing —
seventeen lines and your project now tests itself.

---

## Reading the file

| Key | Means |
|---|---|
| `on` | what triggers it — push, pull request, schedule, manual |
| `jobs` | independent units; they run in parallel by default |
| `runs-on` | which runner image |
| `steps` | commands, in order, inside one job |
| `uses` | a prebuilt action from the marketplace |
| `run` | a shell command — the same shell you learned this morning |

`uv sync` here is doing exactly what it does on your laptop: read `uv.lock`,
create `.venv`, install. Which is why an uncommitted lockfile makes CI fail in a
way your machine never will.

---

## Adding the linter

```yaml
      - run: uv run ruff check src/ tests/
      - run: uv run ruff format --check src/ tests/
```

Two separate steps, so a red run names which one failed without you opening the
log. `--check` means "report, do not rewrite" — a CI runner rewriting your files
would achieve nothing, since it throws the machine away afterwards.

---

## Several Python versions at once

```yaml
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install ${{ matrix.python-version }}
      - run: uv sync && uv run pytest -q
```

Three jobs, in parallel, same steps. This is how a library discovers it broke on
one version — and how you find out that `requires-python = ">=3.11"` in your
`pyproject.toml` was optimistic.

<!-- notes: Worth saying: this is also the cheapest possible demonstration of
why the lockfile and the declaration are different files. -->

---

## Making it a rule

A red run that nobody has to look at is decoration.

*Settings → Branches → Add branch ruleset*, on `main`:

- Require a pull request before merging
- **Require status checks to pass** → select `test`
- Require the branch to be up to date before merging

Now a failing test **blocks the merge button**. The tests stopped being a habit
and became part of the process.

---

## The badge

In your `README.md`:

```markdown
![tests](https://github.com/<you>/<repo>/actions/workflows/tests.yml/badge.svg)
```

It renders as a live green or red image at the top of your repository. It is
also honest in a way a README sentence is not: it goes red on its own.

---

## Caching, and why runs get slow

Every run starts from nothing, so every run reinstalls your dependencies. On a
project with `torch` that is minutes.

```yaml
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
```

The cache key is the lockfile: same lock, same cache. Change a dependency and it
correctly rebuilds.

---

## Secrets

Never put a token in the YAML — the YAML is in the repository.

*Settings → Secrets and variables → Actions → New repository secret*, then:

```yaml
      - run: uv run python scripts/submit.py
        env:
          MLARENA_API_KEY: ${{ secrets.MLARENA_API_KEY }}
```

Secrets are masked in the logs. Two warnings that matter:

- A workflow triggered by a **pull request from a fork** does not get your
  secrets, by design. That is a protection, not a bug.
- Anything a workflow can read, a malicious change to that workflow can print.
  Review changes to `.github/workflows/` like you review changes to code.

---

## Continuous delivery, once

Tag a commit and let GitHub publish it:

```yaml
name: release
on:
  push:
    tags: ["v*"]

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: softprops/action-gh-release@v2
```

```bash
git tag v0.1.0
git push origin v0.1.0
```

The tag becomes a GitHub release, and the release becomes something anyone —
including a Colab notebook — can install:

```python
!pip install -q git+https://github.com/<you>/<repo>.git@v0.1.0
```

Pinning the tag rather than the branch is what makes a notebook reproducible six
months later. `@main` is a moving target.

---

## Other things CI is used for

Once the mechanism exists, it is not only tests:

| Trigger | Useful for |
|---|---|
| `pull_request` | tests, lint, type check, coverage report |
| `schedule` | a nightly run against fresh data |
| `workflow_dispatch` | a button in the UI to run a job by hand |
| `release` | build and publish a package or an image |

In the 30-hour module, a scheduled workflow that retrains a model and posts the
score is a genuinely small amount of YAML.

---

## Cost

Free and unlimited for **public** repositories. Private repositories get a
monthly allowance of runner minutes; a Python test suite of a few seconds will
not come close to exhausting it.

This is one practical reason to make your lab repositories public.

---

## When it is red and you cannot see why

1. **Read the failing step, not the whole log.** The UI collapses to it.
2. **Reproduce it locally the way CI does** — from a clean clone:

   ```bash
   git clone <url> /tmp/ci-check && cd /tmp/ci-check
   uv sync && uv run pytest -q
   ```

3. **If it passes locally and fails in CI**, the difference is something on your
   machine that is not in the repository. Nine times out of ten it is an
   uncommitted file, a data file, or an environment variable.

That third case is not CI being awkward. It is CI doing the one job you set it.
