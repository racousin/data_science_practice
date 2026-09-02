# Lab 1 — Build a Dataset

Assemble a small dataset from **two different source types**, with the
provenance, the checks and the documentation that make it usable by someone else.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.

<!-- notes: They work in their project repo from the 12h module — this is the
first commit of the project, not a throwaway. Circulate: the "what is one row"
question is where most of them are stuck. -->

---

## Setup

Work in the repository you packaged in the 12h module, on a branch.

```text
src/collect/
    __init__.py
    sources.py      <- one function per source
    build.py        <- assemble + check + write
    checks.py       <- the quality assertions
tests/test_collect.py
data/raw/           <- gitignored
DATASET.md
```

`data/` stays out of git. Committing a dataset is an automatic deduction.

---

## Part A — Pick two sources (5 min)

Two of the four, not two of the same:

- a **file** you download (CSV, Parquet, an open-data export)
- a **database** (SQLite counts; build one from a dump)
- an **API** (any public one — no key needed for many)
- a **scrape** (only if the terms allow it, and no personal data)

Write down, in `DATASET.md`, the answer to: **what is one row of the final
table?** Do this before writing code.

---

## Part B — Fetch (15 min)

One function per source, each returning a dataframe:

```python
def fetch_orders(path: str) -> pd.DataFrame: ...
def fetch_weather(city: str, since: str) -> pd.DataFrame: ...
```

Requirements:

- raw responses written under `data/raw/` with a dated filename
- `timeout` on every HTTP call, and backoff on 429/5xx
- no credential in the source — `os.environ["..."]`, which raises if unset
- if you scrape: identifying `User-Agent`, ≥1s delay, `robots.txt` checked

---

## Part C — Join and check (15 min)

```python
df = a.merge(b, on=KEY, how="left", validate="m:1")
```

`checks.py` must assert at least four properties, and one of them must be a
row-count invariant across the join:

```python
assert len(df) == len(a)
assert df[KEY].is_unique
assert df["value"].between(LO, HI).all()
assert df["ts"].notna().all()
```

Write the result to `data/processed/dataset_<date>.parquet` — Parquet, not CSV,
and dated.

---

## Part D — Tests (10 min)

Three tests, on fixtures committed under `tests/fixtures/` (a dozen rows, not the
real data):

```python
def test_parse_handles_european_decimals():
    """'1 234,50 €' parses to 1234.50."""

def test_check_rejects_duplicate_keys():
    """checks.check() raises on a duplicated key."""

def test_fetch_is_resumable():
    """A second run over an existing raw file makes no new requests."""
```

The second is the one that matters: a check that never fails is not a check.

---

## Part E — DATASET.md

- what one row is
- a table of columns: name, type, unit, source
- how each source was collected, and when
- licence / terms for each source
- known problems, and what you did about them

---

## Pull request

The description states:

- which two sources, and why those
- one thing that was wrong in the raw data and how you found it
- the row count before and after the join, and why they match
- what you would collect differently with another week

---

## Grading

| Criterion | Weight |
|---|---|
| Two genuinely different sources, both working | 20% |
| Raw responses persisted, dated, gitignored | 15% |
| Fetch is robust: timeouts, backoff, resumable | 15% |
| Four meaningful checks incl. a join invariant | 20% |
| Three tests passing on fixtures | 15% |
| `DATASET.md` complete | 15% |

---

## Automatic deductions

- data committed to git
- a credential in the source or in the history
- a bare `except`
- `.get(key, default)` for a required configuration value
- scraping personal data, or a source whose terms forbid it

---

## Carry it forward

This dataset is the input to Lab 2, where you will preprocess it without leaking,
and it is a plausible starting point for the project.

Collect something you are willing to look at for ten weeks.
