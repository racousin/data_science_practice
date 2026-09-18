# Lab 1 — Build a Dataset

Assemble a small dataset from **two different source types**, with the
provenance, the checks and the documentation that make it usable by someone else.

**Time:** 70 minutes. **Deliverable:** a merged PR in your project repository,
and a submission to the challenge *Multi-Source Store Sales* that scores −20 or
higher (Part F).

<!-- notes: They work in their project repository — this is the first commit
of the project, not a throwaway. Circulate: the "what is one row" question is
where most of them are stuck. Paste the "Today's sandbox" block (make info in
sql_api_sandbox writes LAB.local.md) into this page before the session: Part F
needs DATABASE_URL and WRITER_URL. -->

---

## Setup

Work in your project repository, on a branch.

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

## Part A — Pick two sources (4 min)

Two of the four, not two of the same:

- a **file** you download (CSV, Parquet, an open-data export)
- a **database** (SQLite counts; build one from a dump)
- an **API** (any public one — no key needed for many)
- a **scrape** (only if the terms allow it, and no personal data)

Write down, in `DATASET.md`, the answer to: **what is one row of the final
table?** Do this before writing code.

Then write down **what you will predict from it** — one column, present for every
row. A supervised model will be fitted on this table, so it needs a target and
at least ~2,000 rows. A classification target (two or more classes) is the
simpler path; a regression target works too, with `KFold` in place of
`StratifiedKFold` and `neg_root_mean_squared_error` in place of `roc_auc`. If
your two sources cannot give you that, say so now, not once modelling has
started.

---

## Part B — Fetch (12 min)

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

## Part C — Join and check (8 min)

Two names, and they are not the same column. `KEY` is what you join on. `ID` is
what identifies one row of `a` — the answer you wrote in Part A.

```python
df = a.merge(b, on=KEY, how="left", validate="m:1")
```

`checks.py` must assert at least four properties, and one of them must be a
row-count invariant across the join:

```python
assert len(df) == len(a)
assert df[ID].is_unique      # ID is the grain of `a`. NOT the join key: you just
                             # declared m:1, so many rows per KEY is the design
assert df["value"].between(LO, HI).all()
assert df["ts"].notna().all()
```

`assert df[KEY].is_unique` is the assertion to *not* write here, and it is the
one everybody writes: `validate="m:1"` says the left frame has many rows per key,
so it is guaranteed to fail on a join that is working perfectly.

Write the result to `data/processed/dataset_<date>.parquet` — Parquet, not CSV,
and dated.

---

## Part D — Tests (8 min)

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

## Part E — DATASET.md (4 min)

- what one row is
- a table of columns: name, type, unit, source
- how each source was collected, and when
- licence / terms for each source
- known problems, and what you did about them

---

## Part F — Multi-Source Store Sales (30 min)

A retail chain runs five stores. **Neighborhood_Market** did not record its
sales: predict `quantity_sold` for its 409 items from the four other stores,
whose data is spread over four sources.

| Source | What it provides |
|---|---|
| files: CSV, Excel, JSON | item features and `quantity_sold` |
| an API behind a password | `unit_cost` |
| a page rendered by JavaScript | `customer_score`, `total_reviews` |
| the course PostgreSQL database | `weekly_footfall` of each store |

**Challenge:** <https://ml-arena.com/viewchallenge/190>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s1-store-sales.ipynb)

<!-- notes: The same four source types as the lessons, on one prediction task.
The first push comes before any source work: it proves the key, the download
and the submission format in the first ten minutes. Most students will not
reach the database section in the lab; they finish after the session. -->

---

## Part F — Setup and your first push (10 min)

1. Open the notebook with the Colab badge.
2. In Colab's *Secrets* panel, add `MLARENA_API_KEY` (ML-Arena, Profile →
   API Keys), and `DATABASE_URL` and `WRITER_URL` from the *Today's sandbox*
   section of this page. Never paste a value into a cell.
3. Write your name as `STUDENT` in the Setup cell.
4. Run *Your first push*. It downloads the five files, predicts CityMart's
   mean for every item, submits `submission.csv` and prints its score.

The constant does not clear the bar, and it is not meant to. It proves the
download, the file format and your key before any modelling.

---

## Part F — One source at a time (20 min)

One notebook section per source. Each **TODO** cell has a hint; the cell
after it checks your result and prints the cross-validated MAE of the same
baseline model, so each source's effect is visible.

- **Files:** the separator, header row, column names and sheet layout differ
  per store. Look at the raw lines before calling pandas.
- **API:** request the password at run time, then the prices.
- **Scraping:** headless Chrome renders the page; pick the table by its headers.
- **Database:** join `weekly_footfall` from `retail.stores` on `store_name`.

Then predict Neighborhood_Market, write `submission.csv` and submit it.

**The bar is a score of −20 or higher.** The score is −MAE, so the bar is an
MAE of at most 20 units per item.

---

## Part F — Bonus: the database board

The course database scores predictions too:

```python
submission.to_sql(f"{STUDENT}_predictions", writer, schema="playground",
                  if_exists="replace", index=False)
pd.read_sql("SELECT * FROM retail.leaderboard ORDER BY mae", engine)
```

`writer` connects with `WRITER_URL`, which can create tables in `playground`
only; `engine` with the read-only `DATABASE_URL`. Your row appears 10 to 20
seconds after the write. This board scores only the even-numbered items, so
its MAE differs from your challenge score, and only the challenge counts.

The playground is shared by the class: drop your table when you are done.

---

## Pull request (4 min)

The description states:

- which two sources, and why those
- one thing that was wrong in the raw data and how you found it
- the row count before and after the join, and why they match
- what you would collect differently with another week

---

## Grading

| Criterion | Weight |
|---|---|
| Two genuinely different sources, both working | 15% |
| Raw responses persisted, dated, gitignored | 15% |
| Fetch is robust: timeouts, backoff, resumable | 15% |
| Four meaningful checks incl. a join invariant | 20% |
| Three tests passing on fixtures | 15% |
| `DATASET.md` complete, target column named | 10% |
| A submission to Multi-Source Store Sales scoring −20 or higher | 10% |

---

## Automatic deductions

- data committed to git
- a credential in the source or in the history
- a bare `except`
- `.get(key, default)` for a required configuration value
- scraping personal data, or a source whose terms forbid it

---

## Carry it forward

This dataset is the one you will preprocess and model next, and it is a
plausible starting point for the project.

Collect something you are willing to look at for weeks.

---

## Did you finish the lab?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] Part A: `DATASET.md` answers "what is one row?" and names the target column
- [ ] Part A: the final table has ≥ ~2,000 rows and that target column is present on every one of them
- [ ] Part B: a run from an empty `data/raw/` writes one dated raw file per source; the second run makes zero new HTTP requests
- [ ] Part C: `python -m src.collect.build` exits 0 and writes `data/processed/dataset_<date>.parquet`
- [ ] Part C: duplicating one row of the right-hand frame makes the build fail — I ran it and saw the error, so the checks are not decoration
- [ ] Part D: `uv run pytest -q` reports 3 passed, and `test_check_rejects_duplicate_keys` goes red when I comment out the uniqueness assertion
- [ ] Part E: `DATASET.md` has one line per column and a licence line per source
- [ ] `git status` is clean and nothing under `data/` is tracked
- [ ] Part F: my first push, the constant prediction, is on the leaderboard of Multi-Source Store Sales
- [ ] Part F: my best submission scores **−20 or higher** (MAE ≤ 20 units)

If the last two are not ticked you have not finished the lab, however good the
code is. A submission is scored within a minute, so the first box is ticked in
the lab; if the time runs out before the last source, finish the notebook after
the session.
