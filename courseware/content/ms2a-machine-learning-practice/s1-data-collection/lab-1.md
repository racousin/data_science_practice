# Lab 1 — Build a Dataset

Assemble a small dataset from **two different source types**, with the
provenance, the checks and the documentation that make it usable by someone else.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository,
and one scored run on the session's competition (Part F).

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

## Part A — Pick two sources (4 min)

Two of the four, not two of the same:

- a **file** you download (CSV, Parquet, an open-data export)
- a **database** (SQLite counts; build one from a dump)
- an **API** (any public one — no key needed for many)
- a **scrape** (only if the terms allow it, and no personal data)

Write down, in `DATASET.md`, the answer to: **what is one row of the final
table?** Do this before writing code.

Then write down **what you will predict from it** — one column, present for every
row. Labs 3 and 4 fit supervised models on this table, so it needs a target and
at least ~2,000 rows. A classification target (two or more classes) is the path
the rest of the course is written for; a regression target works, but you will
substitute `KFold` for `StratifiedKFold` and `neg_root_mean_squared_error` for
`roc_auc` throughout. If your two sources cannot give you that, say so now, not
in week three.

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

## Part F — Put it on the board (5 min)

The session's competition is **Global Weather Forecast** (id `177`): a source
that never stops. Every six hours it hands an agent 48 hours of observations for
120 cities and records what that agent predicts for +6h, +12h and +24h. Nobody
can look the answer up, because the hours have not happened yet.

You are not forecasting today — that is Session 3. You are collecting from a live
feed and getting on the board. Write `agent.py`:

```python
import numpy as np

class Agent:
    def predict(self, request):
        history = np.asarray(request["history"])       # (120, 48, 8)
        f = request["feature_names"]
        t, w = f.index("temperature"), f.index("wind_speed")
        n_h = len(request["horizons"])
        # The collection window IS the feature: average the last 24 hours.
        temp = history[:, -24:, t].mean(axis=1, keepdims=True)
        wind = history[:, -24:, w].mean(axis=1, keepdims=True)
        return {
            "temperature": np.repeat(temp, n_h, 1).tolist(),
            "wind_speed":  np.repeat(wind, n_h, 1).tolist(),
            "rain_prob":   np.full((history.shape[0], n_h), 0.04).tolist(),
        }
```

```bash
uv pip install mlarena-sdk        # the package is mlarena-sdk; it imports as mlarena
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # Profile -> API Keys
client.submit(competition_id=177, files=["agent.py"])
print(client.leaderboard(177).head())
```

**The numbers.** The metric is a skill score against persistence — carrying the
last observation forward — averaged over temperature, wind and rain. **Higher is
better.** Measured by replaying 339 real 6-hourly runs over June–August 2026
(119,520 scored samples) and published on the competition page:

| Agent | Skill |
|---|---|
| Always predict 0 | −0.998 |
| Persistence — the reference the metric is defined against | −0.007 |
| **Trailing 24h mean — the agent above** | **0.097** |
| The starter agent shipped with the competition | 0.217 |

Those are replay numbers: the live board scores a different stretch of weather,
so the same agent will not match them to the third decimal. Two agents within
about **0.006** of each other are not distinguishable in the first place.

The bar for this lab is the reference: **skill > 0.000**. The agent above clears
it by a wide margin; beating 0.217 needs a model, and that is Session 3's
problem.

**Your first run produces no score, and that is correct.** The forecast is
recorded now and scored later, once the hours it covers have actually happened —
about six hours. Submit before you write the PR, not after.

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
| A scored run on competition 177 | 10% |

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

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] Part A: `DATASET.md` answers "what is one row?" and names the target column
- [ ] Part A: the final table has ≥ ~2,000 rows and that target column is present on every one of them
- [ ] Part B: a run from an empty `data/raw/` writes one dated raw file per source; the second run makes zero new HTTP requests
- [ ] Part C: `python -m src.collect.build` exits 0 and writes `data/processed/dataset_<date>.parquet`
- [ ] Part C: duplicating one row of the right-hand frame makes the build fail — I ran it and saw the error, so the checks are not decoration
- [ ] Part D: `uv run pytest -q` reports 3 passed, and `test_check_rejects_duplicate_keys` goes red when I comment out the uniqueness assertion
- [ ] Part E: `DATASET.md` has one line per column and a licence line per source
- [ ] `git status` is clean and nothing under `data/` is tracked
- [ ] Part F: my agent is on the leaderboard of Global Weather Forecast (#177)
- [ ] Part F: my score beats the baseline: **skill > 0.000** — above persistence, the reference floor

If the last two are not ticked you have not finished the lab, however good the
code is. The competition scores a forecast only once the hours it covers have
happened, so the last box goes from *submitted* to *scored* about six hours after
you deploy: submit in the lab, tick it the same evening.
