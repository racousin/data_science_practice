# Lab 2 — A Leak-Free Pipeline

Take the dataset you built in Lab 1 and turn it into a model-ready matrix
through a single fitted object, with tests that prove nothing leaked.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository,
and one scored submission on the session's competition (Part F).

<!-- notes: 45 minutes. Part C is the one that runs over — tell them at minute
20 that a naive target encoder plus a failing test is worth more than a correct
encoder with no test. Circulate for the split: several will still split after
imputing. -->

---

## Setup

Same repository, new branch. Lab 1's Parquet file is the input.

```text
src/preprocess/
    __init__.py
    pipeline.py     <- build_pipeline() -> unfitted Pipeline
    features.py     <- custom transformers
tests/test_pipeline.py
models/pipeline_<date>.joblib
PREPROCESSING.md
```

`build_pipeline()` takes the column lists and returns an **unfitted** object.
Nothing in `src/` may call `fit` on anything at import time.

---

## Part A — Split first (4 min)

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
```

Before any imputation, encoding or scaling. If your rows are ordered in time,
split by time instead — the test set is the tail, not a random sample. If one
entity spans several rows, use `GroupShuffleSplit` on that entity.

Write the three column lists — numeric, categorical, dropped — as module
constants. A column in none of them must raise, not disappear.

---

## Part B — The ColumnTransformer (12 min)

```python
pre = ColumnTransformer([
    ("num", num_pipe, NUM_COLS),
    ("cat", cat_pipe, CAT_COLS),
], remainder="drop")
```

Requirements:

- numeric branch: imputer with `add_indicator=True`, then a scaler you justify
- categorical branch: imputer, then `OneHotEncoder(handle_unknown="ignore")`
- at least two engineered features from Session 2 — a ratio, a datetime
  decomposition, a cyclical pair or a group aggregate
- custom steps subclass `BaseEstimator, TransformerMixin` and live in
  `features.py`; they may not read the target in `transform`

`fit` is called exactly once in your training script, on `X_tr`.

---

## Part C — Target encoding, out of fold (8 min)

Pick your highest-cardinality categorical column — `HIGH_CARD` below, another
module constant — and encode it twice: the naive
way, with a `groupby` mean over the whole training set, and with sklearn's
`TargetEncoder(cv=5)` inside the pipeline.

```python
naive_map = y_tr.groupby(X_tr[HIGH_CARD]).mean()      # the target lives in y_tr,
naive_tr  = X_tr[HIGH_CARD].map(naive_map).fillna(y_tr.mean())   # not in X_tr
naive_te  = X_te[HIGH_CARD].map(naive_map).fillna(y_tr.mean())
```

Score each encoding **on `X_te`** — not with `cross_val_score` on the
pre-computed column. A column that was already fitted on all of `X_tr` is inside
every fold, so cross-validation cannot see the leak that produced it and will
rank the naive encoder *higher*.

Seeing that inversion is the point. It is the same mistake
*The Preprocessing Contract* warns about, arriving one lesson later in your own
code, and it is the whole reason the encoder has to live inside the Pipeline
rather than in a column you computed first.

Report in the PR: each encoding's correlation with the target, its
cross-validated score, and its held-out score on `X_te`. The gap between the
last two is the deliverable.

---

## Part D — Tests (8 min)

In `tests/test_pipeline.py`, on a small committed fixture:

```python
def test_transformer_uses_only_training_statistics():
    """Fitting on train, then transforming one held-out row, yields the value
    predicted from the train median — not from the row itself."""

def test_unseen_category_does_not_crash_inference():
    """A category absent from train transforms to an all-zero block."""

def test_target_encoding_is_out_of_fold():
    """For a category appearing once, the out-of-fold code differs from the
    naive group mean."""

def test_fitted_pipeline_round_trips():
    """joblib.load(joblib.dump(pipe)) transforms a row identically."""
```

The third test is the one that matters. Make it fail against the naive encoder
before you make it pass.

---

## Part E — Serialise and document (4 min)

```python
joblib.dump(pipe, f"models/pipeline_{date.today()}.joblib")
```

Dump the **fitted** pipeline, dated, and gitignore `models/`. In half a page,
`PREPROCESSING.md` records:

- every column: kept, dropped or engineered, and why
- the imputation strategy per column type and the missing rate before it
- the encoding per categorical column, with its cardinality
- rows removed as duplicates or impossible values, with counts
- the sklearn and pandas versions the artefact was produced with

---

## Part F — Put it on the board (5 min)

The session's competition is **Allergies : profils IgE et symptômes cutanés**
(id `176`) — the same problem shape as your lab, on real clinical open data from
the Société Française d'Allergologie: 1,187 training patients, 241 IgE columns
where an empty cell means *not measured* rather than *negative*, 7 demographic
columns, and a binary target.

**The competition page is in French. Here is the whole of what you need.**
Predict, for each of the 639 test patients, the probability that they show skin
symptoms. `train.csv` carries the target in its `skin_symptoms` column — split it
out first, which is Part A again on somebody else's table. Then feed the features
through your `build_pipeline()`, fit a `LogisticRegression`, and write
`submission.csv` with exactly two columns — `patient_id,skin_symptoms` — one row
per test id, the second column a probability.

```bash
uv pip install mlarena-sdk        # the package is mlarena-sdk; it imports as mlarena
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # Profile -> API Keys
client.download_dataset(176, dest_dir="data/raw")
client.submit(competition_id=176, files=["submission.csv"])
print(client.leaderboard(176).head())
```

**The numbers.** The metric is **ROC AUC** and **higher is better**: 0.500 is
chance, 1.000 is perfect. Measured on this exact split and published on the
competition page:

| Approach | ROC AUC |
|---|---|
| Constant submission | 0.500 |
| **Benchmark — a calibrated count of positive IgE** | **0.644** |
| Regularised logistic regression | 0.785 |
| Gradient boosting | 0.795 |
| Random forest, 500 trees | 0.813 |

The bar is the benchmark: **ROC AUC > 0.644**, and you can read it straight off
the leaderboard — the row named `__benchmark__` is that reference solution, run
by the competition itself. Sampling noise on 639 patients is about **±0.016**, so
two scores less than 0.03 apart are not distinguishable: aim for a clear gap, not
a third decimal.

A regularised logistic regression on these columns scores 0.785, so a leak-free
pipeline that lands *under* 0.644 is a wiring problem rather than a modelling
one: check `handle_unknown="ignore"` on the encoder, check that the "not
measured" cells did not become zeros, and check that `fit` was called on `X_tr`
alone.

**One warning about the starter notebook** shipped with the competition. Its
stronger model does this:

```python
X_all = pd.concat([train.drop(columns=[TARGET]), test], keys=["train", "test"])
cat = pd.get_dummies(X_all[CATEG].astype(str), dummy_na=True)   # do not copy this
```

The encoding vocabulary is therefore fitted on the test rows: some of its dummy
columns exist only because test patients were in the frame. On this competition
it costs nothing, because the test *labels* are never touched — but in your own
repository it is the second line of the *Automatic deductions* list below, "a
statistic computed over the full dataframe before the split". Fit the encoder on
`train`, `transform` `test`, and you get the same score with a pipeline you can
defend.

---

## Pull request (4 min)

The description states:

- how you split, and why that split and not a random one
- the naive versus out-of-fold target-encoding numbers from Part C
- one leak you found in your own code while writing the tests
- the shape of the matrix before and after the pipeline

---

## Grading

| Criterion | Weight |
|---|---|
| Split happens before any fitted transformation | 15% |
| `ColumnTransformer` with both branches, `remainder="drop"` | 20% |
| Two engineered features, implemented as transformers | 10% |
| Out-of-fold target encoding, with the naive comparison reported | 15% |
| Four tests passing, incl. the leak test and the unseen category | 20% |
| Fitted pipeline serialised + `PREPROCESSING.md` complete | 10% |
| A scored submission on competition 176 | 10% |

---

## Automatic deductions

- any `fit` or `fit_transform` called on test data
- a statistic computed over the full dataframe before the split
- `remainder="passthrough"` with unlisted columns
- an imputation or encoding done in a notebook and not in `pipeline.py`
- `errors="coerce"` or a bare `except` in the cleaning code
- the serialised artefact committed to git

---

## Carry it forward

Lab 3 fits models on this matrix and compares them honestly. Every score it
produces is exactly as trustworthy as the object you built today.

> If you cannot point at the line that fits a transformer, you do not know
> whether your model works.

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] Part A: `grep -rn "\.fit(\|fit_transform(" src/` shows no call whose argument is `X_te`
- [ ] Part B: `build_pipeline().transform(X_tr)` raises `NotFittedError` — the object it returns is unfitted
- [ ] Part B: both branches present, `remainder="drop"` set, and I can account for every column of `pre.fit_transform(X_tr).shape`
- [ ] Part C: the PR reports three numbers per encoding — correlation, cross-validated score, held-out score on `X_te`
- [ ] Part D: `uv run pytest -q` reports 4 passed, and `test_target_encoding_is_out_of_fold` goes red against the naive column
- [ ] Part E: `models/pipeline_<date>.joblib` exists, is gitignored, and reloads to transform one row identically
- [ ] Part E: `PREPROCESSING.md` names every column as kept, dropped or engineered
- [ ] Part F: my submission is on the leaderboard of Allergies : profils IgE et symptômes cutanés (#176)
- [ ] Part F: my score beats the baseline: **ROC AUC > 0.644** — the benchmark shipped with the competition

If the last two are not ticked you have not finished the lab, however good the
code is. Submissions to #176 are scored immediately, so there is no excuse for
leaving the last box empty at the end of the session.
