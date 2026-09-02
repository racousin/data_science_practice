# Lab 2 — A Leak-Free Pipeline

Take the dataset you built in Lab 1 and turn it into a model-ready matrix
through a single fitted object, with tests that prove nothing leaked.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.

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

## Part A — Split first (5 min)

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
```

Before any imputation, encoding or scaling. If your rows are ordered in time,
split by time instead — the test set is the tail, not a random sample. If one
entity spans several rows, use `GroupShuffleSplit` on that entity.

Write the three column lists — numeric, categorical, dropped — as module
constants. A column in none of them must raise, not disappear.

---

## Part B — The ColumnTransformer (15 min)

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

## Part C — Target encoding, out of fold (10 min)

Pick your highest-cardinality categorical column and encode it twice: the naive
way, with a `groupby` mean over the whole training set, and with sklearn's
`TargetEncoder(cv=5)` inside the pipeline.

```python
naive = X_tr["city"].map(X_tr.groupby("city")[TARGET].mean())
```

Report both in the PR: each column's correlation with the target, and the
cross-validated score using each. The naive column will look better and score
worse. That gap is the deliverable.

---

## Part D — Tests (10 min)

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

## Part E — Serialise and document (5 min)

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

## Pull request

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
| Two engineered features, implemented as transformers | 15% |
| Out-of-fold target encoding, with the naive comparison reported | 15% |
| Four tests passing, incl. the leak test and the unseen category | 20% |
| Fitted pipeline serialised + `PREPROCESSING.md` complete | 15% |

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
