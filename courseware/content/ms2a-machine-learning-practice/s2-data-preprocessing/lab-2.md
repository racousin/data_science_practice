# Lab 2 — A Leak-Free Pipeline

Take the dataset you built in Lab 1 and turn it into a model-ready matrix
through a single fitted object, with tests that prove nothing leaked.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository,
and one scored submission on the session's challenge (Part F).

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

The session's challenge is **DPE Energy Label** (id `191`): predict whether a
French dwelling's energy label is **E, F or G**, from what the diagnostician
recorded during the visit. 69,854 training and 30,146 test dwellings from
ADEME's open data, 99 columns exactly as published: French labels, codes stored
as numbers, empty cells that mean *not applicable*, a construction year of 1300.

**The model is fixed, so this is Parts A–E graded on real data.** The scorer
always fits scikit-learn's default `LogisticRegression()` on the numbers you
send and ranks the test rows by ROC AUC. You do not submit predictions: you
submit the **matrix your pipeline produces**, and every point of AUC comes from
the preprocessing. Fit `build_pipeline()` on the train rows, `transform` train
and test with the same fitted object, and write `submission.csv.gz`:

- an `id` column and **1 to 300** numeric, finite feature columns;
- **every** test id, and the train ids — all of them, or any subset of at least
  **20,000** (dropping rows you do not trust is a preprocessing decision too);
- no target column: the scorer has its own labels.

```bash
uv pip install mlarena-sdk scikit-learn==1.8.0   # the package is mlarena-sdk; it imports as mlarena
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # Profile -> API Keys
client.download_dataset(191, dest_dir="data/raw")  # train, test, EXPERTISE.md, DICTIONNAIRE.md
submission.to_csv("submission.csv.gz", index=False) # the file must have exactly this name
client.submit(challenge_id=191, files=["submission.csv.gz"])
```

Read **`EXPERTISE.md`** before you choose an imputation or an encoding. The DPE
is a regulated calculation, and it says why a missing construction year is
recoverable from its period, why an insulation quality is an order, and why a
département number is not a quantity. The starter notebook linked from the
challenge page does the download-to-submit plumbing with the most naive
features; everything in between is your pipeline.

**The numbers.** Higher is better; 0.500 is chance. Same `LogisticRegression()`,
same split, only the features change:

| Features | ROC AUC |
|---|---|
| numeric columns as read, empty cells set to 0 | 0.641 |
| **benchmark — the same columns, median-imputed and standardised** | **0.761** |
| + domain cleaning: construction years, implausible values, structural gaps | 0.821 |
| + codes one-hot encoded as categories | 0.888 |
| + insulation qualities as an order | 0.924 |
| + features built from the regulation | 0.927 |

**The bar is ROC AUC ≥ 0.85**, above the benchmark on purpose: a median imputer
and a scaler do not reach it; the documented domain steps do. With 30,146 test
rows the sampling noise is about ±0.003, so a gap of 0.01 is real.

Two warnings can come back with your score. **"lbfgs did not converge"** means
columns on very different scales — the scaler in your numeric branch. **"train
AUC is … above test AUC"** means a feature carries the target on the train rows:
the naive target encoder of Part C, caught by the scorer. A rejected file (a
NaN, a missing test id, a text column) names the first offender and still uses
one of your submissions of the day, so run the starter's checks first.

Challenge 176 (allergy IgE profiles) stays attached to this module as extra
practice on the same skills.

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
| A scored submission on challenge 191, ROC AUC ≥ 0.85 | 10% |

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
- [ ] Part F: my `submission.csv.gz`, built by my fitted pipeline, is on the leaderboard of DPE Energy Label (#191)
- [ ] Part F: my score reaches the bar: **ROC AUC ≥ 0.85**

If the last two are not ticked you have not finished the lab, however good the
code is. Submissions to #191 are scored within minutes, so there is no excuse for
leaving the last box empty at the end of the session.
