# The Preprocessing Contract

A model does not learn from a table. It learns from a matrix of numbers, and
every step between the two has parameters that were fitted on something. Which
rows you fit them on is the entire subject of this session.

<!-- notes: 20 minutes. Ask who has ever scaled before splitting — most hands go
up. Do not moralise: show the number the leak produces, then show the Pipeline
that makes the mistake impossible to write. -->

---

## What preprocessing has to produce

Session 1 ended with a table you can defend. A model needs something narrower:

- every cell numeric — no strings, no dates, no lists
- no missing values, unless the model declares that it handles them
- comparable scales, for anything that measures a distance or takes a gradient
- a fixed, ordered set of columns, identical at training and at inference

Preprocessing is the code that turns the first into the second. It is part of
the model, not part of the exploration.

---

## Fit and transform are different operations

Every step holds parameters learned from data:

| Step | What it learns |
|---|---|
| `SimpleImputer` | the median of each column |
| `StandardScaler` | a mean and a standard deviation |
| `OneHotEncoder` | the category vocabulary |
| `TargetEncoder` | a target mean per category |
| `PCA` | the component basis |

`fit` reads data and stores parameters; `transform` applies them. The contract
is one line: **fit on train, transform everywhere.**

---

## The leak

```python
X = scaler.fit_transform(X)                        # wrong
X_tr, X_te = train_test_split(X, test_size=0.2)
```

The scaler saw the test rows. Its mean and standard deviation encode them, so
the test score is measured on data the pipeline already knows.

```python
X_tr, X_te = train_test_split(X, test_size=0.2)
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)                      # transform, never fit
```

---

## What the leak costs

A scaler leaks little. The steps that leak a lot share one property: they touch
the target, or they aggregate information across rows.

| Step | Leak | Why |
|---|---|---|
| `StandardScaler` | small | two moments of one column |
| `SimpleImputer(median)` | small | one statistic |
| `KNNImputer` | moderate | test rows become neighbours |
| `PCA` | moderate | the basis is fitted on everything |
| Target encoding | large | the label enters the feature |
| Feature selection on the full set | large | the choice used test labels |

A cross-validated 0.94 that becomes 0.71 in production is almost always one of
the last two rows.

---

## Pipeline makes the rule structural

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler()),
                 ("model", Ridge())])
pipe.fit(X_tr, y_tr)
```

`pipe.fit` calls `fit_transform` on every step and `fit` on the last.
`pipe.predict` calls `transform` only. There is no code path in which a
transformer is fitted on `X_te`.

That is the difference between a rule you remember and a rule you cannot break
without deleting the object that enforces it.

---

## ColumnTransformer routes columns

Real tables are heterogeneous: numbers need imputing and scaling, categories
need imputing and encoding.

```python
from sklearn.compose import ColumnTransformer

pre = ColumnTransformer([
    ("num", num_pipe, ["age", "income"]),
    ("cat", cat_pipe, ["city", "plan"]),
], remainder="drop")
```

`remainder="drop"` is the fail-fast setting: a column you did not name never
silently reaches the model. `remainder="passthrough"` is how a raw string column
ends up crashing the estimator three steps later — or worse, not crashing.

---

## Cross-validation only means something inside a Pipeline

> **Borrowed from Session 3.** `cross_val_score(pipe, X, y, cv=5)` splits the
> training rows into five parts, fits on four and scores on the fifth, five
> times, and returns the five scores. `scoring="roc_auc"` is the ranking quality
> of a binary classifier: 0.5 is a coin flip, 1.0 is perfect, higher is better.
> You do not need more than that today — Session 3 does it properly. Read the
> number here as *a score that should go down when you stop leaking*.

```python
from sklearn.model_selection import cross_val_score

cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
```

Passing the *pipeline* re-fits every transformer on each training fold. Passing
an already-transformed `X` leaks the fold you are about to score into the
transformer that produced it — the score comes out optimistic and no line of
code looks wrong.

---

## Where it lives in the repository

```text
src/preprocess/
    __init__.py
    pipeline.py     <- build_pipeline() -> an unfitted sklearn Pipeline
    features.py     <- custom transformers
tests/test_pipeline.py
models/pipeline_2026-09-21.joblib
```

`build_pipeline()` returns an unfitted object; training fits and serialises it,
inference loads it and calls `transform`. Nothing is re-derived at inference,
because anything re-derived at inference is a bug waiting for a distribution
shift.

---

## The order of operations

1. **split** — before anything else, and by time if the rows are ordered
2. **clean what needs no fitting** — duplicates, unit fixes, type casts
3. **build the pipeline** — impute, encode, scale, engineer
4. **fit on train**, transform train and test
5. **serialise** the fitted pipeline next to the model

Steps 2 and 3 differ in exactly one respect: step 2 has no learned parameter, so
it cannot leak. Everything with a parameter belongs in step 3.

> Every statistic used to transform a row must come from rows the model was
> allowed to see.

The rest of the session is that sentence applied to missing values, duplicates,
outliers, categories, scales and engineered features — in that order.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler

   X = np.arange(10, dtype=float).reshape(-1, 1)
   X_tr, X_te = train_test_split(X, test_size=0.2, shuffle=False)

   print(StandardScaler().fit(X).mean_)      # -> [4.5]   fitted before the split
   print(StandardScaler().fit(X_tr).mean_)   # -> [3.5]   fitted on train only
   ```

   **Answer.** 4.5 is a number the test rows helped produce; 3.5 is not. That
   difference is the whole leak, on the smallest transformer there is.

2. What does `remainder="drop"` buy you in a `ColumnTransformer`, and what is the
   failure it prevents?

   **Answer.** It is the fail-fast setting: a column you did not name never
   silently reaches the model. With `remainder="passthrough"` a raw string column
   crashes the estimator three steps later — or worse, does not crash.

3. Two rows of the leak-cost table are marked *large*. Which are they, what do
   they have in common, and what symptom do they produce?

   **Answer.** Target encoding and feature selection performed on the full
   dataset. Both use the label. The symptom is a cross-validated 0.94 that
   becomes 0.71 in production.
