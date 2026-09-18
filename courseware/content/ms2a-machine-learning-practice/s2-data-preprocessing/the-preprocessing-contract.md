# The Preprocessing Contract

A model does not learn from a table. It learns from a matrix of numbers, and
every step between the two has parameters that were fitted on something. Which
rows you fit them on is the entire subject of this session.

<!-- notes: 20 minutes. Ask who has ever scaled before splitting — most hands go
up. Do not moralise: show the number the leak produces, then show the Pipeline
that makes the mistake impossible to write. -->

---

## What preprocessing has to produce

Session data collection ended with a table you can defend. A model needs something narrower:

- every cell numeric — no strings, no dates, no lists
- no missing values, (unless the model declares that it handles them)
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

## The same scaler, fitted twice

![Fit before the split versus fit on train](assets/preprocessing/fit-on-train-vs-leak.png)

Ten rows `0 … 9`, the last two held out. Fitted before the split, the scaler's
mean is 4.5; fitted on train, it is 3.5.

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

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale",  StandardScaler())])

cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("ohe",    OneHotEncoder(handle_unknown="ignore"))])

pre = ColumnTransformer([
    ("num", num_pipe, ["age", "income"]),
    ("cat", cat_pipe, ["city", "plan"]),
], remainder="drop")

pipe = Pipeline([("pre", pre), ("model", Ridge())])
pipe.fit(X_tr, y_tr)
```

`remainder="drop"` is the fail-fast setting: a column you did not name never
silently reaches the model. `remainder="passthrough"` is how a raw string column
ends up crashing the estimator three steps later — or worse, not crashing.

---

## Cross-validationinside a Pipeline

```python
from sklearn.model_selection import cross_val_score

cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
```

Passing the *pipeline* re-fits every transformer on each training fold. Passing
an already-transformed `X` leaks the fold you are about to score into the
transformer that produced it — the score comes out optimistic and no line of
code looks wrong.
