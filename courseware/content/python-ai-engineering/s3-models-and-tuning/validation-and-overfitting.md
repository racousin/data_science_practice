# Validation & Overfitting

Is the model learn and be able to generalize ? How to get a number you can trust ?

---

## Overfitting

The model has learned the training set, including its noise.

![Overfitting](assets/s3-models-and-tuning/validation-and-overfitting/overfitting_illustration.png)

Symptom: training error keeps falling, validation error starts rising.


> **Theorem.** For $n$ data points with distinct $x$ values, a polynomial of
> degree $d = n - 1$ fits all of them exactly.

Training error **0.0000**, test error **9,209,639**. Any model family rich enough
to interpolate your training set will do so if you let it, and it will be
worthless.

---




## The split

![Train / validation / test](assets/s3-models-and-tuning/validation-and-overfitting/train-val-test-split.png)

The test set is separated **first** and touched **once**. Everything else —
training, hyperparameter search, model comparison — happens inside the other
80%.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

---

## What each set is for

| Set | Share | Used for |
|---|---|---|
| Train | 70% | fitting parameters |
| Validation | 15% | choosing hyperparameters, early stopping |
| Test | 15% | one final, honest estimate |

```python
model1.fit(X_train, y_train)                    # train: fit
model2.fit(X_train, y_train)                   # train: fit
f1_score(y_val,  model1.predict(X_val))        # validation: compare candidates
f1_score(y_val,  model2.predict(X_val))

f1_score(y_test, best_model.predict(X_test))        # test: report, once
```

The validation score is not the number you report — you picked the model with it,
so it is optimistic. Only the test score is honest.

---

## K-fold cross-validation

![kfold.png](assets/s3-models-and-tuning/validation-and-overfitting/kfold.png)

With little data, one split wastes most of it and the estimate is noisy.
K-fold uses everything.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, test_idx in cv.split(X):
    m = clone(model)                       # a fresh, unfitted model each fold
    m.fit(X[train_idx], y[train_idx])
    scores.append(f1_score(y[test_idx], m.predict(X[test_idx])))

print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Stratified K-fold

![stratifiedkfold.png](assets/s3-models-and-tuning/validation-and-overfitting/stratifiedkfold.png)

With imbalanced classes, a random fold might contain no positives at all.
Stratification preserves the class ratio in every fold.

`cross_val_score` runs exactly the loop above for you — pass it the splitter and
get the array of fold scores back:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Time series — do not shuffle

![tskfold.png](assets/s3-models-and-tuning/validation-and-overfitting/tskfold.png)

With temporal data, a random split trains on the future and tests on the past.
The score is meaningless.

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
```

Each fold trains on a prefix of the series and tests on the block that follows,
so the training set grows at every split. Rows are assumed to be in chronological
order — `TimeSeriesSplit` slices positionally and never looks at a date column.

---

## Data leakage

**Any information in your training data that will not be available at
prediction time.**

The signature: a validation score that is much better than you expected, and a
production score that is much worse.

---

## Leak 1 — preprocessing before the split

```python
# WRONG — the scaler has seen the test set's distribution
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled, ...)
```

```python
# RIGHT — fit on train, apply to test
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Pipelines make this structural

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
cross_val_score(pipe, X, y, cv=5)
```

Inside `cross_val_score`, the scaler is refitted on each training fold
automatically. The leak becomes impossible rather than merely discouraged —
which is the fail-fast principle applied to methodology.

---

## Leak 2 — a feature that encodes the answer

A `discharge_date` column when predicting whether a patient was admitted. A
`refund_issued` flag when predicting fraud.

These give near-perfect validation scores and are worthless, because at
prediction time they do not exist yet.

**Test:** for every feature, ask *would I have this value at the moment I need
the prediction?* If no, drop it.

---


## Leak 3 — using the test set to choose

Trying twenty models and reporting the best test score is leakage through you.
The reported number is the maximum of twenty noisy draws, not an estimate of
performance.

Choose on validation. Touch test once.

---
## Fighting overfitting

- **More data** — the most reliable fix, and usually the least available
- **Regularisation** — L1/L2 penalties, dropout, weight decay
- **Simpler model** — fewer parameters, shallower trees
- **Early stopping** — stop when validation error turns up
- **Data augmentation** — more effective variety from the same data
