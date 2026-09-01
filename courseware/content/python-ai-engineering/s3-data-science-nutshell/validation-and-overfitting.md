# Validation & Overfitting

How to get a number you can trust — and the ways students routinely get one
they cannot.

<!-- notes: 35 minutes. Leakage is the highest-value 10 minutes in the whole
session. Every year several project teams lose their score to it. -->

---

## Overfitting

The model has learned the training set, including its noise.

![Overfitting](assets/ds/overfitting_illustration.png)

Symptom: training error keeps falling, validation error starts rising.

| | Train | Validation |
|---|---|---|
| Good fit | 0.15 | 0.18 |
| Overfit | 0.01 | 0.42 |

---

## Underfitting

The model is too simple to represent the pattern at all.

![Underfitting](assets/ds/underfitting_illustration.png)

Symptom: both errors are high, and they are close together.

| | Train | Validation |
|---|---|---|
| Underfit | 0.38 | 0.40 |

---

## Diagnosing from two numbers

```text
train low,  val low   -> good
train low,  val high  -> overfitting  -> simplify, regularise, get more data
train high, val high  -> underfitting -> more capacity, better features
train high, val low   -> a bug, or a leak. Investigate.
```

The fourth row should never happen. When it does, something is wrong with the
split.

---

## Fighting overfitting

- **More data** — the most reliable fix, and usually the least available
- **Regularisation** — L1/L2 penalties, dropout, weight decay
- **Simpler model** — fewer parameters, shallower trees
- **Early stopping** — stop when validation error turns up
- **Data augmentation** — more effective variety from the same data

---

## The split

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

| Set | Share | Used for |
|---|---|---|
| Train | 70% | fitting parameters |
| Validation | 15% | choosing hyperparameters, early stopping |
| Test | 15% | one final, honest estimate |

---

## Why three sets

Every time you look at the validation set and change something, you leak a
little information into your choices. After fifty experiments, validation
performance is optimistic too.

The test set is touched **once**, at the end. If you tune against it, you no
longer have a test set — you have a second validation set and no honest number.

<!-- notes: Frame this as what the ML-Arena hidden test split is for. They will
meet it as a rule; better they meet it as a reason. -->

---

## K-fold cross-validation

With little data, one split wastes most of it and the estimate is noisy.
K-fold uses everything:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

Split into 5 folds; train on 4, validate on 1; rotate. Report the mean **and the
spread** — a mean of 0.80 ± 0.02 and 0.80 ± 0.15 are very different results.

---

## Stratified K-fold

With imbalanced classes, a random fold might contain no positives at all.
Stratification preserves the class ratio in every fold.

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
```

For classification, this should be your default rather than a special case.

---

## Time series — do not shuffle

With temporal data, a random split trains on the future and tests on the past.
The score is meaningless.

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
```

Each fold trains on everything before a cut point and validates on what comes
after — which is the situation you will actually face.

```text
fold 1: train [....]              val [..]
fold 2: train [......]            val [..]
fold 3: train [........]          val [..]
```

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

## Leak 3 — duplicates across the split

The same row, or a near-duplicate, in both train and test. The model recalls
rather than generalises.

Common with: augmented images, multiple records per patient, near-identical
text. Deduplicate — and split by **group** (patient, user, document) rather than
by row:

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=patient_ids, cv=cv)
```

---

## Leak 4 — using the test set to choose

Trying twenty models and reporting the best test score is leakage through you.
The reported number is the maximum of twenty noisy draws, not an estimate of
performance.

Choose on validation. Touch test once.

---

## Checklist

- [ ] Split before any preprocessing
- [ ] All transformations inside a `Pipeline`
- [ ] Stratified folds for classification
- [ ] Time-ordered folds for temporal data
- [ ] Grouped folds when rows share an entity
- [ ] Every feature available at prediction time
- [ ] No duplicates across the split
- [ ] Test set used exactly once

If a validation score surprises you on the upside, assume a leak until you have
found the reason it is real.
