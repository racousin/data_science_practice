# Model Selection and Validation

A score is a claim about data you have not seen. Everything in this lesson is
machinery for making that claim honestly, and a catalogue of the ways it turns
out to be a lie.

<!-- notes: 50 minutes, the lesson that decides whether their project is worth
anything. Ten minutes on over- and underfitting and the three-way split, fifteen
on the cross-validation schemes, then the leakage catalogue as a quiz: show each
snippet, ask the room what is wrong, then explain. Budget 15 minutes for
leakage alone — it is the part they will actually get wrong. -->

---

## Overfitting

![A degree-9 polynomial through ten training points](assets/tabular/overfit-polynomial.png)

The model has learned the training set, including its noise. The symptom:
training error keeps falling while validation error starts rising.

> **Theorem.** For $n$ data points with distinct $x$ values, a polynomial of
> degree $d = n - 1$ fits all of them exactly.

---

## The split

![Train, validation and test](assets/tabular/train-val-test-split.png)

The test set is separated **first** and touched **once**. Everything else —
training, hyperparameter search, model comparison — happens inside the rest.


---

## Three sets, three jobs

| Set | Typical share | Used for | Seen how often |
|---|---|---|---|
| Train | 60–70% | fitting parameters | every fit |
| Validation | 15–20% | choosing hyperparameters and models | many times |
| Test | 15–20% | one final, honest estimate | **once** |

Early stopping takes its own slice of the training part — `X_es` in *Gradient
Boosting in Practice* — so once a set has picked the number of trees, its score
is a training-side number.

```python
from sklearn.model_selection import train_test_split

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)   # test set first
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=0)
# 60 / 20 / 20; dev = train + validation
```

---

##  Model selection

```python
from sklearn.metrics import f1_score

model1.fit(X_train, y_train)                   # train: fit
model2.fit(X_train, y_train)
f1_score(y_val, model1.predict(X_val))         # validation: compare
f1_score(y_val, model2.predict(X_val))

f1_score(y_test, best_model.predict(X_test))   # test: report, once
```

Two candidates compared on validation, the winner reported on test — once.

---


## k-fold cross-validation

![Five folds inside the training data, the test set held out](assets/tabular/kfold.png)

One validation split wastes data and gives a noisy estimate. Split the training
portion into $k$ parts; fit $k$ times, each time holding out one part. Every row
is validated exactly once, and the test set stays outside.

```python
from sklearn.model_selection import cross_val_score, KFold
cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
```

Five folds is the default worth defending; ten costs twice as much for a
marginally lower-variance estimate. For independent rows `shuffle=True` is not
optional — a file sorted by class turns unshuffled folds into a different
experiment. Rows that are not independent (groups) or that are ordered in time
get their own splitters, a few slides on.

---

## The loop inside `cross_val_score`

```python
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, val_idx in cv.split(X):
    m = clone(model)                   # fresh and unfitted, every fold
    m.fit(X[train_idx], y[train_idx])
    scores.append(f1_score(y[val_idx], m.predict(X[val_idx])))

scores = np.array(scores)              # a list has no .mean()
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

`cross_val_score(model, X, y, cv=cv, scoring="f1")` returns exactly these five
numbers. `clone` is the important line: refitting one shared object would work
here and silently break the day a step keeps state between fits.

---

## Read the spread, not just the mean

```python
print(f"{scores.mean():.4f} +/- {scores.std():.4f}")
```

A model at 0.812 ± 0.004 and one at 0.818 ± 0.030 are not ranked by their means.
The second is one unlucky fold away from being worse, and its fold-to-fold
variance is telling you the model is unstable on this dataset.

Compare models on the **same folds** — instantiate the splitter once with a fixed
`random_state` and pass it everywhere. Two means over two different random
partitions differ for reasons that have nothing to do with the models.

If two models differ by less than one fold standard deviation, you have not shown
that one is better. Prefer the simpler one.

---

## Stratified k-fold

![Stratified folds preserve the class balance](assets/tabular/stratified.png)

Plain k-fold on a 3%-positive target produces a fold with 1.5% positives and
one with 5% by chance alone — the snippet in *Check yourself* — so scores vary
because the folds do, not the model.


```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
```


---

## Leave-one-out, and repeating the split

![Leave-one-out cross-validation](assets/tabular/loofold.png)

The extreme case: $k = n$. Each fit sees $n - 1$ rows, so the bias of the
estimate is minimal, and you pay $n$ fits for it.

The variance is not minimal, which surprises people: the $n$ training sets are
nearly identical, so the $n$ errors are correlated and averaging removes less
noise than five independent folds do.

Reach for it under a couple of hundred rows with a model that fits in
milliseconds. On small data prefer `RepeatedStratifiedKFold(n_splits=5,
n_repeats=5)`, which re-partitions five times and averages twenty-five folds —
it shrinks the noise from *which partition you drew*, though it fixes no bias.

---


## Time series: do not shuffle

![Expanding-window folds: train on the past, validate on what follows](assets/tabular/tskfold.png)

With temporal data, a random split trains on the future and tests on the past.
The score is meaningless.

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
```

`TimeSeriesSplit` trains on a prefix of the series and validates on the block
that follows, so the training set grows at every split. *Time Series Models*
adds the `gap` and `test_size` a real forecast needs, and the sort-first rule.

---



## Fighting overfitting

- **More data** — the most reliable fix, and usually the least available
- **Regularisation** — L1/L2 penalties, `reg_lambda`, dropout, weight decay
- **Simpler model** — fewer parameters, shallower trees, smaller
  `num_leaves`, larger `min_child_samples`
- **Early stopping** — stop when validation error turns up
- **Data augmentation** — more effective variety from the same data; common
  for images and text, rare for tables

Underfitting goes the other way: more capacity, better features, less
regularisation. Either way, the validation curve — not the training curve — is
what tells you which way to move.
