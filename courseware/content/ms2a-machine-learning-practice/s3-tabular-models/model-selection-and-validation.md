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

Ten training points, a degree-9 polynomial. Training error **0.0000**; on the
test points, which continue the series to the right, the error is
**9,209,639**. Between the points the curve already swings; beyond them it
explodes.

---

## Interpolation is not learning

> **Theorem.** For $n$ data points with distinct $x$ values, a polynomial of
> degree $d = n - 1$ fits all of them exactly.

A zero training error is therefore guaranteed, not earned. Any model family rich
enough to interpolate your training set will do so if you let it, and it will be
worthless.

A fully grown decision tree, a 1-nearest-neighbour classifier and a boosted
model with no early stopping are all in that family. Training error tells you
nothing about them; only held-out error does.

---

## Underfitting

![A straight line through data on two levels](assets/tabular/underfitting_illustration.png)

The opposite failure: the model is too simple for the pattern. A straight line
through data that sits on two levels scores **0.2502** on training and
**0.2507** on test.

The gap between them is zero and the model is useless. A small train–test gap
is not the same as a good model: read both numbers. High and equal means too
little capacity or too few useful features; low and far apart means too much.

---

## The split

![Train, validation and test](assets/tabular/train-val-test-split.png)

The test set is separated **first** and touched **once**. Everything else —
training, hyperparameter search, model comparison — happens inside the rest.

---

## The split in two calls

```python
from sklearn.model_selection import train_test_split

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)   # test set first
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=0)
# 60 / 20 / 20; dev = train + validation
```

Carve the test set off the full data, then the validation set off what is
left: a quarter of 80% is 20%. `stratify` keeps the class balance the same in
all three.

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

Selecting on the validation set consumes it: the best of many noisy scores is
biased upward (Leak 3 below; *Hyperparameter Optimisation* gives it a slide,
"The multiple-comparisons trap"). That is why the test set exists, and why
touching it twice destroys it. Split it off before you look at anything.

---

## Each set in code

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

## Model selection

![Fit every candidate on train, score it on validation, keep the best](assets/tabular/model-selection-flow.png)

Train each candidate on the training set, score each on held-out data, keep the
best. The loop is the same whether the candidates differ by hyperparameter or by
model family — which is why the next lesson treats the family as one more
hyperparameter.

The test set is not in this picture. It enters once, after the loop is closed.

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

---

## Stratify every classification problem

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
```

Stratified k-fold preserves the class proportions in every fold. Use it for
every classification problem — there is no case where plain k-fold is preferable
— and `StratifiedKFold` on binned values for a skewed regression target.

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

## Group k-fold

If rows are not independent, random folds put related rows on both sides of the
split and the score measures memorisation.

```python
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, groups=df["patient_id"], cv=cv)
```

Ask what the *unit of generalisation* is. Deploying on new patients means folds
split on patients; new shops, new users, new sessions — same answer. It is the
"what is one row?" question from Session 1 asked at split time, and the answer is
often not the row.

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

## Nested cross-validation

![Nested cross-validation](assets/tabular/d-nested-cv.png)

Tuning inside the same folds you report gives an optimistic score, because the
hyperparameters were chosen with knowledge of those folds.

Nested CV runs a full search in an inner loop on each outer training set
(`GridSearchCV` here; *Hyperparameter Optimisation* covers it), then scores the
winner on the untouched outer fold.

```python
inner = StratifiedKFold(3, shuffle=True, random_state=0)
outer = StratifiedKFold(5, shuffle=True, random_state=0)
search = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc")
scores = cross_val_score(search, X, y, cv=outer, n_jobs=-1)
```

It costs `outer × (|grid| × inner + 1)` fits, the `+ 1` being each outer
fold's refit of its winner: 50 here for three candidate values, five times the
plain search's ten. It answers "how well does *this procedure* generalise", not
"which hyperparameters should I ship". You still refit the search on all the
data to get the model.

---

## Data leakage

**Any information in your training data that will not be available at
prediction time.**

The signature: a validation score that is much better than you expected, and a
production score that is much worse.

Every leak below passes every unit test and produces a number that looks like a
result. The only defence is to know the catalogue.

---

## Leak 1 — preprocessing before the split

```python
# WRONG: the scaler has seen the test set's distribution
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled, random_state=0)
```

```python
# RIGHT: fit on train, apply to both
X_train, X_test = train_test_split(X, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler's mean and standard deviation were computed over the test rows, so
the model trains with knowledge of the test distribution. The effect is small
for a scaler; it is enormous for a target encoder, whose statistics do encode
the answer.

---

## Pipelines make this structural

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
cross_val_score(pipe, X, y, cv=cv)             # right
```

> Every fitted transformation belongs inside a `Pipeline` that is passed to the
> cross-validator, so that it is refitted from scratch on each training fold.

The leak becomes impossible rather than merely discouraged — the fail-fast
principle applied to methodology. The same holds for a `ColumnTransformer` in
front of an `LGBMClassifier`. Session 2 builds the pipeline; this is the reason
it exists.

---

## Leak 2 — a feature that encodes the answer

A `discharge_date` column when predicting whether a patient was admitted. A
`refund_issued` flag when predicting fraud. `days_since_last_payment` when the
missed payment is what you predict.

The symptom: one feature dominates and the AUC sits near 1.0. These features
are worthless, because at prediction time they do not exist yet.

**Test:** for every feature, ask *would I have this value at the moment I need
the prediction?* If no, drop it. The column is legitimately in the table, which
is what makes this the hard one: build a timeline of what is known when, and
keep it in the repo.

---

## Leak 3 — using the test set to choose

Trying twenty models and reporting the best test score is leakage through you.
The reported number is the maximum of twenty noisy draws, not an estimate of
performance.

The same happens slowly: look at the test score, change a feature, look again.
Each look moves a little information from the test set into your decisions.

Choose on validation. Touch test once.

---

## Leak 4 — the rest of the catalogue

| Leak | Symptom | Fix |
|---|---|---|
| Duplicate rows across folds | CV much better than the leaderboard | deduplicate before splitting |
| Group leakage | good CV, useless in production | `StratifiedGroupKFold` on the entity |
| Selecting features on all the data | small, persistent optimism | select inside the pipeline |
| Test set used for early stopping | optimism proportional to rounds | eval set from the training folds |
| Shuffled time series | great CV, a forecast that fails | `TimeSeriesSplit`, sorted |

Each row is Leak 1, 2 or 3 in disguise. The question is always the same: did
anything about the held-out rows reach the model, or you, before the score was
computed?

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

---

## Local CV and the leaderboard

An ML-Arena challenge scores your submission on data you never see. On a file
challenge, where you upload predictions, that is a hidden held-out split,
scored once. On *European Rain Forecast*, this session's challenge, there is no
split at all: every 6 h your agent is called on the last 48 h of weather, each
forecast is scored 48 h later, once its target hours have happened, and the
leaderboard is the mean of your run scores so far.

- Trust your **local CV** for decisions. A hidden split is one draw and noisier
  than it looks; a live board is a growing average over one stretch of weather.
  The replay in Lab 3 is your CV.
- Track the *gap* between them. A stable offset means the distributions differ;
  a gap that grows as you tune means you are fitting the board.
- Resubmitting until a number looks good is selection on noise (Leak 3). On a
  hidden split the leaderboard becomes a validation set; on a live board you
  are keeping the agent that had the luckiest weeks.

---

## What the board shows

For challenge 177: the ranking column *CRPS skill*, a mean over runs, with a
30-day mean beside it; then the skill and the CRPS in mm at +6 h and +48 h, the
sample count, the slowest predict call and the number of runs.

`RewardCi95`, the interval the platform draws next to the mean, is 1.96 s/√n
over the episodes of a submission's runs. It is filled in on RL challenges
whose env reports an episode-to-episode spread; this env reports none, so the
board shows ±0.0000 — a blank, not a confirmation. The lab's weekly-block
standard error, about 0.005 per row, is the ruler: a gap under 0.01 between
two rows is weather, not skill. Treat such ranks as tied.

Decide locally, submit rarely, and report the number you got the first time you
were finished.

---

## Check yourself

1. Your fraud model reaches a cross-validated AUC of 0.99, and one feature,
   `refund_issued`, carries almost all of it. What is wrong?

   **Answer.** Leak 2: the refund is issued *after* the fraud is found, so the
   flag does not exist when the prediction is needed. Ask of every feature
   whether you would have it at prediction time, and drop the ones you would
   not.

2. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.model_selection import StratifiedKFold, KFold
   y = np.repeat([0, 1], [970, 30])          # 3% positive
   s = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
   k = KFold(n_splits=5, shuffle=True, random_state=0)
   print([int(y[te].sum()) for _, te in s.split(y, y)])
   # -> [6, 6, 6, 6, 6]
   print([int(y[te].sum()) for _, te in k.split(y)])
   # -> [10, 5, 4, 8, 3]
   ```

   The second line is why plain k-fold has no place in a classification
   problem: those folds differ more than most models do.

3. You will deploy on patients the model has never seen, and each patient
   contributes several rows. Which splitter, keyed on what?

   **Answer.** `StratifiedGroupKFold` with `groups=df["patient_id"]`. The unit
   of generalisation is the patient, not the row; random folds put the same
   patient on both sides and measure memorisation.
