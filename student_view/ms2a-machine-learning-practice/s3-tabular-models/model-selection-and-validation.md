# Model Selection and Validation

A score is a claim about data you have not seen. Everything in this lesson is
machinery for making that claim honestly, and a catalogue of the ways it turns
out to be a lie.

<!-- notes: 35 minutes, the lesson that decides whether their project is worth
anything. Do the leakage catalogue as a quiz: show each snippet, ask the room
what is wrong, then explain. Budget 12 minutes for leakage alone — it is the
part they will actually get wrong. -->

---

## Three sets, three jobs

| Set | Used for | Seen how often |
|---|---|---|
| Train | fitting parameters | every fit |
| Validation | choosing hyperparameters, early stopping | many times |
| Test | one final estimate | **once** |

The validation set is consumed by the act of selecting on it: after fifty Optuna
trials, your best validation score is the maximum of fifty noisy numbers and is
biased upward by construction.

That is why the test set exists, and why touching it twice destroys it. Split it
off before you look at anything.

---

## k-fold cross-validation

![k-fold cross-validation](/api/academic_courses/assets/lessons/74/kfold-cross-validation.png)

Split the training portion into $k$ parts; fit $k$ times, each time holding out
one part. Every row is validated exactly once.

```python
from sklearn.model_selection import cross_val_score, KFold
cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
```

Five folds is the default worth defending; ten costs twice as much for a
marginally lower-variance estimate. `shuffle=True` is not optional — data sorted
by date or by class turns unshuffled folds into a different experiment.

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

![Stratified folds preserve the class balance](/api/academic_courses/assets/lessons/74/stratified.png)

Plain k-fold on a 3%-positive target produces a fold with 1.2% positives and one
with 4.6% by chance alone, so scores vary because the folds do, not the model.

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
```

Stratified k-fold preserves the class proportions in every fold. Use it for
every classification problem — there is no case where plain k-fold is preferable
— and `StratifiedKFold` on binned values for a skewed regression target.

---

## Leave-one-out, and repeating the split

![Leave-one-out cross-validation](/api/academic_courses/assets/lessons/74/loofold.png)

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

## Nested cross-validation

![Nested cross-validation](/api/academic_courses/assets/lessons/74/d-nested-cv.png)

Tuning inside the same folds you report gives an optimistic score, because the
hyperparameters were chosen with knowledge of those folds.

Nested CV runs a full search in an inner loop on each outer training set, then
scores the winner on the untouched outer fold.

```python
inner = StratifiedKFold(3, shuffle=True, random_state=0)
outer = StratifiedKFold(5, shuffle=True, random_state=0)
search = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc")
scores = cross_val_score(search, X, y, cv=outer, n_jobs=-1)
```

It costs `inner × outer` fits and answers "how well does *this procedure*
generalise", not "which hyperparameters should I ship". You still refit the
search on all the data to get the model.

---

## Leakage 1 — preprocessing before the split

```python
X = StandardScaler().fit_transform(X)        # wrong
X_tr, X_te = train_test_split(X)
```

The scaler's mean and standard deviation were computed over the test rows, so
every fold trains on statistics that encode the answer. The effect is small for a
scaler and enormous for a target encoder.

> Every fitted transformation belongs inside a `Pipeline` that is passed to the
> cross-validator, so that it is refitted from scratch on each training fold.

```python
pipe = Pipeline([("prep", preprocessor), ("model", LGBMClassifier())])
cross_val_score(pipe, X, y, cv=cv)           # right
```

Session 2 builds the pipeline. This is the reason it exists.

---

## Leakage 2 — the rest of the catalogue

| Leak | Symptom | Fix |
|---|---|---|
| Duplicate rows across folds | CV much better than the leaderboard | deduplicate before splitting |
| Target leakage from a future column | one feature dominates, AUC near 1.0 | ask when each column is *known* |
| Group leakage | good CV, useless in production | `GroupKFold` on the entity |
| Selecting features on all the data | small, persistent optimism | select inside the pipeline |
| Test set used for early stopping | optimism proportional to rounds | eval set from the training folds |

Target leakage is the hard one, because the column is legitimately in the table:
`days_since_last_payment` does not exist at scoring time if the payment is what
you predict. Build a timeline of what is known when, and keep it in the repo.

---

## Local CV and the leaderboard

An ML-Arena competition scores your submission on a held-out split you never see
— the test set, run by someone else, which is what makes it worth having.

- Trust your **local CV** for decisions. It averages several folds; the
  leaderboard is one split and noisier than it looks.
- Track the *gap* between them. A stable offset means the splits differ in
  distribution; a gap that grows as you tune means you are overfitting the
  leaderboard by resubmitting.
- Submitting fifty times and keeping the best is selection on the test set.
  The leaderboard becomes a validation set and stops being an honest estimate.

Decide locally, submit rarely, and report the number you got the first time you
were finished.
