# Lab 3 — Beat the Baseline

Take the pipeline you built in Lab 2, put a tuned gradient-boosting model behind
it, and prove — with a protocol you can defend — that it beats a linear
baseline.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.

<!-- notes: 45 minutes, tight. Insist they write the protocol down in Part A
before touching a model; the ones who skip it spend Part D discovering their CV
is leaking. Circulate during Part D — the usual bug is `eval_set` pointing at the
test split. Reserve the last five minutes for reading numbers out loud. -->

---

## Setup

Work in the project repository, on a branch, on Lab 2's output.

```text
src/models/
    __init__.py
    protocol.py     <- the splitter, the metric, the seed. One place.
    baseline.py     <- linear / logistic on the Lab 2 pipeline
    boosted.py      <- the gradient-boosting estimator
    tune.py         <- the Optuna study
tests/test_models.py
reports/lab3.md     <- the numbers, written down
```

The Lab 2 preprocessing is imported, not copied. If it is not yet a
`ColumnTransformer` inside a `Pipeline`, that is the first commit.

---

## Part A — Freeze the protocol (7 min)

Before any model, write `protocol.py` and do not touch it again:

```python
SEED = 0
METRIC = "roc_auc"            # or neg_root_mean_squared_error
def splitter():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

Then hold out a test set once, and record in `reports/lab3.md`:

- what one row is, and what the **unit of generalisation** is
- which splitter that implies — `StratifiedKFold`, `StratifiedGroupKFold` on
  which column, or `TimeSeriesSplit` with what `gap`
- the metric, and one sentence on why it fits the problem

If your data is ordered in time or has repeated entities, the plain stratified
splitter is wrong, and choosing it anyway is the mistake this lab is about.

---

## Part B — The baseline (8 min)

```python
base = Pipeline([("prep", preprocessor),
                 ("model", LogisticRegressionCV(Cs=10, max_iter=2000))])
scores = cross_val_score(base, X_tr, y_tr, cv=splitter(), scoring=METRIC)
```

Regularised linear or logistic, on the same pipeline, folds and metric. Record
mean and standard deviation.

This number is the bar. A boosted model that does not clear it by more than one
fold standard deviation has not earned its place, and saying so in the report is
worth more marks than a fabricated improvement.

---

## Part C — Gradient boosting (10 min)

```python
boosted = Pipeline([("prep", preprocessor),
                    ("model", LGBMClassifier(n_estimators=2000,
                                             learning_rate=0.05, random_state=SEED))])
```

Score it on the same folds, then add early stopping with the eval set taken
**from inside the training fold** — never the outer validation fold, never the
test set.

Record baseline mean ± std, untuned boosting mean ± std, and the gap. If the gap
is smaller than the standard deviation, say so.

---

## Part D — The Optuna study (12 min)

One study, one fixed budget, declared before it runs:

```python
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=40, timeout=600)
```

Requirements:

- the objective returns the cross-validated score over `splitter()`, nothing else
- `learning_rate` and `reg_lambda` sampled with `log=True`
- four to six parameters, no more; `n_estimators` is not one of them
- `n_trials` fixed in advance and stated in the report — not "until it stopped
  improving"
- the study persisted (`storage=`, or `study.trials_dataframe().to_csv(...)`)

Report the best trial's score alongside the *median* of the top five. A large gap
between them means you found a lucky fold, not a better model.

---

## Part E — One number (8 min)

Refit the winning configuration on all of `X_tr`, predict the held-out test set
and compute the metric **once**.

```python
best = {f"model__{k}": v for k, v in study.best_params.items()}
final = clone(boosted).set_params(**best).fit(X_tr, y_tr)
test_score = scorer(final, X_te, y_te)     # called exactly once
```

`reports/lab3.md` states, in a table: baseline CV, boosted CV, tuned CV and the
single test number, plus two lines on the gap between the last two. A test score
far below the tuned CV means the search overfitted the folds — a finding.

Report the top five features by **permutation importance on held-out data**, and
one sentence on any that surprises you.

---

## Required tests

```python
def test_preprocessing_is_fitted_inside_each_fold():
    """A transformer spy records the fit row count. Cross-validating the
    pipeline over 5 folds must fit it 5 times on 4/5 of the rows each,
    never once on all of them."""

def test_baseline_and_model_use_the_same_splits():
    """protocol.splitter() yields identical fold indices on two calls."""

def test_no_index_appears_in_both_sides_of_a_fold():
    """For every fold, train and validation index sets are disjoint."""

def test_study_is_reproducible():
    """Two studies with the same seed and 5 trials give the same best_params."""
```

The first is the one that matters. Write it against a transformer that records
`len(X)` at `fit` time, and check that it fails when you scale before splitting.

---

## Pull request

The description states:

- the splitter chosen and the unit of generalisation that forced it
- baseline, untuned and tuned CV scores with their fold standard deviations
- the search budget, fixed in advance, and what the best trial found
- the single test number, and whether it agrees with the tuned CV score
- one thing you tried that did not work

---

## Grading and deductions

| Criterion | Weight |
|---|---|
| Protocol written first, splitter justified by the data | 20% |
| Baseline fitted on the same folds and metric | 15% |
| Boosting with early stopping on an in-fold eval set | 15% |
| Optuna study: fixed budget, log-scale spaces, seeded, persisted | 20% |
| Test set scored exactly once, reported honestly | 15% |
| Four tests passing, including the fold-fitting test | 15% |

Automatic deductions:

- a transformer fitted outside the cross-validation folds
- the test set used as `eval_set`, or scored more than once
- `n_estimators` tuned in the search
- a reported "best score" that is the maximum over trials
- a bare `except` around a trial, or a silent fallback for a missing parameter

---

## Carry it forward

You now have a validated tabular model, a protocol file, and a number you can
defend. The protocol is what transfers: every later lab reuses the same three
questions — what is one row, what is the unit of generalisation, what is the
metric.

If your project takes the prediction track, this model is the submission you have
to beat with everything you learn afterwards. Most of you will not beat it by
much.
