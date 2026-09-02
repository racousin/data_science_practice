# Hyperparameter Optimization

Search is the cheapest source of accuracy you have and the easiest place to fool
yourself. This lesson is about spending a fixed budget well and reporting the
result honestly.

<!-- notes: 35 minutes. The Bergstra and Bengio picture — nine grid points
sampling three values of the important parameter, nine random points sampling
nine — is the one thing to draw on the board. Write the Optuna objective live;
they will copy it into the lab twenty minutes later. -->

---

## What is being searched

![Hyperparameter tuning](assets/tabular/tunning.jpg)

Parameters are learned by `fit`. Hyperparameters are fixed before it and control
*how* the fitting happens: learning rate, number of leaves, regularisation.

There is no gradient to follow. The objective is the cross-validated score —
expensive, noisy, non-convex. Every method below is a different way of spending a
fixed number of evaluations on a black box.

---

## Grid search

```python
from sklearn.model_selection import GridSearchCV
grid = {"model__num_leaves": [31, 63, 127], "model__reg_lambda": [0.1, 1, 10]}
search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
```

Every combination, exhaustively: reproducible, trivially parallel, complete
within the grid you wrote.

The cost is the product of the axes times the folds — three values of four
parameters at five folds is 405 fits — and each new parameter multiplies the
bill. It also cannot find a value you did not list.

---

## Why random search wins

Bergstra and Bengio (2012): with a budget of $n$ evaluations, a grid over $d$
parameters tests only $n^{1/d}$ distinct values *per parameter*. Random search
tests $n$ distinct values of every parameter.

In practice two or three hyperparameters dominate. The grid spends most of its
budget re-testing the important one at the same three values while varying ones
that do nothing; random search never repeats a value.

> With more than three hyperparameters, random search dominates grid search at
> equal budget. There is no regime where an exhaustive grid over seven axes is
> the right call.

---

## Random search in practice

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
dist = {"model__learning_rate": loguniform(1e-2, 3e-1),
        "model__num_leaves": randint(16, 256)}
search = RandomizedSearchCV(pipe, dist, n_iter=60, cv=cv, random_state=0)
```

Sample from distributions, not lists — a list is a grid with extra steps — and
fix `random_state`. `n_iter` is a budget: the time you will spend, divided by the
cost of one cross-validated fit.

---

## Bayesian optimization

Random search has no memory. Bayesian optimization fits a cheap surrogate of
"hyperparameters to score" from the trials so far and uses it to pick the next
point, balancing predicted quality against uncertainty.

$$
EI(\theta) = \mathbb{E}\left[\max(0, f(\theta) - f^{*})\right]
$$

Optuna's default sampler is **TPE**: it models the densities of good and bad
trials separately and samples where their ratio is highest. Unlike a Gaussian
process it handles categorical and conditional parameters naturally, and it
reaches random search's best in roughly half the trials.

---

## Optuna

```python
import optuna

def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 256, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
    )
    pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
    return cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=60)
```

The objective returns one number. Everything the search may vary goes through
`trial.suggest_*`; the splitter, the metric and the data are closed over and
fixed. Seed the sampler — an unseeded study is not an experiment.

---

## Search spaces in log scale

```python
trial.suggest_float("reg_lambda", 1e-3, 1e2, log=True)
```

A learning rate of 0.01 and one of 0.02 differ as much as 0.1 and 0.2 do.
Sampling uniformly from $[10^{-3}, 10^{2}]$ puts 99% of the draws above 1 and
never explores the small end at all.

Log scale for anything spanning orders of magnitude: learning rates,
regularisation strengths, `C`, `gamma`. Linear for fractions and counts inside
one decade: `subsample`, `colsample_bytree`, `max_depth`.

If the best value sits on an edge of the range, the range was wrong. Widen it
rather than reporting a boundary as an optimum.

---

## Pruning

```python
study = optuna.create_study(direction="maximize",
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
```

Most trials are visibly bad long before they finish. A pruner reports an
intermediate value — fold 2 of 5, or boosting round 200 of 5000 — and kills a
trial whose partial score is below the median of previous trials at that step.

`trial.report(value, step)` then `if trial.should_prune(): raise
optuna.TrialPruned()` — two lines, and a typical budget stretches two or three
times further.

Pruning is a budget multiplier, not a better search, and it biases against slow
starters: do not prune the first rounds of a low-learning-rate model.

---

## The budget question

| Budget | Do this |
|---|---|
| Minutes | defaults plus early stopping |
| An hour | 30–60 TPE trials over 4–6 parameters |
| A night | 200+ trials, pruning on, learning rate dropped at the end |
| Longer | stop searching; get more or better features |

Diminishing returns arrive early. Tuning buys one to three percent over sensible
defaults; a new feature buys more, and fixing a leak changes the answer entirely.
Search is the last thing you do — tuning against a broken split optimises it.

---

## The multiple-comparisons trap

Two hundred trials produce two hundred noisy estimates of the same quantity. The
maximum of two hundred noisy numbers is above the truth, and the more you search
the further above it lands. Your reported best validation score is a biased
estimate of your model's quality, and the bias grows with the budget.

Three consequences you must live with:

- The best trial's score is **not** the model's score. Report the held-out test
  number instead — computed once, after the search is closed.
- A trial beating the second-best by less than the fold standard deviation is a
  tie. Take the simpler configuration.
- If you need an honest estimate of the *whole procedure*, wrap the search in
  nested CV.

> The search picks the hyperparameters. It does not get to report the result.
