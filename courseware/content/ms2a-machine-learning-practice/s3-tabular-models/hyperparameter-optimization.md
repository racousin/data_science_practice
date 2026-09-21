# Hyperparameter Optimisation

Every model in this session came with a dial: $K$, $C$, `max_depth`, $\lambda$,
`learning_rate`. Search is the cheapest source of accuracy you have and the
easiest place to fool yourself. This lesson is about spending a fixed budget
well, about what Optuna actually does with that budget, and about reporting the
result honestly.

<!-- notes: 80 minutes. First 20: grid, random, and the Bergstra and Bengio
picture — nine grid points sampling three values of the important parameter,
nine random points sampling nine — the one thing to draw on the board. Next 25:
Optuna and TPE; draw the three panels (split, two densities, ratio) on the
board before showing the figure. Write the recipe objective live; they will
copy it into the lab. Storage, parallelism, visualisation and multi-objective
are reference slides: skim them if the clock is short. -->

---

## Parameters versus hyperparameters

| Parameters ($\theta$) | Hyperparameters |
|---|---|
| Learned during training | Set **before** training |
| Weights, coefficients, splits | $K$, $C$, $\lambda$, `max_depth`, `learning_rate`, … |
| Minimise **training** loss | Optimise **validation** performance |

The last row is why they cannot be learned the same way: there is no gradient
of the validation score with respect to `max_depth`. You search instead.

---

## What is being searched

![Grid, random and Bayesian search on the same loss](assets/tabular/tunning.jpg)

The objective is the cross-validated score — expensive, noisy, non-convex, and
without a gradient. Every method below is a different way of spending a fixed
number of evaluations on a black box: on a lattice, anywhere at random, or
where the past trials say the good region is.

---

## Grid search

Every combination of a discrete list per hyperparameter, exhaustively:
reproducible, trivially parallel, complete within the grid you wrote.

The cost is the product of the axes times the folds — three values of four
parameters at five folds is 405 fits — and each new parameter multiplies the
bill. It also cannot find a value you did not list.

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    "num_leaves":        [15, 31, 63],
    "min_child_samples": [10, 20, 50, 100],
    "learning_rate":     [0.03, 0.1, 0.3],
}
model = LGBMClassifier(n_estimators=200, verbose=-1)   # fixed, not searched
grid = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
```

`cv=5` means each of the $3 \times 4 \times 3 = 36$ combinations is fitted five
times — 180 fits. That is the practical argument for `RandomizedSearchCV` with
an explicit `n_iter`.

---


## Grid versus random search

![Grid search versus random search](assets/tabular/grid-vs-random-search.png)

Grid search tries every combination of a list per hyperparameter; random search
samples combinations from ranges. Look carefully at the figure: with 9 trials, grid search tests 3 distinct values
of each hyperparameter; random search tests 9. When only one of the two matters
— the usual case — random search has explored it three times as finely for the
same cost.


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

## Bayesian optimisation

Random search has no memory. Bayesian optimisation fits a cheap surrogate of
"hyperparameters to score" from the trials so far and uses it to choose the next
point, trading predicted quality against uncertainty. The usual criterion is the
expected improvement over the best loss $y^{*}$ (for a score, minimise its
negative):

$$
EI_{y^*}(x) = \mathbb{E}\left[\max(0,\ y^{*} - y)\right]
$$

Two ways to build the surrogate:

- a **Gaussian process** models $p(y \mid x)$: a predicted loss and an
  uncertainty at every point
- **TPE** models $p(x \mid y)$: where do the good trials live, and where do the
  bad ones? It is Optuna's default, and the rest of this lesson

---

## Optuna: define-by-run

```python
import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=50)
print(study.best_params, study.best_value)
```

- **Study** — one optimisation: a direction, a sampler, a pruner, a storage,
  and the history of trials
- **Trial** — one call of the objective; `trial.suggest_*` asks the sampler for
  a value *and* records the parameter
- **Objective** — your function: a trial in, the number to optimise out

---

## The `suggest_*` calls

| Call | Samples |
|---|---|
| `suggest_float(name, low, high, log=False, step=None)` | a real number |
| `suggest_int(name, low, high, log=False, step=1)` | an integer |
| `suggest_categorical(name, choices)` | one of a list of `None`, bool, int, float or str |

The name is the parameter's identity across trials: the sampler learns from
every earlier trial that suggested the same name. Bounds are inclusive.
`log=True` samples uniformly in $\log x$; `step` puts the values on a grid.

---

## An objective for a boosting pipeline

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-2, 2e-1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 256, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
    )
    model = clone(pipe).set_params(
        **{f"model__{k}": v for k, v in params.items()})
    return cross_val_score(model, X_tr, y_tr, cv=cv,
                           scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=60)
```

---

## What the objective must do

The objective returns one number. Everything the search may vary goes through
`trial.suggest_*`; the splitter, the metric and the data are closed over and
fixed, so every trial is scored on the same folds.

`clone(pipe)` gives each trial its own pipeline. Calling `set_params` on one
shared object works until trials run in parallel threads and overwrite each
other's settings.

Seed the sampler — an unseeded study is not an experiment.

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
one decade: `subsample`, `colsample_bytree`, `max_depth`. TPE also fits its
densities in log space when `log=True`, so the scale shapes how it generalises
from one trial to the next.

If the best value sits on an edge of the range, the range was wrong. Widen it
rather than reporting a boundary as an optimum.

---

## Conditional search spaces

```python
def objective(trial):
    family = trial.suggest_categorical("family", ["logreg", "lgbm"])
    if family == "logreg":
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, max_iter=2000))
    else:
        model = LGBMClassifier(
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            learning_rate=trial.suggest_float(
                "learning_rate", 1e-2, 0.2, log=True),
            verbose=-1)
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
```

This is the model-selection loop of the previous lesson, with the model family
as one more hyperparameter. `C` exists only in the logistic-regression trials,
`num_leaves` only in the LightGBM ones.

---


---

## Samplers

| Sampler | Use it when |
|---|---|
| `TPESampler` | the default: mixed, categorical or conditional spaces |
| `RandomSampler` | a baseline; many parallel workers; a sanity check |
| `GridSampler` | you must reproduce a fixed grid, or the space is tiny |
| `CmaEsSampler` | continuous parameters, hundreds of trials; needs `cmaes` |
| `GPSampler` | few, expensive trials, mostly continuous; needs `torch`, `scipy` |
| `NSGAIISampler` | several objectives: the default for those studies |

Every sampler takes `seed=`, and swapping one is a single argument to
`create_study`: the objective does not change. Run a `RandomSampler` study of
the same budget once: if TPE cannot beat it, the parameters you search barely
matter, and the accuracy is somewhere else.

---


## The budget question

| Budget | Do this |
|---|---|
| Minutes | defaults plus early stopping |
| An hour | 30–60 TPE trials over 4–6 parameters |
| A night | 200+ trials, pruning on, learning rate dropped at the end |
| Longer | stop searching; get more or better features |

Time one trial, divide the time you have by it, and write `n_trials` down
before the search starts. Do not use `study.optimize(timeout=...)`: the number
of trials then depends on the machine, and the study is not reproducible.

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
