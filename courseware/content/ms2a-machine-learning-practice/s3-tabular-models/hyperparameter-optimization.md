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

---

## Grid search in scikit-learn

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

## What `best_score_` is not

`best_score_` is a **cross-validated** score on the training data, and it is the
best of 36. It is not your test score, and reporting it as one is Leak 3 of the
previous lesson: using the evaluation set to choose.

`grid.best_estimator_` has already been refitted on all of `X_train` with the
winning combination (`refit=True` is the default). Score it on the test set,
once, and report that.

---

## Grid versus random search

![Grid search versus random search](assets/tabular/grid-vs-random-search.png)

Grid search tries every combination of a list per hyperparameter; random search
samples combinations from ranges. Look carefully at the figure: with 9 trials, grid search tests 3 distinct values
of each hyperparameter; random search tests 9. When only one of the two matters
— the usual case — random search has explored it three times as finely for the
same cost.

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

## Why build the space by running code

scikit-learn's searches are *define-and-run*: the space is a dict written before
the search starts, so every parameter exists in every trial.

Optuna is *define-by-run*: the space is whatever the objective asks for on this
call, built with ordinary Python.

- a **branch** — the family decides which parameters exist
- a **loop** — `n_layers` decides how many `units_{i}` to suggest
- a **computed bound** — `suggest_int("num_leaves", 2, 2 ** depth)` after
  `depth` has been drawn

The difficulty moves to the sampler: parameters appear and vanish between
trials, and it has to model a space it only discovers as it goes. That is what
TPE was designed for.

---

## TPE, step 1: start at random

TPE is the **Tree-structured Parzen Estimator** (Bergstra et al., 2011).
*Tree-structured*: the search space may branch, as above. *Parzen estimator*:
the kernel density model at its core.

```python
optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24,
                           seed=0)            # the defaults, plus a seed
```

The first `n_startup_trials` — ten by default — come from a plain
`RandomSampler`. A density needs data before it means anything, and these ten
are your random-search baseline for free. They count trials already in the
study, so a resumed study does not repeat them.

---

## TPE, step 2: split the history

Sort the $n$ finished trials by loss and cut them at a quantile $y^{*}$: the
best `min(ceil(0.1 * n), 25)` are **good**, the rest are **bad**. That is the
fraction γ, 10% by default, capped at 25 trials.

$$
l(x) = p(x \mid y < y^{*}) \qquad g(x) = p(x \mid y \geq y^{*})
$$

$l(x)$ is where the good trials live; $g(x)$ is where the bad ones live.
Pruned trials count too: they join the good group only when completed trials
run short, so in practice they sharpen $g(x)$ — where not to go.

TPE never predicts a loss. It only asks which of the two groups a new $x$ looks
more like.

---

## TPE, step 3: fit two densities

A Parzen estimator — a kernel density — for each group, parameter by parameter:

- one Gaussian kernel on each observed value, plus a wide **prior** kernel on
  the centre of the range, so no region ever gets zero density
- each kernel as wide as the gap to its farther neighbour, but never narrower
  than $(high - low)/(n + 2)$ for a group of $n$: few points give wide, smooth
  densities
- `log=True` fits in log space; a categorical gets a smoothed histogram over
  its choices instead of kernels
- past 25 observations, older trials weigh less, so $g(x)$ follows the search
  as it moves

---

## TPE, step 4: sample, score, pick

1. Draw `n_ei_candidates` — 24 by default — from $l(x)$.
2. Evaluate $l(x) / g(x)$ at each candidate.
3. Return the candidate with the largest ratio as the next trial.

Drawing from $l(x)$ keeps the candidates near past good trials; the ratio
prefers those where bad trials are rare. The prior kernel and the wide
bandwidths keep some candidates far from anything tried: that is the
exploration.

---

## One TPE step, read out of Optuna

![TPE's good/bad split, its two densities and the ratio it maximises](assets/tabular/optuna-tpe-anatomy.png)

A real `TPESampler` after 100 trials, all random (`n_startup_trials=100`), so
the two densities are easy to see. The ten best trials sit in the two deepest
basins; $l(x)$ peaks there, $g(x)$ everywhere else, and the chosen candidate
lands at $x = 1.565$, a hair from the global minimum at $x = 1.556$.

---

## Why the ratio is expected improvement

Bergstra et al. (2011) plug the split into the expected improvement. By Bayes,
$p(y \mid x) = p(x \mid y)\,p(y) / p(x)$; below $y^{*}$, $p(x \mid y) = l(x)$,
and $p(x) = \gamma\, l(x) + (1 - \gamma)\, g(x)$:

$$
EI_{y^*}(x) = \int_{-\infty}^{y^*} (y^* - y)\, \frac{l(x)\, p(y)}{p(x)}\, dy
= \frac{l(x) \int_{-\infty}^{y^*} (y^* - y)\, p(y)\, dy}{\gamma\, l(x) + (1 - \gamma)\, g(x)}
$$

$$
EI_{y^*}(x) \propto \left(\gamma + (1 - \gamma)\, \frac{g(x)}{l(x)}\right)^{-1}
$$

The integral does not depend on $x$, so expected improvement grows with
$l(x)/g(x)$. Under TPE's model, maximising the ratio *is* maximising EI — the
loss distribution $p(y)$ cancels, so TPE never has to predict a loss.

---

## One parameter at a time, or jointly

Default TPE is **univariate**: each parameter has its own $l$ and $g$, and each
is chosen on its own ratio. It cannot see that a small `learning_rate` wants
more leaves, or that two parameters only work together.

```python
optuna.samplers.TPESampler(multivariate=True, group=True, seed=0)
```

`multivariate=True` fits joint kernels over all parameters and scores each
candidate as a whole, which captures interactions. `group=True` splits a
conditional space into the sets of parameters that occur together, so each
branch is modelled on the trials that took it. Both still print an
experimental warning; both are worth trying when parameters interact.

---

## Why TPE and not a Gaussian process

A Gaussian process puts a kernel on a fixed-length numeric vector $x$:

- a categorical must be embedded, and distances between its choices invented
- a conditional parameter has no value in the trials that skipped its branch,
  so the vector has holes
- fitting costs $O(n^3)$ in the number of trials

TPE models each parameter from the trials where it appeared: a categorical is a
histogram, a conditional parameter simply has fewer observations, and the cost
grows linearly with $n$.

The price: univariate TPE ignores interactions, and on a small, smooth,
continuous space a GP usually needs fewer trials. That is where `GPSampler`
earns its place.

---

## Same budget, three searches

![Best-so-far loss for grid, random and TPE over five seeds](assets/tabular/optuna-grid-random-tpe.png)

A budget of $3^5 = 243$ trials is exactly the grid of three values on five
parameters. The grid stalls at 0.031, as close as three values per axis allow.
Random search keeps finding better values of the two parameters that matter
(0.0030). TPE, after its ten random trials, concentrates on them: it matches
random search's 243-trial result by trial 54 and ends ten times lower, at
0.0003.

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

## Pruning: the mechanics

```python
from sklearn.linear_model import SGDClassifier

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=0)
    for epoch in range(20):
        clf.partial_fit(X_fit, y_fit, classes=[0, 1])
        acc = clf.score(X_val, y_val)
        trial.report(acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return acc
```

Most trials are visibly bad long before they finish. `trial.report` records an
intermediate value — an epoch, a boosting round, a fold — and `should_prune`
asks the pruner whether to stop. A pruned trial is stored as `PRUNED`, and TPE
still learns from it.

---

## Which pruner

| Pruner | Stops a trial when |
|---|---|
| `MedianPruner` | its best value so far is worse than the median of finished trials at this step |
| `SuccessiveHalvingPruner` | at a rung (steps $r$, $4r$, $16r$, …) it is not in the top quarter of the trials that reached it |
| `HyperbandPruner` | the same rule with a factor of 3 by default, over several brackets that start judging at different rungs |

`MedianPruner(n_startup_trials=5, n_warmup_steps=0)` is also what
`create_study` uses when you choose nothing, so `should_prune` is live by
default. Successive halving is more aggressive; Hyperband hedges how early to
judge. Optuna's documentation, from its (non-deep-learning) benchmarks, pairs
`MedianPruner` with `RandomSampler` and `HyperbandPruner` with `TPESampler`.

---

## What pruning costs

Pruning is a budget multiplier, not a better search. In the study below, 21 of
60 trials were pruned, at 4.8 s each against 9.8 s for a completed one on the
laptop that drew the figures: roughly a fifth of the wall time saved. Pruning per fold can only save the
folds not yet run; pruning per boosting round saves more.

It biases against slow starters. A low learning rate or a strong regulariser
looks bad at round 50 and best at round 2,000; a pruner that judges at round
50 kills it every time. Set `n_warmup_steps` past the point where learning
curves stop crossing, and do not prune the first rounds of a low-learning-rate
model.

The intermediate values must be comparable across trials at the same step:
the same metric, on the same data, at the same point of training.

---

## Pruning a LightGBM fit

```python
import lightgbm as lgb
from optuna_integration import LightGBMPruningCallback

def objective(trial):
    model = lgb.LGBMClassifier(**sample_params(trial))
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False),
                         LightGBMPruningCallback(trial, "auc")])
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
```

The callback reports the eval set's `"auc"` after every boosting round and
raises `TrialPruned` when the pruner says so. The returned score comes from
`X_val`, not from `X_es`, which early stopping has already selected on.

---

## The LightGBM callback, in detail

- It lives in the separate `optuna-integration` package:
  `pip install "optuna-integration[lightgbm]"`. The old
  `optuna.integration` path only forwards to it.
- The metric name is LightGBM's own — `"auc"`, `"binary_logloss"`, `"l2"` —
  and it must agree with the study's direction, or the callback raises.
- `valid_name` defaults to `"valid_0"`, the first entry of `eval_set`.
- One split only. Inside a CV loop every fold reports rounds 0, 1, 2… again,
  and Optuna ignores the repeated steps with a warning. With folds, report
  once per fold instead — the recipe below does.

---

## Storage: resume a study

```python
study = optuna.create_study(
    study_name="lgbm-v1",
    storage="sqlite:///hpo.db",
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=0),
)
study.optimize(objective, n_trials=20)
```

The default storage is memory: the study dies with the process, and a crash at
trial 190 of 200 loses all of it. A database URL — SQLite, PostgreSQL, MySQL —
writes every trial as it finishes. Run the script again and
`load_if_exists=True` reattaches: twenty more trials, forty in the history.

`optuna.load_study(study_name=..., storage=...)` reopens it in a notebook. The
sampler is not stored: pass the same one when you resume.

---

## Parallel trials

**Several processes, one study.** Start the same script in four terminals, or
four cluster jobs, with the same `study_name` and `storage`. Each reads the
history, runs a trial and writes it back.

SQLite is for one process that stops and resumes; Optuna's FAQ advises against
it for parallel runs (`database is locked`). On one machine use
`JournalStorage(JournalFileBackend("journal.log"))`; across machines,
PostgreSQL or MySQL.

**Threads in one process.** `study.optimize(objective, n_trials=40, n_jobs=4)`
runs four trials at a time. It helps only when the objective releases the GIL,
as LightGBM and NumPy do; keep four times the model's own threads within your
cores.

---

## What parallel TPE costs

**Sample efficiency.** Four trials in flight are all suggested from the same
history; none knows where the other three went, so they cluster. On the toy
objective of the race above, 60 TPE trials in four threads ended at a median
best of about 0.0035 in one run, against 0.0019 in sequence (eight seeds); the
parallel number changes from run to run, for the reason in the next paragraph.
`TPESampler(constant_liar=True)` counts running trials as bad to push the
suggestions apart; in our runs it recovered some or all of the gap. Turn it on
whenever `n_jobs > 1`.

**Reproducibility.** A seed fixes the sampler's random stream, not the order in
which trials finish, and that order decides which history each suggestion
sees. Two seeded runs with `n_jobs=4` diverge from the very first trials; with
`n_jobs=1` they are identical. Run the study you report with `n_jobs=1`, or say
that it is not reproducible.

---

## Seed the search with what you know

```python
study.enqueue_trial({"learning_rate": 0.1, "num_leaves": 31,
                     "min_child_samples": 20}, skip_if_exists=True)
study.optimize(objective, n_trials=40)
```

The next `optimize` evaluates the queued configuration before it asks the
sampler; it counts towards `n_trials` and enters TPE's history like any other
trial. Parameters you leave out are sampled as usual, and `skip_if_exists=True`
stops a resumed script from queueing it twice.

Queue the library defaults, last month's best, or a colleague's configuration.
The search then starts from a known-good point, and the history shows what the
tuning bought over it.

---

## The study as a table

```python
df = study.trials_dataframe(attrs=("number", "value", "params", "state",
                                   "user_attrs"))
df.sort_values("value", ascending=False).head(5)
```

One row per trial: `number`, `value`, a `params_<name>` column per parameter,
`state` (`COMPLETE`, `PRUNED`, `FAIL`), and a `user_attrs_<key>` column for each
value stored with `trial.set_user_attr`. Without `attrs` you also get start,
end and duration.

Report the median of the top five next to the best. A large gap between them
means you found a lucky fold, not a better model.

---

## Optimisation history

![Optimisation history of the LightGBM study](assets/tabular/optuna-history.png)

The recipe at the end of this lesson, 60 trials on 12,000 rows of the Adult
census table. Each dot is a trial; the line is the best so far.

---

## Reading the history

Trial 0 is LightGBM's defaults for the five searched parameters: 0.917. The random startup does no better;
TPE's second suggestion reaches 0.922, and the best, 0.924 at trial 49, is
0.006 above the defaults. After trial 21 the best moves by less than 0.001:
more budget would buy little.

Late trials cost three times the early ones (16 s against 5 s on the laptop
that drew the figures): TPE moved to small learning rates, which need more
trees.

---

## Parameter importance

![fANOVA importance of the five hyperparameters](assets/tabular/optuna-importance.png)

`optuna.importance.get_param_importances(study)` uses **fANOVA** by default: fit
a random forest from parameters to objective over the completed trials, then
split the variance of its predictions into the share each parameter explains.

Here `min_child_samples` (0.50) and `colsample_bytree` (0.45) explain 95% of
what varied; `reg_lambda` explained nothing anywhere between 0.001 and 100.

---

## Reading importance with care

- It describes *these ranges* and *these trials*. Widen a range and the ranking
  moves; TPE concentrated its trials, so they are not a uniform sample.
- It says how much a parameter matters, not which way: that is the slice plot.
- It depends on the forest's seed: pass
  `FanovaImportanceEvaluator(seed=0)` to get the same numbers twice.
- Conditional parameters are left out unless you name them in `params=`.

Use it to shrink the next search: fix the parameters that explain almost
nothing at a sensible value, and spend the budget on the rest.

---

## Slice and contour plots

![Slice plot: each completed trial against one parameter](assets/tabular/optuna-slice.png)

The late, dark trials pile up at `colsample_bytree` 0.3–0.45 and
`min_child_samples` 5–10: against the floor of both ranges. Even the recipe's
ranges were too narrow for this dataset — widen them before trusting the
optimum. `num_leaves` is a flat cloud from 8 to 50: it barely matters here.

---

## The plots in code

```python
from optuna.visualization import (
    plot_contour, plot_intermediate_values, plot_optimization_history,
    plot_param_importances, plot_slice)

plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_slice(study, params=["learning_rate", "num_leaves"]).show()
plot_contour(study, params=["learning_rate", "num_leaves"]).show()
plot_intermediate_values(study).show()      # the pruning curves
```

These draw with plotly, interactive in a notebook.
`optuna.visualization.matplotlib` has the same functions for static figures
(experimental); the figures in this lesson are redrawn from the study itself
with matplotlib.

---

## Two objectives: the Pareto front

```python
import time

def objective(trial):
    model = fit_model(trial)        # sample_params(trial), then fit
    t0 = time.perf_counter()
    p = model.predict_proba(X_val)[:, 1]
    ms = 1000 * (time.perf_counter() - t0)
    return roc_auc_score(y_val, p), ms

study = optuna.create_study(directions=["maximize", "minimize"])
study.optimize(objective, n_trials=60)
front = study.best_trials                    # the Pareto-optimal trials
```

---

## Reading a Pareto front

There is no single best trial any more. A trial is on the **Pareto front** when
no other trial is at least as good on both objectives and better on one.
`study.best_trials` returns that front, and `plot_pareto_front(study)` draws it.

You choose the trade-off afterwards — the most accurate model under 5 ms, say
— instead of baking a weighting into the objective before you know the
options.

With `directions=`, the default sampler becomes `NSGAIISampler`; `TPESampler`
handles several objectives too. Pruning does not: `trial.report` raises
`NotImplementedError` in a multi-objective study.

---

## The recipe, 1/3: the search space

```python
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

def sample_params(trial):
    return dict(
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-2, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 256, log=True),
        min_child_samples=trial.suggest_int(
            "min_child_samples", 5, 200, log=True),
        colsample_bytree=trial.suggest_float(
            "colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float(
            "reg_lambda", 1e-3, 100.0, log=True),
        n_estimators=5000,      # a ceiling: early stopping picks
        subsample=0.8, subsample_freq=1,
        random_state=0, verbose=-1,
    )
```

---

## The recipe, 2/3: the objective

```python
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def objective(trial):
    params, scores, trees = sample_params(trial), [], []
    for k, (tr, va) in enumerate(CV.split(X, y)):
        X_fit, X_es, y_fit, y_es = train_test_split(
            X.iloc[tr], y.iloc[tr], test_size=0.2,
            stratify=y.iloc[tr], random_state=k)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        p = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y.iloc[va], p))
        trees.append(model.best_iteration_)
        trial.report(float(np.mean(scores)), step=k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("n_trees", int(np.median(trees)))
    return float(np.mean(scores))
```

---

## The recipe, 3/3: the study

```python
study = optuna.create_study(
    study_name="lgbm-v1", storage="sqlite:///hpo.db",
    load_if_exists=True, direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=0),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                       n_warmup_steps=1),
)
study.enqueue_trial({"learning_rate": 0.1, "num_leaves": 31,
                     "min_child_samples": 20, "colsample_bytree": 1.0,
                     "reg_lambda": 1e-3}, skip_if_exists=True)
study.optimize(objective, n_trials=60)
```

Trial 0 is LightGBM's defaults for the five searched parameters
(`reg_lambda` defaults to 0, outside a log range, so its floor stands in; the
recipe also fixes `subsample=0.8, subsample_freq=1`, which LightGBM does not). Afterwards, refit on all the training data with
`study.best_params` and `n_estimators` from the best trial's `n_trees`, then
score the test set once.

---

## Why the recipe looks like this

- **`n_estimators` is not searched.** It trades against `learning_rate` almost
  exactly; it is a ceiling, and early stopping picks the count in every fold.
- **Log scale** for `learning_rate`, `num_leaves`, `min_child_samples` and
  `reg_lambda`, which span decades; linear for `colsample_bytree`.
- **The early-stopping set comes from inside the training fold.** The
  validation fold `va` is touched once, after the fit.
- **Pruning is per fold.** The running mean is reported after each fold, and
  `n_warmup_steps=1` means no trial is judged on its first fold alone.
- **Seeded and stored.** The sampler has a seed; the study survives a crash.
- **Reproducing the figures** needs LightGBM pinned too: `deterministic=True,
  force_row_wise=True` and a fixed `n_jobs`, as in
  `tools/figures/ms2a_s3_optuna.py`.

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

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   lo, hi = 1e-3, 1e2
   print(round((hi - 1.0) / (hi - lo) * 100, 1))   # -> 99.0
   ```

   Sampling `reg_lambda` uniformly from that range puts 99% of the draws above
   1 and never explores the small end. That is why the space is `log=True`.

2. A default `TPESampler` has finished 40 trials. How many count as "good", and
   how does it choose trial 41?

   **Answer.** Four: `min(ceil(0.1 * 40), 25)`. It fits $l(x)$ to them and
   $g(x)$ to the other 36, draws 24 candidates from $l(x)$ and runs the one
   with the largest $l(x)/g(x)$, the largest expected improvement.

3. Two hundred trials, best score 0.842, fold standard deviation 0.011. What
   number goes in the report?

   **Answer.** Not 0.842: the maximum of two hundred noisy estimates is biased
   upward. Report the test score, computed once after the search is closed,
   and treat every trial within 0.011 of the best as a tie.
