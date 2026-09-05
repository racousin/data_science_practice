# Gradient Boosting in Practice

This is the workhorse. On the project's prediction track, a tuned
gradient-boosting model on a clean feature table is the baseline your submission
has to beat and, most of the time, the submission itself.

<!-- notes: 30 minutes, the most immediately useful lesson of the session. Do the
early-stopping demo live: fit with n_estimators=5000 and no early stopping, show
the validation curve turning up, then add the callback. The feature-importance
warning is the one they will otherwise get wrong in the project report. -->

---

## Three implementations of one algorithm

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | level-wise (depth) | leaf-wise (loss) | symmetric / oblivious |
| Speed on wide data | good | fastest | slower |
| Categorical features | `enable_categorical`, else one-hot | integer codes | native, ordered target statistics |
| Default quality | needs tuning | needs tuning | strong out of the box |
| Overfits small data | some | easily | least |

All three optimise the same regularised objective. Pick LightGBM when rows are
many and iteration speed matters, CatBoost when the table is mostly categorical
or small, XGBoost when you want the most predictable behaviour. The difference
between them is smaller than the difference between tuned and untuned.

---

## Leaf-wise growth

Level-wise growth expands every node at a depth before going deeper. Leaf-wise
growth always expands the leaf with the largest loss reduction, wherever it is.

Leaf-wise reaches a lower loss for the same number of leaves and overfits far
more readily on small data, because it will grow one deep branch to isolate a
handful of rows. This is why LightGBM's depth control is `num_leaves`, not
`max_depth`: keep it well below $2^{d}$ for the depth $d$ you have in mind, and
raise `min_child_samples` to 50 or more when rows are few.

---

## The parameters that actually matter

| Parameter | Direction | Typical range |
|---|---|---|
| `learning_rate` | lower = better, slower | 0.01–0.1 |
| `n_estimators` | set high, let early stopping choose | 2000–10000 |
| `num_leaves` / `max_depth` | capacity per tree | 31–127 / 4–8 |
| `min_child_samples` | higher = smoother leaves | 20–200 |
| `subsample` | row sampling per tree | 0.6–1.0 |
| `colsample_bytree` | column sampling per tree | 0.5–1.0 |
| `reg_lambda` | L2 on leaf values | 0–10, log scale |

Seven parameters. Everything else in the documentation is a refinement you will
not measure on your data.

---

## Learning rate and number of trees are one parameter

They trade against each other almost exactly: halving `learning_rate` doubles the
trees needed for the same fit, at slightly better generalisation and twice the
cost.

The practical consequence: **do not tune them jointly.** Fix the learning rate
by budget (0.05 while searching, 0.01–0.02 for the final fit) and let early
stopping pick the number of trees for you at each setting.

Tuning `n_estimators` in a grid search is the most common way to waste a
compute budget on this algorithm.

---

## Early stopping

```python
import lightgbm as lgb
model = lgb.LGBMRegressor(n_estimators=10000, learning_rate=0.03)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
print(model.best_iteration_)
```

Set `n_estimators` far higher than you need and stop when the validation metric
has not improved for `early_stopping_rounds` iterations.

> The evaluation set used for early stopping is part of training. It is not a
> validation set any more, and it is certainly not a test set.

Inside cross-validation, the eval set must come from the training folds. Passing
your outer test set as `eval_set` is leakage with a progress bar.

---

## Early stopping inside a Pipeline

`eval_set` is handed straight to the final estimator, so it never passes through
`prep`. Transposing the block above into a `Pipeline` raises before a tree grows:

```python
boosted.fit(X_a, y_a, model__eval_set=[(X_b, y_b)])
# ValueError: pandas dtypes must be int, float or bool.
# Fields with bad pandas dtypes: sex: str, race: str, ...
```

Fit the preprocessor on the training part, then transform both sides yourself:

```python
from sklearn.base import clone
prep = clone(preprocessor).fit(X_a)
model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05)
model.fit(prep.transform(X_a), y_a,
          eval_set=[(prep.transform(X_b), y_b)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
```

`cross_val_score` gives you no hook for this: split each training fold by hand,
or run `lgb.cv` with the preprocessing applied per fold.

---

## Native categorical handling

```python
X["city"] = X["city"].astype("category")
model = lgb.LGBMClassifier().fit(X_tr, y_tr, categorical_feature=["city"])
```

LightGBM sorts a categorical feature's levels by gradient statistics and splits
the sorted order, which finds subsets one-hot encoding cannot reach in a shallow
tree. CatBoost goes further with ordered target statistics: the encoding for a
row uses only rows that came before it in a random permutation, which is
target encoding without the leakage.

Native handling beats one-hot when cardinality is high; below ten levels the
difference is noise. Session 2 covers the encoding alternatives — the point here
is that you often do not need them.

---

## Monotonic constraints

```python
model = lgb.LGBMRegressor(monotone_constraints=[1, 0, -1])
```

Force the prediction to be non-decreasing in feature 1 and non-increasing in
feature 3, whatever the data says locally. One entry per feature: `1`, `0`, `-1`.

It costs a little accuracy and buys a model that cannot embarrass you: a price
that never falls with quantity, a risk score that never falls with debt. Use it
where an expert will read the model, or where training data is sparse in a region
you still have to behave sensibly in.

---

## Feature importance misleads

`model.feature_importances_` reports **gain** — the total loss reduction
attributed to splits on each feature — or, worse, **split count**, which is
LightGBM's default. Both are biased:

- toward high-cardinality features, which offer more places to split
- toward continuous over binary features, for the same reason
- arbitrarily among correlated features: one absorbs the credit, the others read
  as unimportant

It is also computed on the training data. A feature the model overfits scores
highly precisely because it overfits.

Gain importance answers "what did this model split on", not "what matters".

---

## Measure importance on held-out data

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)
```

Shuffle one column in the validation set and measure how much the score drops. A
feature carrying no information drops nothing. It is model-agnostic and uses data
the model did not train on. Correlated features still confuse it — both members
of a pair look unimportant, because either can substitute for the other.

SHAP decomposes one prediction into per-feature contributions with a consistency
guarantee gain importance lacks. Use permutation importance to decide what to
drop, SHAP to explain a single case:

```python
import shap
values = shap.TreeExplainer(model).shap_values(X_val)
```

---

## A starting configuration

```python
params = dict(n_estimators=10000, learning_rate=0.03, num_leaves=63,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0)
```

Fit that with early stopping, record the score, and only then search: capacity
first (`num_leaves`, `min_child_samples`), then sampling, then regularisation,
then drop the learning rate for the final fit.

Automated pipeline search is a real tool and a poor teacher — it is in the
Reference module, not in this session. The two lessons that follow decide whether
the score you just recorded means anything.

---

## Check yourself

1. Why is `n_estimators` the one parameter you never put in a grid search?

   **Answer.** It trades against `learning_rate` almost exactly — halving the
   rate doubles the trees needed for the same fit — so fix the rate by budget
   and let early stopping choose the tree count at each setting.

2. Run this. You should get exactly the output shown.

   ```python
   from lightgbm import LGBMClassifier
   print(LGBMClassifier().importance_type)   # -> split
   ```

   That is the default this lesson warns about: LightGBM ranks features by how
   often they were split on, not by how much loss they removed.

3. `boosted.fit(X_a, y_a, model__eval_set=[(X_b, y_b)])` raises
   `ValueError: pandas dtypes must be int, float or bool`. Why?

   **Answer.** `eval_set` is forwarded to the estimator untouched, so it never
   passes through the `prep` step and LightGBM receives the raw string columns.
   Fit the preprocessor yourself and transform both sides before the call.
