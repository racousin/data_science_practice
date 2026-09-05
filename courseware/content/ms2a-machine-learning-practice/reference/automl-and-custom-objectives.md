# AutoML and Custom Objectives

Reference only — not covered in class. Both are useful tools that teach none of
the judgement this course is about: AutoML hides the decisions, custom
objectives presuppose you already know which decision to make. Read the first
half before a deadline, the second half when the metric you are graded on stops
matching the loss you are minimising. Session 3 covers the models both sit on.

<!-- notes: Self-study, linked from Session 3. Never lectured. -->

---

## What AutoML actually automates

| Tool | Searches | Notable for |
|---|---|---|
| **AutoGluon** | model zoo + multi-layer stacking | strongest tabular default, no tuning needed |
| **FLAML** | cost-aware hyperparameter search | fast, small dependency footprint |
| **auto-sklearn** | sklearn pipelines, Bayesian search | meta-learned warm start |
| **H2O AutoML** | GLM, GBM, XGBoost, DL + ensembles | JVM stack, enterprise deployment |
| **TPOT** | genetic programming over pipelines | exports readable sklearn code |

All five automate the same three things: model choice, hyperparameter search,
and ensembling. That is the last 30% of a tabular problem — the part you were
going to do with a loop anyway.

---

## Using one

```python
from autogluon.tabular import TabularPredictor

pred = TabularPredictor(label="target", eval_metric="roc_auc").fit(
    train_df, time_limit=600, presets="medium_quality"
)
print(pred.leaderboard(valid_df))
```

Ten minutes of compute buys a stacked ensemble of a dozen models plus a
leaderboard saying which contributed. On a clean dataset with a well-posed
split this is genuinely hard to beat by hand in an afternoon.

Notice that you passed `eval_metric` yourself and built `valid_df` yourself.
Those are the decisions that matter, and the tool made neither.

---

## What it cannot do

- **Design your validation split** — it will happily run random K-fold on time
  series or grouped data and report an inflated score.
- **Build your features** — it searches the model space, not the feature space,
  and domain features are where tabular problems are won.
- **Choose your metric** — `roc_auc` on an asymmetric-cost problem is a wrong
  answer computed very efficiently.
- **Explain itself** — a three-layer stack of 15 models is not something you
  can defend in a project report or ship into a regulated setting.

**The recommendation.** Run AutoML early, on your own split, as the number to
beat. If your hand-built model does not beat it you have learned something real
about your features. Never submit a model you cannot explain — including in
this course's project, where the report carries the marks.

---

## When the built-in loss is not the business loss

Squared error assumes over- and under-predicting cost the same; log loss
assumes every misclassification costs the same. Usually false, usually
harmless — until it is not.

| Situation | Built-in loss says | Reality |
|---|---|---|
| Stock-out costs 5× overstock | symmetric squared error | penalise under-prediction |
| Need a 90% service level | the mean | you want the 0.9 quantile |
| Missing a fraud costs €2000 | balanced log loss | weight the positive class |
| Ranking search results | pointwise regression | only the order matters |

Reach for a custom objective when the asymmetry is in the *cost*, not in the
class frequencies. For imbalance alone, `scale_pos_weight` or `sample_weight`
is simpler and does the same job.

---

## What a booster actually needs

Gradient boosting does not need your loss. It needs its first two derivatives
with respect to the prediction, per sample:

$$
g_i = \frac{\partial L}{\partial \hat{y}_i}, \qquad h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2}
$$

because the value it puts in leaf $j$ is a Newton step:

$$
w_j = - \frac{\sum_{i \in j} g_i}{\sum_{i \in j} h_i + \lambda}
$$

$h$ sits in a denominator. That single fact is the whole story of the pitfalls
below: it must stay positive, and it must have a sensible scale.

---

## Asymmetric cost

Weight the squared error by a factor depending on the sign of the residual. For
$L = w (y - \hat{y})^2$ the derivatives are $g = -2w(y - \hat{y})$, $h = 2w$.

```python
def asymmetric_mse(y_true, y_pred):
    r = y_true - y_pred
    w = np.where(r > 0, 5.0, 1.0)      # under-prediction costs 5x
    return -2 * w * r, 2 * w

model = xgb.XGBRegressor(objective=asymmetric_mse)
```

LightGBM takes the same `(grad, hess)` signature. The resulting model is
deliberately biased: it over-predicts on average, and that is the point. Do not
then evaluate it with RMSE and conclude it got worse.

---

## Quantile regression, and the hessian trap

The pinball loss for quantile $\tau$ is piecewise linear, so its true second
derivative is zero everywhere:

$$
L_{\tau} = \max\left(\tau (y - \hat{y}),\ (\tau - 1)(y - \hat{y})\right)
$$

```python
def pinball(y_true, y_pred, tau=0.9):
    r = y_true - y_pred
    grad = np.where(r > 0, -tau, 1.0 - tau)
    hess = np.ones_like(y_pred)        # NOT 1e-6
    return grad, hess
```

A hessian of `1e-6` is the mistake in half the blog posts on this: each leaf
value becomes $\sum g / 10^{-6}$, the trees explode, and the run yields `nan`
or wild predictions. A constant 1 turns the Newton step into a plain gradient
step at a sane scale — which is what LightGBM's built-in `quantile` does. The
general rules: $h > 0$ strictly, $g$ and $h$ on comparable scales across
samples, and check your return shapes, because a broadcast mistake here trains
silently.

---

## Custom evaluation metrics

The objective is what the model optimises; the eval metric is what early
stopping watches. Keeping them separate is the point — optimise a smooth
surrogate, stop on the metric you are graded on.

```python
def rmspe(y_true, y_pred):                       # -> a float, lower is better
    return float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)))

model = xgb.XGBRegressor(eval_metric=rmspe, early_stopping_rounds=50)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
```

`eval_metric` and `early_stopping_rounds` moved from `fit()` to the constructor
in xgboost 2.0; the old form raises `TypeError: XGBModel.fit() got an unexpected
keyword argument 'eval_metric'` on the 3.2.0 that the platform runtime ships.
The callable now returns a **plain float**, not the old `(name, value)` tuple —
return the tuple and training dies with `TypeError: must be real number, not
tuple` several iterations in. The label in `evals_result()` comes from the
function's `__name__`, and early stopping watches the *last* metric listed.

Get the direction right: XGBoost's sklearn API minimises, LightGBM wants an
explicit `is_higher_better`, sklearn's `make_scorer` takes `greater_is_better`.
Passing an accuracy as if it were a loss gives early stopping that reliably
selects the worst iteration, and nothing warns you.

---

## Checklist

- run AutoML on your own split, as a baseline to beat, never as the submission
- the split, the features and the metric stay yours
- custom objective only when the *cost* is asymmetric, not merely the classes
- supply $g$ and $h$; keep $h$ strictly positive and O(1), never `1e-6`
- verify a new objective on data whose answer you know before trusting it
- objective and eval metric differ — check the sign convention of each

---

## Check yourself

1. AutoML has run for ten minutes and beaten your hand-built model. Name two
   things it did **not** do for you, and say which of them is the reason its
   number might be a lie.

   **Answer.** It did not design your validation split, build your features,
   choose your metric, or explain itself. The split is the one that makes the
   number a lie: run random K-fold on time-series or grouped data and it
   reports an inflated score very efficiently.

2. Run this. You should get exactly the output shown.

   ```python
   g = [-0.9, -0.9, 0.1]                     # three samples in one leaf, lambda = 0
   print(-sum(g) / sum([1.0] * 3))           # -> 0.5666666666666667
   print(-sum(g) / sum([1e-6] * 3))          # -> 566666.6666666666
   ```

   **Answer.** That is the hessian trap. `h` sits in the denominator of the
   Newton step, so `hess = 1e-6` multiplies every leaf value by a million: the
   trees explode and the run yields `nan` or wild predictions. Return a
   constant `1` for a piecewise-linear loss.

3. On xgboost 3.2.0 — the version the platform runtime ships — where do
   `eval_metric` and `early_stopping_rounds` go, and what must your custom
   metric return?

   **Answer.** Both are constructor arguments, not `fit()` arguments; passing
   `eval_metric` to `fit()` raises `TypeError`. The callable returns a plain
   float, not a `(name, value)` tuple, and its `__name__` becomes the label.
