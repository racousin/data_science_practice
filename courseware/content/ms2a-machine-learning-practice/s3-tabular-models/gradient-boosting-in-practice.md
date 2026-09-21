# Gradient Boosting in Practice

This is the workhorse. On the project's prediction track, a tuned
gradient-boosting model on a clean feature table is the baseline your submission
has to beat and, most of the time, the submission itself.

<!-- notes: 30 minutes, the most immediately useful lesson of the session. Do the
early-stopping demo live: fit with n_estimators=5000 and no early stopping, show
the validation curve turning up, then add the callback. The feature-importance
warning is the one they will otherwise get wrong in the project report. -->

---


## AdaBoost

![AdaBoost reweighting: each round focuses on the last round's mistakes](assets/tabular/adaboost.jpg)

Each round trains a weak learner on weighted data, then upweights the examples
it got wrong. The final model is a weighted vote of all the rounds.

---

## AdaBoost, step by step

Labels and learner outputs are coded $\pm 1$: $y_i, h_t(x_i) \in \{-1, +1\}$.

**Step 1 — initialise** uniform sample weights:

$$
w_i = \frac{1}{n} \qquad \forall i \in 1, \ldots, n
$$

**Step 2 — for each round** $t = 1, \ldots, T$:

**2a.** Train a weak learner $h_t$ on the weighted data.

---

## AdaBoost, the error and the vote weight

**2b.** Compute the weighted error:

$$
\epsilon_t = \sum_{i=1}^{n} w_i \, \mathbb{1}\big(h_t(x_i) \neq y_i\big)
$$

**2c.** Compute the learner's weight; a good learner gets a high $\alpha_t$:

$$
\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}
$$

---

## AdaBoost, the weight update

**2d.** Update the sample weights, so that misclassified points get heavier, then
normalise:

$$
w_i \leftarrow w_i \, e^{-\alpha_t y_i h_t(x_i)}
\qquad\text{then}\qquad
w_i \leftarrow \frac{w_i}{\sum_j w_j}
$$

$y_i h_t(x_i)$ is $+1$ on a correct row and $-1$ on a wrong one, so:

$$
w_i \leftarrow w_i \, e^{-\alpha_t} \quad \mathrm{if\ correct}, \qquad w_i \leftarrow w_i \, e^{\alpha_t} \quad \mathrm{if\ wrong}
$$

After the update, $h_t$ scores exactly 50% on the new weights, so the next
learner has to find something new.

---

## AdaBoost, the final vote

**Step 3 — final prediction**, a weighted vote:

$$
H(x) = \text{sign}\Big( \sum_{t=1}^{T} \alpha_t h_t(x) \Big)
$$

A learner with weighted error $\epsilon_t$ votes with weight $\alpha_t$. The
exponential loss behind the update makes AdaBoost brittle under label noise: a
mislabelled row is upweighted forever.

---

## AdaBoost in scikit-learn

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    learning_rate=0.5)
```


---

## Gradient boosting

Generalise: instead of reweighting, fit each new tree to the **negative gradient
of the loss** at the current predictions. For squared loss that gradient is the
residual, so each tree predicts what the ensemble still gets wrong.

$$
r_i^{(m)} = - \left[ \frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}}
$$

$$
F_m(x) = F_{m-1}(x) + \nu h_m(x)
$$

The shrinkage $\nu$ scales each correction down. The libraries call it
`learning_rate` — the same name a neural network's step size $\eta$ goes by, a
different quantity. Any differentiable loss works, which is why one algorithm
covers regression, classification and ranking.

---

## Gradient boosting's three dials

| Parameter | Name | Role |
|---|---|---|
| `n_estimators` ($T$) | boosting rounds | how many corrections are added |
| `learning_rate` ($\nu$) | shrinkage | how much of each correction is kept |
| `max_depth` | tree depth | how much each tree can correct: shallow, 4–8 |

`max_depth` stays shallow because boosting wants *weak* learners: a deep tree
already has low bias and leaves the ensemble nothing to correct. How $\nu$ and
$T$ trade against each other, and why early stopping picks $T$, is the next
lesson.



---

## Which implementation in sklearn

`GradientBoostingClassifier` and `GradientBoostingRegressor` are the textbook
implementation: exact splits, one core, slow beyond about 10,000 rows.
`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` are
scikit-learn's fast, binned version of the same algorithm.


## To go beyond : Three implementations of one algorithm

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

## Gradient boosting in code

```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
gb.fit(X_train, y_train)
```

```python
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=4)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)
```
