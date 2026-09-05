# Ensembles: Boosting

Bagging trains models in parallel and averages away variance. Boosting trains
them in sequence, each one repairing the last, and drives down **bias**. It is
the family that wins tabular competitions.

<!-- notes: ~15 of the 30 ensemble minutes. AdaBoost is worked through because it
is the one where the weight update is legible; gradient boosting is what they
will actually use. -->

---

## The idea

- Train models **sequentially**, each correcting the previous one's errors
- Focus on the hard examples — misclassified points, or large residuals
- Reduces **bias**: makes weak learners strong

![Weak learners combined into a strong one](assets/s3-models-and-tuning/ensembles-boosting/boosting-weak-learners.png)

The mirror image of bagging. Same building block, opposite failure mode
addressed.

---

## AdaBoost, step by step

**Step 1 — initialise** uniform sample weights:

$$
w_i = \frac{1}{n} \qquad \forall i \in 1, \ldots, n
$$

**Step 2 — for each round** $t = 1, \ldots, T$:

**2a.** Train a weak learner $h_t$ on the weighted data.

**2b.** Compute the weighted error:

$$
\epsilon_t = \sum_{i=1}^{n} w_i \, \mathbb{1}\big(h_t(x_i) \neq y_i\big)
$$

**2c.** Compute the learner's weight — a good learner gets a high $\alpha_t$:

$$
\alpha_t = \frac{1}{2} \ln \frac{1 - \epsilon_t}{\epsilon_t}
$$

**2d.** Update the sample weights — misclassified points get heavier:

$$
w_i \leftarrow w_i \cdot \exp\big(\alpha_t \, \mathbb{1}(h_t(x_i) \neq y_i)\big)
\qquad\text{then normalize:}\qquad
w_i \leftarrow \frac{w_i}{\sum_j w_j}
$$

**Step 3 — final prediction**, a weighted vote:

$$
H(x) = \text{sign}\left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)
$$

### Reading $\alpha_t$

| $\epsilon_t$ | $\alpha_t$ | Effect |
|---|---|---|
| low | high | strong vote |
| high | low | weak vote |
| $0.5$ (random) | $0$ | **ignored entirely** |

A learner that is no better than a coin gets exactly zero weight, because
$\ln(1) = 0$. The formula throws away useless models for free.

---

## Gradient boosting

Same sequential idea, but each new tree is fitted to the **residuals** — the
gradient of the loss — rather than to reweighted points. That generalises
boosting to any differentiable loss.

| Parameter | Name | Effect |
|---|---|---|
| `n_estimators` ($T$) | boosting rounds | more rounds → better fit (risk of overfit) |
| `learning_rate` ($\eta$) | shrinkage | smaller → needs more rounds, but generalises better |
| `max_depth` | tree depth | usually shallow: 3–8 |

> **Trade-off:** small $\eta$ + large $T$ → better, but slower.

Note `max_depth` 3–8. Boosting wants *weak* learners; a deep tree already has low
bias and leaves the ensemble nothing to correct.

---

## In code

```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
gb.fit(X_train, y_train)
```

```python
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

gb  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)
```

---

## Bagging versus boosting

![Bagging and boosting from the same building block](assets/s3-models-and-tuning/ensembles-boosting/bagging-vs-boosting.png)

| | Bagging | Boosting |
|---|---|---|
| Training | parallel, independent | sequential, dependent |
| Diversity from | bootstrap rows (+ random features) | reweighting / residuals |
| Attacks | **variance** | **bias** |
| Base learner | deep, low-bias trees | shallow, weak trees |
| Overfits if | rarely — more trees is safe | yes — too many rounds |
| Examples | Random Forest | AdaBoost, XGBoost, LightGBM, CatBoost |

The "overfits if" row is the practical one: adding trees to a random forest is
close to free, adding rounds to a booster is not.
