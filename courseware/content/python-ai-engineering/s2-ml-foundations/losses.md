# Losses

The model family says which functions are available. The loss says which one you
want. It is the only place in the pipeline where "good" is defined, so it is
worth being precise about.

<!-- notes: 30 minutes. Two things to land: the loss is a modelling choice with
consequences (MSE vs MAE on outliers), and loss ≠ metric. The second one causes
more confusion than anything else in this session. -->

---

## Definition

A loss maps a set of true targets and the model's predictions to a single real
number — larger means worse:

$$
\ell : \mathcal{Y}^n \times \mathcal{Y}^n \longrightarrow \mathbb{R}_{+}
\qquad
(Y, \hat{Y}) \longmapsto \ell(Y, \hat{Y})
$$

Almost every loss you will meet is an average of a **pointwise** loss $L$:

$$
\ell(Y, \hat{Y}) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{y}_i)
$$

which matters more than it looks: it makes the loss an empirical mean, so it
estimates $\mathbb{E}\,[L(y, f_\theta(x))]$.

---

## The learning problem

Composing the model with the loss turns a modelling question into an
optimisation one:

$$
\theta^{*} = \arg\min_{\theta \in \mathbb{R}^{d}} \ \ell\big(Y, f_\theta(X)\big)
= \arg\min_{\theta \in \mathbb{R}^{d}} \ \frac{1}{n}\sum_{i=1}^{n} L\big(y_i, f_\theta(x_i)\big)
$$

That single line is supervised learning. Everything in the rest of this course is
a choice of $f$, a choice of $L$, or a way of computing the $\arg\min$.

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| | Data | Model | Loss | Optimise | Parameters |
| | collect $(X, Y)$ | choose $\mathcal{F}$ | choose $\ell$ | solve the $\arg\min$ | learned $\theta^{*}$ |

---

## Regression losses

$$
\ell_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\qquad
\ell_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

![Residuals against a fitted line](assets/s2-ml-foundations/losses/residuals.png)

The residuals $y_i - \hat{y}_i$ are the vertical distances. The loss is a choice
of how to aggregate them.

---

## The shape is the whole story

![Shapes of common regression losses](assets/s2-ml-foundations/losses/loss-function-shapes.png)

The shape is the whole story. MSE is quadratic, so a residual of 10 costs 100
while a hundred residuals of 1 cost 100 between them — one bad point weighs as
much as a hundred ordinary ones. MAE is linear and charges them proportionally.

---

## One outlier, two fits

![MAE is more robust than MSE](assets/s2-ml-foundations/losses/mae-vs-mse-robustness.png)

The statistical version of the same statement: the minimiser of MSE over a
constant is the **mean**, and the minimiser of MAE is the **median**.

---

## Mean or median, inherited

$$
\arg\min_{c} \sum_i (y_i - c)^2 = \bar{y}
\qquad
\arg\min_{c} \sum_i |y_i - c| = \mathrm{med}(y)
$$

So the robustness of MAE is not a heuristic — it is the robustness of the
median, inherited. MSE keeps the default anyway because it is differentiable
everywhere and, for a linear model, has a closed-form solution. MAE is not
differentiable at zero.

---

## Classification losses

You cannot minimise the error rate directly: it is piecewise constant, so its
gradient is zero almost everywhere and undefined on the boundary. Predict a
probability instead and score it with **cross-entropy** — the negative
log-likelihood of the data under the model.

$$
L_{\text{BCE}}(y, \hat{p}) = -\big[\, y \log \hat{p} + (1 - y) \log (1 - \hat{p}) \,\big]
$$

$$
L_{\text{CE}}(y, \hat{p}) = -\sum_{k=1}^{K} y_k \log \hat{p}_k
$$

![Log loss](assets/s2-ml-foundations/losses/log-loss.png)

---

## Reading the two branches

| If | The term that survives | What it punishes |
|---|---|---|
| $y = 1$ | $-\log \hat{p}$ | a low $\hat{p}$ |
| $y = 0$ | $-\log(1 - \hat{p})$ | a high $\hat{p}$ |

Both branches diverge as the model becomes confidently wrong. Take $y = 1$ and
$\hat{p} = 0.01$: cross-entropy charges $-\log 0.01 = 4.61$, while squared error
charges $(1 - 0.01)^2 = 0.98$ — less than its own maximum of 1. An unbounded
penalty on confident mistakes is what MSE lacks, and it is the reason
classification uses cross-entropy.

---

## The four you will actually use

| Task | Loss | Formula |
|---|---|---|
| Regression | Mean squared error | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ |
| Regression, outliers | Mean absolute error | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ |
| Binary classification | Binary cross-entropy | $-\frac{1}{n}\sum [\, y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \,]$ |
| $K$-class classification | Categorical cross-entropy | $-\frac{1}{n}\sum_i \sum_k y_{ik} \log \hat{p}_{ik}$ |

---

## Loss is not metric

The distinction that causes the most confusion in this session.

| | Loss | Metric |
|---|---|---|
| Consumed by | the optimiser | you, and the leaderboard |
| Must be | differentiable (and ideally convex) | anything you can compute |
| Example | cross-entropy, MSE | F1, accuracy, AUC, RMSE in euros |

You minimise cross-entropy because you can differentiate it. You are *judged* on
F1 because that is what the problem cares about. They are two different numbers
and they do not always move together — a model can have the lower loss and the
worse F1, usually because the decision threshold is wrong, and the loss never
sees the threshold.

<!-- notes: This is the setup for the evaluation lesson at the end of this
session: threshold tuning is a metric-side fix, applied after training, and it
changes no parameter. -->

---

## The interface

A prediction is only worth something next to the truth. The loss is a plain
function of two vectors, and it lives in `sklearn.metrics`:

```python
from sklearn.metrics import mean_squared_error, log_loss

mean_squared_error(y_test, y_hat)   # regression:     (y - f(x))^2
log_loss(y_test, proba)             # classification: -log p(y | x)
```

Nothing here mentions a model — you could compute the same number on a constant
prediction, or on a guess written by hand. That is the point: the loss defines
what "good $\theta$" means, independently of how you found $\theta$.
