# Logistic Regression

The same linear combination, wrapped in one function, trained with a different
loss — and it becomes a classifier. Three small changes, each with a reason.

<!-- notes: 30 minutes. The "why not MSE" slide is the one that earns its place:
students reach for MSE by default, and the non-convexity argument is the answer.
-->

---

## Why not just regress

The target is a class, and what you want is $P(y = 1 \mid x)$ — a number in
$[0,1]$.

![Linear regression cannot model a probability](/api/academic_courses/assets/lessons/151/linear-vs-logistic-fit.png)

A linear model has codomain $\mathbb{R}$. It will return $-0.3$ and $1.4$, and
neither is a probability. This is the output-space argument from the models
lesson, in its first concrete instance.

---

## The sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
\qquad
\sigma : \mathbb{R} \longrightarrow (0, 1)
$$

![The sigmoid curve](/api/academic_courses/assets/lessons/151/sigmoid-curve.png)

Strictly increasing, smooth, $\sigma(0) = 1/2$, and $\sigma'(z) = \sigma(z)
\,(1 - \sigma(z))$ — a derivative expressible in the function's own value, which
is what makes the gradients cheap.

---

## The model

**1.** The same linear score as before:

$$
z = \theta^\top x
$$

**2.** Squash it into a probability:

$$
P(y = 1 \mid x) = \sigma(\theta^\top x) = \frac{1}{1 + e^{-\theta^\top x}}
$$

**3.** Decide:

| Predict | when | equivalently |
|---|---|---|
| $\hat{y} = 1$ | $P(y = 1 \mid x) \geq 0.5$ | $\theta^\top x \geq 0$ |
| $\hat{y} = 0$ | otherwise | $\theta^\top x < 0$ |

The two columns say the same thing, because $\sigma$ crosses $1/2$ exactly
where its argument crosses 0. So **the decision boundary is the hyperplane**
$\theta^\top x = 0$: logistic regression is a linear classifier, and the sigmoid
only decides how confidence varies as you move away from that hyperplane.

Equivalently, the model is linear in the log-odds:

$$
\log \frac{P(y = 1 \mid x)}{P(y = 0 \mid x)} = \theta^\top x
$$

which is how $\theta_j$ should be read — a one-unit change in $x_j$ shifts the
log-odds by $\theta_j$, multiplying the odds by $e^{\theta_j}$.

The threshold is a **choice**, not part of the model. Moving it trades precision
against recall; Session 3.

---

## Why not MSE

![Convex versus non-convex loss surfaces](/api/academic_courses/assets/lessons/151/convex-vs-nonconvex.png)

Squared error composed with a sigmoid is **non-convex** in $\theta$ — multiple
local minima, and the guarantee from the training lesson is gone. Worse, the
gradient carries a factor $\sigma'(z)$, which is near zero exactly when the
model is confidently wrong: the points you most need to fix produce almost no
gradient.

Cross-entropy fixes both. It is convex in $\theta$ here, and its gradient is

$$
\nabla_\theta \, \ell = \frac{1}{n} \sum_{i=1}^{n} \big(\hat{p}_i - y_i\big)\, x_i
$$

The $\sigma'$ cancels. The update is driven by the raw residual $\hat{p}_i -
y_i$, so a confident mistake produces a large step — which is the behaviour you
wanted.

$$
\min_{\theta} \; -\frac{1}{n}\sum_{i=1}^{n} \big[\, y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \,\big]
$$

There is **no closed form** for this $\arg\min$. Logistic regression is the first
model in the course that has to be fitted iteratively — but the problem is
convex, so gradient descent finds the global optimum.

![Gradient descent trajectory for a classifier](/api/academic_courses/assets/lessons/151/gd-trajectory-classification.gif)

| | Initial | Trained |
|---|---|---|
| Parameters | $a = 0.19$, $b = -1.17$ | $a = -1.01$, $b = 1.6$ |
| Prediction for $(1.4, 1.2)$ | blue | red |

---

## In scikit-learn

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
```

Same interface as `LinearRegression` — different family, different loss,
different solver, identical two methods. With $p$ features it learns $p + 1$
parameters, exactly as the linear model does; the sigmoid adds none.

One warning: `sklearn` applies L2 regularisation **by default**, at strength
`C=1.0`. So the fit above is *not* the unpenalised maximum-likelihood estimate
you would derive on paper — pass `C=np.inf` for that. (The older `penalty=None`
spelling was deprecated in scikit-learn 1.8 and is removed in 1.10.)
Regularisation itself is the opening of Session 3.
