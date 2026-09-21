# The Tabular Landscape

Deep learning took vision and text. It did not take tables. On a few thousand
rows of mixed numeric and categorical columns, a tree ensemble still wins — and
the models in this lesson are the ones it has to beat.

<!-- notes: 40 minutes. Ask the room who expects a neural network to win on their
project data; most say yes. Correct that early. The recap of linear and logistic
regression is deliberate: half the room will have forgotten the normal equation
since Python AI Engineering. Then ten minutes on non-linearity and fifteen on
regularisation. The KKT slide is optional depth — skip it if the room is
struggling, but the constrained view is what makes the diamond picture make
sense. -->

---

## Deep learning did not take tables

Repeated benchmarks on medium-sized tabular data — Shwartz-Ziv and Armon (2022),
Grinsztajn et al. (2022) — reach the same conclusion: tree ensembles beat tuned
neural networks on most datasets, with a fraction of the tuning budget.

Three structural reasons:

- Columns have no translation invariance and no ordering. The inductive bias
  that makes convolution work has no analogue in a table.
- Features are heterogeneous in scale, type and meaning. A tree splits each on
  its own terms; a dense layer mixes them all in the first matmul.
- Real tables are small. Five thousand rows is normal, and that starves a
  network while being plenty for boosting.

---

## The setup

Every model in this session answers the same question. Given $n$ labelled rows
with $p$ features, pick from a family $\mathcal{F}$ of functions the one with
the smallest loss $\ell$:

$$
D = (X, Y) = (x_i, y_i)_{i=1}^{n}
\qquad
f : \mathbb{R}^p \rightarrow \mathbb{R}
\qquad
\hat{f} = \arg\min_{f \in \mathcal{F}} \ell\big(Y, f(X)\big)
$$

A model is a choice of $\mathcal{F}$ and of $\ell$. Fitting is the minimisation.

---

## Linear regression

A hyperplane:

$$
\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j = X\beta
$$

with a column of ones in $X$ for the intercept. Minimising the residual sum of
squares has a closed form, the normal equation:

$$
\min_\beta \|y - X\beta\|^2 \;\Rightarrow\; \hat{\beta} = (X^T X)^{-1} X^T y
$$

Closed form when $p$ is small, gradient descent when it is not. scikit-learn
calls a least-squares solver (`scipy.linalg.lstsq`) rather than inverting
$X^T X$.

---

## Linear regression in scikit-learn

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
```

One coefficient per feature, each readable as "one unit of $x_j$ moves the
prediction by $\beta_j$, all else equal". No other model gives that for free.

---

## Logistic regression

The same linear score, squashed to a probability by the sigmoid $\sigma$:

$$
P(y = 1 \mid x) = \sigma(\beta_0 + \beta^T x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}
$$

and fitted by maximum likelihood, which is minimising the log-loss:

$$
\ell(\beta) = -\sum_{i=1}^{n} \big[ y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i) \big]
$$

There is no closed form this time. The minimum is found iteratively (gradient descent).

---

## Logistic regression in scikit-learn

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, y_train)
```

`LogisticRegression` is penalised by default: an L2 penalty whose strength is
set by `C`. The "plain" logistic regression you fit is already the regularised
model of the second half of this lesson, and `C` gets its own slide there.

---

## A linear boundary

![A logistic-regression boundary: the line x2 = -x1 + 4 separates the two classes](assets/tabular/linear-decision-boundary.png)

Predicting class 1 when $\hat{p} \geq 0.5$ is predicting it when
$\beta_0 + \beta^T x \geq 0$. The boundary is a hyperplane: in two dimensions, a
straight line.

---

## The same hyperplane, twice

![Linear versus logistic regression](assets/tabular/linear-vs-logistic.png)

Left, the hyperplane *is* the prediction. Right, it is squashed into a
probability and then thresholded.

---

## Where they fail

![Three non-linear datasets, each with its best straight line](assets/tabular/non-linear-datasets.png)

Both models draw straight lines, and most data is not on a straight line. The
best straight line through a sine or a V misses the shape. And in
classification, no straight line separates a circle from the ring around it.

---

## Lift it, and a plane will do

![Concentric circles are not linearly separable; lifted into 3D, they are](assets/tabular/circles-lifted-3d.png)

There is no such line — but there is a *plane*, once you add a third dimension.
One extra feature is enough: $x_3 = x_1^2 + x_2^2$, the squared distance to the
centre, is small on the inner circle and large on the ring. That observation is
the whole idea behind the next four slides, and behind the kernel trick of the
next lesson.

---

## Polynomial regression

Same linear regression. New features.

$$
x \mapsto \phi(x) = (1, \, x, \, x^2, \, x^3, \, \ldots, \, x^d)
$$

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \ldots + \beta_d x^d = \phi(x)^T \beta
$$

With $\Phi$ the matrix whose rows are the $\phi(x_i)$, the normal equation is
unchanged:

$$
\min_\beta \|y - \Phi\beta\|^2 \;\Rightarrow\; \hat{\beta} = (\Phi^T \Phi)^{-1} \Phi^T y
$$

![Polynomials of degree 0 to 5](assets/tabular/polynomial-degrees.png)


---

## Polynomial features in scikit-learn

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_poly, y_train)
y_pred = reg.predict(X_test_poly)
```

`fit_transform` on train, `transform` on test — the standard transformer
pattern. Here `fit` only records the number of input columns and the list of
exponents, so nothing is learned from the data. But keep the habit: with a
scaler or an imputer, calling `fit_transform` on the test set leaks.

---

## How many columns you built

`poly.get_feature_names_out()` shows what you actually built. With $p$ input
columns at degree $d$, the bias column included, the number of columns is

$$
\binom{p + d}{d}, \qquad \mathrm{e.g.} \quad \binom{5 + 3}{3} = 56
$$

Five features at degree 3 become 56 columns. That growth is why Session 2
applied `PolynomialFeatures` to a handful of chosen columns, never the whole
matrix — and why a high-degree fit needs the penalty that comes next.

---

## Regularisation

![The same points fitted without regularisation (overfit) and with it (good fit)](assets/tabular/regularization-effect.png)

> Penalising large parameters forces the model to find simpler solutions that
> generalise better.

Unpenalised least squares with correlated features — or with high-degree
polynomial features — gives huge coefficients that cancel each other out.
Penalising their size fixes it.

---

## One extra term, one hyperparameter

The simplest guard against overfitting: one extra term in the loss.

$$
\hat{\beta} = \arg\min_\beta \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \lambda R(\beta)
$$

$\lambda$ controls the strength of the regularisation. It is a hyperparameter:

| $\lambda$ | Effect |
|---|---|
| $\lambda = 0$ | no penalty → the standard loss, back to square one |
| $\lambda \rightarrow \infty$ | all coefficients driven to 0 → the model predicts a constant (the intercept is not penalised) |

Everything useful is in between, and you find it on the validation set.

---

## Two penalties

$$
\|\beta\|_1 = |\beta_1| + |\beta_2| + \ldots + |\beta_p|
\qquad
\|\beta\|_2^2 = \beta_1^2 + \beta_2^2 + \ldots + \beta_p^2
$$

![Unit balls of the L1, L2 and L-infinity norms](assets/tabular/lp-norm-balls.png)

The unit balls — every $\beta$ of norm at most 1 — for L1 (a diamond), L2 (a
disc) and, for comparison, L∞, $\max_j |\beta_j|$ (a square).

---

## The constrained view (KKT)

$$
\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|
\qquad \Longleftrightarrow \qquad
\min_\beta \|y - X\beta\|^2 \quad \text{s.t.} \quad \|\beta\| \leq t
$$

For every $\lambda \geq 0$ there is a $t \geq 0$ such that both problems have
the same solution — the Karush–Kuhn–Tucker (KKT) conditions.

So regularisation is the same thing as confining $\beta$ to a ball of radius
$t$. Larger $\lambda$, smaller ball. The solution lands where the loss contours
first touch the ball.

---

## Why lasso zeroes coefficients

![Ridge, lasso and elastic-net constraint regions touched by the loss contours](assets/tabular/ridge-lasso-elasticnet-regions.png)

The $L_1$ ball has **corners on the axes**. Contours touch corners, and a corner
means some $\beta_j$ is exactly zero. The $L_2$ ball is round and has no
corners, so it shrinks coefficients without ever zeroing them.

---

## Ridge, lasso, elastic net

$$
\mathrm{Ridge:} \quad \min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2
$$

$$
\mathrm{Lasso:} \quad \min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1
$$

$$
\mathrm{Elastic\ net:} \quad \min_\beta \|y - X\beta\|^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2
$$

| Penalty | Effect |
|---|---|
| Ridge (L2) | shrinks together, keeps all features |
| Lasso (L1) | drives coefficients to exactly zero |
| Elastic net | sparse, but stable under collinearity |

---

## Ridge, lasso, elastic net in scikit-learn

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge   = Ridge(alpha=1.0)
lasso   = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

model = elastic                     # any of the three
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

> **Note:** `alpha` in scikit-learn is $\lambda$ in the formulas — up to a
> scale, next slide.

---

## What alpha means exactly

`Ridge` minimises exactly the formula above. `Lasso` and `ElasticNet` divide
the squared error by $2n$, and `ElasticNet` splits `alpha` with
$\rho$ = `l1_ratio`:

$$
\frac{1}{2n}\|y - X\beta\|^2 + \alpha\rho\|\beta\|_1 + \frac{\alpha(1 - \rho)}{2}\|\beta\|_2^2
$$

So the same `alpha` is not the same strength in `Ridge` and in `Lasso`.
`l1_ratio=1` is the lasso; `Lasso` is exactly that case.

---

## The same thing for classification

Swap the squared error for the log-loss $\ell(\beta)$ of logistic regression,
with $\hat{p}_i = \sigma(x_i^T\beta)$. The penalties do not change.

$$
\mathrm{Ridge\ (L2):} \quad \min_\beta \; \ell(\beta) + \lambda\|\beta\|_2^2
$$

$$
\mathrm{Lasso\ (L1):} \quad \min_\beta \; \ell(\beta) + \lambda\|\beta\|_1
$$

$$
\mathrm{Elastic\ net:} \quad \min_\beta \; \ell(\beta) + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2
$$

$$
\ell(\beta) = -\sum_{i=1}^{n}\big[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\big],
\qquad
\hat{p}_i = \frac{1}{1 + e^{-x_i^T\beta}}
$$

---

## Penalised logistic regression in scikit-learn

```python
from sklearn.linear_model import LogisticRegression
ridge   = LogisticRegression(C=1.0)                 # l1_ratio=0
lasso   = LogisticRegression(C=10, l1_ratio=1, solver="saga",
                             max_iter=1000)
elastic = LogisticRegression(C=10, l1_ratio=0.5, solver="saga",
                             max_iter=1000)
```


scikit-learn puts `C` on the loss instead of $\lambda$ on the penalty — it
minimises $C \cdot \ell(\beta) + R(\beta)$ — so $C$ plays the role of
$1/\lambda$. **Larger `C` means less regularisation**; small `C` means a
strongly regularised model.

---

## The rule: fit the baseline first

> Before any boosting or advanced methods, fit a regularised linear or logistic model on the same
> split with the same metric, and write the number down.

It costs two minutes and buys three things: proof the pipeline runs end to end,
a floor any later model must clear, and an early warning. A linear model scoring
0.99 AUC on a hard problem is not a triumph — it is leakage, found for free.
