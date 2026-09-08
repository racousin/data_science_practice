# Linear Regression

The first complete model: a family, a loss, two ways to solve it, and a
`scikit-learn` call. Everything later in the course is a variation on this
structure.

<!-- notes: 30 minutes. This audience has seen the algebra. What they usually
have not seen is the normal equation read as a projection, or `fit`/`predict`
framed as the thing that never changes. Spend the time there. -->

---

## The family

$$
f_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_p x_p = \theta^\top x
$$

with $x_0 \equiv 1$ absorbing the intercept. $p$ features means $d = p + 1$
parameters, and the output space is $\mathbb{R}$.

- $\theta_0$ is the **intercept** — the prediction when every feature is zero.
- $\theta_j$ is the effect on $\hat{y}$ of a one-unit change in feature $j$,
  the other features held fixed.

![A line in 2D, a plane in 3D](assets/s2-ml-foundations/linear-regression/regression-line-and-plane.png)

One feature fits a line, two a plane, $p$ a hyperplane. The picture stops being
drawable at $p = 3$; the algebra does not change.

---

## The loss

For example the MSE: differentiable everywhere, and
the only choice here that admits a closed form:

$$
\mathcal{L}(\theta) = \frac{1}{n} \| Y - X\theta \|_2^{2}
$$

---

## The normal equation

$\mathcal{L}$ is a convex quadratic in $\theta$, so the stationary point is the
global minimum. Differentiate and set to zero:

$$
\nabla_\theta \mathcal{L}(\theta) = -\frac{2}{n} X^\top (Y - X\theta) = 0
$$

$$
X^\top X \, \theta = X^\top Y
\qquad\Longrightarrow\qquad
\theta^{*} = (X^\top X)^{-1} X^\top Y
$$

The middle line is worth more than the last. $X^\top(Y - X\theta) = 0$ says the
residual is **orthogonal to the column space of $X$**, so $X\theta^{*}$ is the
orthogonal projection of $Y$ onto $\mathrm{span}(X)$. Least squares is a
projection, and the normal equations are the statement that you cannot do better
within the span.

This needs $X^\top X$ invertible — equivalently $\mathrm{rank}(X) = p + 1$. It
fails on collinear features: a `surface_m2` column and a `surface_ft2` column
are exactly proportional, the matrix is singular, and $\theta^{*}$ is not
unique. In practice `sklearn` returns the pseudo-inverse solution rather than
raising, which is worse than an error — the coefficients are then an arbitrary
choice among infinitely many, and reading them as effect sizes is meaningless.

| Pros | Cons |
|---|---|
| Exact, one shot | $O(p^3)$ in the number of features |
| No learning rate, no epochs | Does not scale to large $p$ |
| Fast for small and medium $p$ | Needs full column rank |

---

## Or descend

The same $\theta^{*}$ is reachable by the previous lesson's algorithm, and for
large $p$ that is the only practical route. It is also open the option with another losse.

![Gradient descent trajectory in parameter space](assets/s2-ml-foundations/linear-regression/gd-trajectory-regression.gif)


---

## In scikit-learn

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
```

`sklearn` solves the normal equations via a numerically stable least-squares
routine rather than inverting $X^\top X$ literally. `model.coef_` and
`model.intercept_` hold $\theta^{*}$.

---

## Reading the coefficients

| Surface (m²) | Rooms | Dist. centre (km) | Price (€) |
|---|---|---|---|
| 55 | 2 | 3.2 | 185,000 |
| 80 | 3 | 1.5 | 320,000 |
| 120 | 4 | 5.0 | 280,000 |
| 45 | 1 | 0.8 | 250,000 |
| 95 | 3 | 2.1 | 350,000 |

$$
\widehat{\text{price}} = 3200 \times \text{surface} - 1500 \times \text{distance} + 45000
$$

| | |
|---|---|
| $\theta_1 = 3{,}200$ | each additional m² adds €3,200, at fixed distance |
| $\theta_2 = -1{,}500$ | each additional km from the centre removes €1,500, at fixed surface |
| $\theta_0 = 45{,}000$ | the intercept — the fitted value at zero surface and zero distance, which is outside the data and should not be interpreted |

Two cautions that this audience will want stated. *At fixed distance* is not
optional: with correlated features, a coefficient is a partial effect and it can
change sign when you add or drop a column. And none of this is causal — it is
the best linear predictor of the joint distribution you sampled.

That said, this readability is why linear regression survives. Gradient boosting
will beat it on this table and will not hand you those three sentences.
