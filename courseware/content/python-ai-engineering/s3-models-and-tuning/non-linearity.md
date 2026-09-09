# Non-Linearity

Session 2 left you with two models, both of which draw straight lines. Most data
is not on a straight line. This lesson is the cheapest fix — and the trap that
comes with it.

<!-- notes: 20 minutes. Opens the session. The recap at the top is deliberate:
half the room will have forgotten the normal equation between sessions. -->

---

## Where we were

$$
D = (X, Y) = (x_i, y_i)_{i \in \mathbb{N}}
\qquad
f_\theta : \mathbb{R}^p \rightarrow \mathbb{R}
\qquad
\arg\min_{f \in \mathcal{F}} \ell\big(Y, f(X)\big)
$$

**Linear regression** — a hyperplane, fitted in closed form:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \ldots + \theta_p x_p = X\theta
\qquad
\min_\theta \|y - X\theta\|^2 \;\Rightarrow\; \hat{\theta} = (X^T X)^{-1} X^T y
$$

**Logistic regression** — the same hyperplane, squashed and thresholded:

$$
P(Y = 1 \mid x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

$$
\min_\theta \; -\sum_{i=1}^{n} \big[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \big]
$$

---

## The same hyperplane, twice

![Linear versus logistic regression](assets/s3-models-and-tuning/non-linearity/linear-vs-logistic.png)

Left, the hyperplane *is* the prediction. Right, it is squashed into a
probability and then thresholded.



---

## Where they fail

![Non-linear datasets](assets/s3-models-and-tuning/non-linearity/non-linear-datasets.png)

Linear and logistic regression cannot cover any of these. No straight line
separates a circle from the ring around it.

---

## Lift it, and a plane will do

![Concentric circles are not linearly separable](assets/s3-models-and-tuning/non-linearity/circles-lifted-3d.png)

There is no such line — but there is a *plane*, once you add a third dimension.
That observation is the whole idea behind this lesson and behind the kernel
trick later in the session.

---

## Polynomial regression

Same linear regression. New features.

$$
x \mapsto \phi(x) = (1, \, x, \, x^2, \, x^3, \, \ldots, \, x^d)
$$

$$
\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \ldots + \theta_d x^d = \phi(x)^T \theta
$$

$$
\min_\theta \|y - \Phi\theta\|^2 \;\Rightarrow\; \hat{\theta} = (\Phi^T \Phi)^{-1} \Phi^T y
$$

---

## Non-linear in $x$, linear in $\theta$

![Polynomials of increasing degree](assets/s3-models-and-tuning/non-linearity/polynomial-degrees.png)

> **Non-linear in $x$, linear in $\theta$.**

That is why the normal equation still applies unchanged. You did not build a new
model; you built new columns and handed them to the old one.

---

## In scikit-learn

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('reg',  LinearRegression()),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Wrapping it in a `Pipeline` is not cosmetic: it makes `PolynomialFeatures` fit
inside each cross-validation fold, which is the leakage rule from the previous
lesson enforced structurally.

---

## The bill

Raising $d$ raises capacity, and capacity is exactly what overfits. Degree
$n - 1$ interpolates your training set perfectly and predicts nothing. The next
lesson is how you keep the flexibility without paying that bill.
