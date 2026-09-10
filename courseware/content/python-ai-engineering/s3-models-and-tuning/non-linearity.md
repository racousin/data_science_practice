# Non-Linearity

Session 2 left you with two models, both of which draw straight lines. Most data
is not on a straight line.

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
That observation is the whole idea behind this lesson.

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


---

## In scikit-learn

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

`fit_transform` on train, `transform` on test — the standard transformer pattern.
Here `fit` only records the shape and the list of exponents, so nothing is learned
from the data. But keep the habit: with a scaler or an imputer, calling
`fit_transform` on the test set leaks.

`poly.get_feature_names_out()` shows what you actually built — 5 features at
degree 3 becomes 56 columns.
