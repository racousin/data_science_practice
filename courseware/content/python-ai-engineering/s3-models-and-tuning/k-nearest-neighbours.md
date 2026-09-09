# K-Nearest Neighbours

The first non-parametric model, and the simplest idea in the course: to predict
for a new point, look at the points near it.

<!-- notes: ~10 of the 30 minutes budgeted for the three models. The scaling
requirement and the curse of dimensionality are the two things that must land. -->

---

## The idea

- **Non-parametric**, instance-based (memory-based)
- **No training phase** — just store the data
- Prediction: look at the $K$ closest neighbours, then vote or average

![Neighbourhoods at K=3 and K=5](assets/s3-models-and-tuning/k-nearest-neighbours/knn-neighbourhoods.png)

Note what the picture already shows: the same point is classified differently at
$K=3$ and $K=5$. $K$ is the model.

---

## Classification

For a new point $x$, find the $K$ nearest neighbours:

$$
\forall i \in 1, \ldots, n, \quad \text{compute } d(x, x_i)
$$

$$
N_k(x) = \{\, x_i : d(x, x_i) \leq d(x, x_{(k)}) \,\}
$$

Then predict the majority class:

$$
\hat{y} = \arg\max_c \sum_{i \in N_k(x)} \mathbb{1}(y_i = c)
$$

![KNN classification](assets/s3-models-and-tuning/k-nearest-neighbours/knn-classification.png)

---

## Regression

Same idea, average the target values instead of voting:

$$
\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i
$$

![Linear versus KNN regression](assets/s3-models-and-tuning/k-nearest-neighbours/knn-vs-linear-regression.png)

---

## Distance metrics

$$
\text{Euclidean:} \quad d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}
$$

$$
\text{Manhattan:} \quad d(x, x') = \sum_{j=1}^{p} |x_j - x'_j|
$$

$$
\text{Minkowski:} \quad d(x, x') = \left( \sum_{j=1}^{p} |x_j - x'_j|^q \right)^{1/q}
$$

---

## Scale before you measure

![Three distance metrics](assets/s3-models-and-tuning/k-nearest-neighbours/distance-metrics.png)

> **Features must be scaled.** Use `StandardScaler`.

This is not a style preference. A feature measured in metres and one measured in
euros are added together inside that square root, so the one with the larger
numeric range silently becomes the only feature that matters.

---

## The key hyperparameter

![Bias-variance as K increases](assets/s3-models-and-tuning/k-nearest-neighbours/knn-bias-variance.png)

| $K$ | Boundary | Regime |
|---|---|---|
| small (e.g. 1) | complex, follows every point | overfitting — low bias, high variance |
| large (e.g. $n$) | smooth, nearly constant | underfitting — high bias, low variance |

$K$ is chosen on the validation set, like every other hyperparameter.

---

## In scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)
```

```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)
```

`fit_transform` on train, `transform` on test — never `fit_transform` on both.

---

## The curse of dimensionality

![Data becomes sparse as dimensions grow](assets/s3-models-and-tuning/k-nearest-neighbours/curse-of-dimensionality.png)

As dimensions increase, data becomes exponentially sparse. KNN's neighbours
become far away and meaningless — every point is roughly equidistant from every
other, and "nearest" stops carrying information.

This is the single reason KNN is a poor default on wide data, and it is why the
500-column one-hot encoding from Session 2 was flagged as a problem.
