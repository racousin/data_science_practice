## k-nearest neighbours

- **Non-parametric**, instance-based (memory-based)
- **No training phase** — `fit` stores the data, at most indexing it for fast
  search, and the cost moves to prediction
- Prediction: find the $k$ closest training points, then vote or average


The same point is a triangle at $k = 3$ and a square at $k = 5$. $k$ is the
model.

---

## Classification

For a new point $x$, compute $d(x, x_i)$ for every $i = 1, \ldots, n$ and keep
the indices of the $k$ nearest:

$$
N_k(x) = \{\, i : d(x, x_i) \leq d(x, x_{(k)}) \,\}
$$

where $x_{(k)}$ is the $k$-th closest training point. Then predict the majority
class:

$$
\hat{y} = \arg\max_c \sum_{i \in N_k(x)} \mathbb{1}(y_i = c)
$$

---

## Classification, in a picture

![A new point classified at k = 3 and at k = 7](assets/tabular/knn-classification.png)

At $k = 3$ the vote is two triangles to one star: class B. At $k = 7$ it is four
stars to three triangles: class A.

---

## Regression

Same idea: average the neighbours' target values instead of voting.

$$
\hat{y}(x) = \frac{1}{k} \sum_{i \in N_k(x)} y_i
$$

![Linear regression versus kNN regression](assets/tabular/knn-vs-linear-regression.png)

The line is fitted to every point. kNN predicts at the star from the circled
neighbourhood alone.

---

## Distance metrics

![Euclidean, Manhattan and Minkowski paths between the same two points](assets/tabular/distance-metrics.png)

$$
\mathrm{Euclidean:} \quad d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}
$$

$$
\mathrm{Manhattan:} \quad d(x, x') = \sum_{j=1}^{p} |x_j - x'_j|
$$

Straight across, or along the grid: the two distances rank neighbours
differently, and the right one depends on what a unit of each feature means.

---


## Scale before you measure

> **Features must be scaled.** Use `StandardScaler`.

This is not a style preference. A feature measured in metres and one measured in
euros are added together inside that square root, so the one with the larger
numeric range silently becomes the only feature that matters.

---

## The key hyperparameter

![kNN regression at k = 1, 3 and 10: from following every point to smooth](assets/tabular/knn-bias-variance.png)

| $k$ | Boundary | Regime |
|---|---|---|
| small (e.g. 1) | complex, follows every point | overfitting — low bias, high variance |
| large (up to $n$) | smooth; constant at $k = n$ | underfitting — high bias, low variance |

$k$ is chosen on the validation set, like every other hyperparameter. Bias and
variance are made precise in *Trees and Ensembles*; for now read them as "wrong
on average" and "changes with the sample".

---

## kNN in scikit-learn

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

`fit_transform` on train, `transform` on test — never `fit_transform` on both.

---

## Regression, and weighted votes

```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)
```

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=15, weights="distance")
clf.fit(X_train_s, y_train)
```

`weights="distance"` weights each neighbour by the inverse of its distance, so
the closest count most. The default, `"uniform"`, counts all $k$ equally.

---

## The curse of dimensionality

![A neighbourhood of fixed size holds less and less of the data in one, two and three dimensions](assets/tabular/curse-of-dimensionality.png)

As dimensions increase, data becomes exponentially sparse. A box of side 0.1
holds about 10% of uniform data on a line, 1% on a square, 0.1% in a cube. To
keep $k$ neighbours, the neighbourhood has to grow until it is no longer local.
