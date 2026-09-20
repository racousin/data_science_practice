# k-Nearest Neighbours and Support Vector Machines

Two classical models, and two reference points a tree ensemble is measured
against. k-nearest neighbours, the first non-parametric model of the session, is
the simplest idea in the course: to predict for a new point, look at the points
near it. The support vector machine changes the objective — not "minimise the
average error" but "put the boundary as far from both classes as possible" — and
adds a trick that buys non-linearity for almost nothing.

<!-- notes: 30 minutes: about twelve for kNN and eighteen for the SVM. Keep
both in proportion: they are reference points, not candidates. For kNN, the
scaling requirement and the curse of dimensionality are the two things that
must land. For the SVM, the margin derivation is short and worth doing in full
— it is the one place in this session where the geometry produces the
objective. -->

---

## k-nearest neighbours

- **Non-parametric**, instance-based (memory-based)
- **No training phase** — `fit` stores the data, at most indexing it for fast
  search, and the cost moves to prediction
- Prediction: find the $k$ closest training points, then vote or average

![Neighbourhoods at k = 3 and k = 5 around the same new point](assets/tabular/knn-neighbourhoods.png)

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

## Minkowski, the general case

$$
\mathrm{Minkowski:} \quad d(x, x') = \left( \sum_{j=1}^{p} |x_j - x'_j|^q \right)^{1/q}
$$

$q = 1$ is Manhattan, $q = 2$ Euclidean. It is scikit-learn's default metric,
with the exponent — called `p` there — set to 2.

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

---

## Where kNN stands

As the dimension grows, the nearest and the farthest neighbour converge. For
2,000 points uniform in the unit cube, the farthest is about 150 times as far as
the nearest in 2 dimensions, 2.5 times in 20 and 1.5 times in 100. Long before
they coincide, the vote is dominated by points that are not local, and
"nearest" stops carrying information.

This is the single reason kNN is a poor default on wide data — the previous
lesson's rule of thumb draws the line at roughly twenty features — and why a
wide one-hot encoding (Session 2, "What one-hot costs") hurts it. Small $k$
overfits, large $k$ underfits, and neither fixes the dimension. Keep it as a
sanity check on a few scaled features, not a submission.

---

## Support vector machines

- Find the hyperplane that **maximises the margin** between classes
- Margin = distance between the hyperplane and the nearest data points
- Those nearest points are the **support vectors** — they alone define the
  boundary

![An SVM and a logistic regression on the same two classes](assets/tabular/svm-vs-logistic.png)

Both draw a line. Logistic regression puts it where the likelihood is highest;
the SVM puts it in the middle of the widest empty corridor it can find.

---

## Linear SVM: the maths

$$
\mathrm{Hyperplane:} \quad w^T x + b = 0
\qquad
\mathrm{Decision:} \quad \hat{y} = \mathrm{sign}(w^T x + b)
$$

With labels $y_i \in \{-1, +1\}$, the distance from a point $x_0$ to the
hyperplane is

$$
d(x_0) = \frac{|w^T x_0 + b|}{\|w\|}
$$

---

## The margin

![The margin: two dashed lines through the support vectors, 2/‖w‖ apart](assets/tabular/svm-margin.png)

The figure writes the hyperplane as $w \cdot x - b = 0$: the same thing with $b$
negated.

---

## Fixing the scale

$(w, b)$ and $(cw, cb)$ describe the same hyperplane for any $c > 0$, so the
scale is ours to choose. Choose it so the support vectors sit at $\pm 1$:

$$
w^T x_+ + b = +1 \;\Longrightarrow\; d_+ = \frac{|+1|}{\|w\|} = \frac{1}{\|w\|}
$$

$$
w^T x_- + b = -1 \;\Longrightarrow\; d_- = \frac{|-1|}{\|w\|} = \frac{1}{\|w\|}
$$

$$
\mathrm{margin} = d_+ + d_- = \frac{1}{\|w\|} + \frac{1}{\|w\|} = \frac{2}{\|w\|}
$$

---

## Maximum margin is minimum $\|w\|$

Maximising the margin is therefore minimising $\|w\|$:

$$
\max \frac{2}{\|w\|} \quad \Longleftrightarrow \quad \min \frac{1}{2}\|w\|^2
$$

$$
\min_{w, b} \frac{1}{2}\|w\|^2 \quad \mathrm{s.t.} \quad y_i(w^T x_i + b) \geq 1 \quad \forall i
$$

The constraint puts every point on its correct side, at least as far out as the
support vectors. Squaring and halving leave the minimiser unchanged and make the
problem a convex quadratic: one global optimum, no local minima.

---

## Soft margin

Real data is not perfectly separable, so allow violations $\xi_i \geq 0$ and
charge for them:

$$
\min_{w, b, \xi} \; \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

$$
\mathrm{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \qquad \xi_i \geq 0
$$

$\xi_i = 0$ for a point outside the margin, between 0 and 1 for a point inside
it on the correct side, above 1 for a misclassified point.

---

## The hinge loss

At the optimum each slack is as small as its constraint allows:

$$
\xi_i = \max\big(0, \; 1 - y_i(w^T x_i + b)\big)
$$

This is the **hinge loss**: zero for a point outside the margin, growing
linearly once it crosses in. Substituting it gives the soft-margin SVM the
shape of every penalised model in *The Tabular Landscape*, loss plus penalty:

$$
\min_{w, b} \; C \sum_{i=1}^{n} \max\big(0, \; 1 - y_i(w^T x_i + b)\big) + \frac{1}{2}\|w\|^2
$$

with $C$ in the place of $1/\lambda$.

---

## Hard and soft margin

![Hard margin versus soft margin, with the points that violate it](assets/tabular/hard-vs-soft-margin.png)

Left, no point may enter the margin. Right, a few points violate it, each
paying its $\xi_i$ in the objective.

---

## C: margin against errors

$C$ is the margin-versus-misclassification trade-off:

| $C$ | Effect |
|---|---|
| Small $C$ | wide margin, more errors allowed |
| Large $C$ | narrow margin, fewer errors tolerated |

The same `C` as `LogisticRegression`, for the same reason: it multiplies the
loss, so larger `C` means less regularisation. Only the loss differs — the
log-loss there, the hinge here.

---

## The kernel trick

The circles of the previous lesson needed a lift into a higher dimension. Doing
that explicitly through $\phi(x)$ is expensive. But the SVM, solved in its dual
form, only ever needs *inner products* between points — so replace them:

$$
K(x, x') = \phi(x)^T \phi(x')
$$

and never compute $\phi(x)$ at all. Every inner product becomes $K(x_i, x_j)$:
a non-linear boundary, without ever building the high-dimensional features.

---

## Lifting with a kernel

![A kernel lifts the inner class above a flat decision surface](assets/tabular/kernel-lift.png)

The circles of *The Tabular Landscape*, lifted: a flat decision surface in the
lifted space is a closed curve around the inner class back in the plane. The
kernel computes inner products in that space without ever visiting it.

---

## Kernels you will meet

| Kernel | Formula | In scikit-learn |
|---|---|---|
| Linear | $K(x, x') = x^T x'$ | `kernel="linear"` |
| Polynomial | $K(x, x') = (x^T x' + c)^d$ | `kernel="poly"` |
| RBF (Gaussian) | $K(x, x') = \exp(-\gamma \Vert x - x' \Vert^2)$ | `kernel="rbf"`, the default |

The RBF kernel corresponds to an *infinite-dimensional* $\phi$, which you could
never compute directly and never need to.

In `SVC`, $d$ is `degree`, $c$ is `coef0`, and the polynomial kernel also
multiplies $x^T x'$ by `gamma`. The default `gamma="scale"` sets
$\gamma = 1 / (p \cdot \mathrm{Var}(X))$, the variance taken over every entry
of $X$.

---

## γ: the kernel's reach

In the RBF kernel, $\gamma$ sets how far a support vector's influence reaches:

| $\gamma$ | Effect |
|---|---|
| Large $\gamma$ | each support vector influences only its immediate neighbourhood; the boundary wraps individual points — overfitting |
| Small $\gamma$ | every point influences everything; the boundary flattens towards linear — underfitting |

Search $\gamma$ on a log scale, jointly with $C$: the two interact, and
*Hyperparameter Optimisation* searches them together.

---

## SVMs in scikit-learn

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

svm_linear = SVC(kernel="linear", C=1.0)
svm_rbf    = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_rbf.fit(X_train_s, y_train)
y_pred = svm_rbf.predict(X_test_s)
```

> **SVMs need scaled features** — like kNN, for the same reason: the kernel is a
> function of distances and inner products.

---

## Regression: SVR

```python
from sklearn.svm import SVR
svr_rbf = SVR(kernel="rbf", C=1.0, gamma="scale")
svr_rbf.fit(X_train_s, y_train)
y_pred = svr_rbf.predict(X_test_s)
```

`SVR` fits a tube around the function and charges only for points more than
$\varepsilon$ from it (`epsilon=0.1` by default). `C` and the kernels mean what
they mean in `SVC`.

---

## Why the SVM lost the tabular crown

Training is quadratic to cubic in rows. scikit-learn's own documentation says
`SVC`'s fit time scales at least quadratically with the number of samples and
may be impractical beyond tens of thousands; above roughly 50,000 rows it stops
being practical. That, not accuracy, is why it lost the tabular crown.

On a large table where a linear boundary is enough, `LinearSVC` or
`SGDClassifier` scale. Otherwise, gradient boosting. Below about 1,000 rows, a
small SVM is still a reasonable second model after the regularised linear
baseline.

---

## Check yourself

1. A 1-nearest-neighbour classifier scores 100% accuracy on its own training
   set. What does that number tell you?

   **Answer.** Nothing about new data. Every training point is its own nearest
   neighbour, at distance zero, so it gets its own label back: memory, not
   learning. Choose $k$ on a validation set.

2. Run this. You should get exactly the output shown.

   ```python
   from sklearn.datasets import load_wine
   from sklearn.model_selection import cross_val_score
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   X, y = load_wine(return_X_y=True)
   knn = KNeighborsClassifier(n_neighbors=5)
   scaled = make_pipeline(StandardScaler(), knn)
   a = cross_val_score(knn, X, y, cv=5).mean()
   b = cross_val_score(scaled, X, y, cv=5).mean()
   print(f"{a:.2f} {b:.2f}")        # -> 0.69 0.95
   ```

   Proline, from 278 to 1,680, swamps every other column: unscaled, "nearest"
   means "similar proline".

3. Your table has two million rows and twelve features, and a colleague
   proposes an RBF SVM. What is the objection?

   **Answer.** Cost, not accuracy. Kernel SVM training is quadratic to cubic in
   rows and impractical beyond tens of thousands. Use gradient boosting, or
   `LinearSVC` if a linear boundary is enough.
