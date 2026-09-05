# Support Vector Machines

A different objective: not "minimise the average error" but "put the boundary as
far from both classes as possible". And a trick that buys non-linearity for
almost nothing.

<!-- notes: ~10 minutes. The margin derivation is short and worth doing in full —
it is the only place in the course where the geometry produces the objective. -->

---

## The idea

- Find the hyperplane that **maximises the margin** between classes
- Margin = distance between the hyperplane and the nearest data points
- Those nearest points are the **support vectors** — they alone define the
  boundary

![SVM versus logistic regression](assets/s3-models-and-tuning/support-vector-machines/svm-vs-logistic.png)

Both draw a line. Logistic regression puts it where the likelihood is highest;
SVM puts it in the middle of the widest empty corridor it can find.

---

## Linear SVM: the maths

$$
\text{Hyperplane:} \quad w^T x + b = 0
\qquad
\text{Decision:} \quad \hat{y} = \text{sign}(w^T x + b)
$$

With labels $y_i \in \{-1, +1\}$, the distance from a point to the hyperplane is

$$
d(x_0) = \frac{|w^T x_0 + b|}{\|w\|}
$$

![The margin](assets/s3-models-and-tuning/support-vector-machines/svm-margin.png)

Fix the scale so the support vectors sit at $\pm 1$:

$$
w^T x_+ + b = +1 \;\Longrightarrow\; d_+ = \frac{|+1|}{\|w\|} = \frac{1}{\|w\|}
$$

$$
w^T x_- + b = -1 \;\Longrightarrow\; d_- = \frac{|-1|}{\|w\|} = \frac{1}{\|w\|}
$$

$$
\text{margin} = d_+ + d_- = \frac{1}{\|w\|} + \frac{1}{\|w\|} = \frac{2}{\|w\|}
$$

Maximising the margin is therefore minimising $\|w\|$:

$$
\max \frac{2}{\|w\|} \quad \Longleftrightarrow \quad \min \frac{1}{2}\|w\|^2
$$

$$
\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 \quad \forall i
$$

---

## Soft margin

Real data is not perfectly separable, so allow violations $\xi_i$ and charge for
them:

$$
\min_{w, b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

![Hard versus soft margin](assets/s3-models-and-tuning/support-vector-machines/hard-vs-soft-margin.png)

$C$ is the margin-versus-misclassification trade-off:

| | |
|---|---|
| Small $C$ | wide margin, more errors allowed |
| Large $C$ | narrow margin, fewer errors tolerated |

Compare with `LogisticRegression`, where `C` means the same thing and runs the
same way: larger `C`, less regularisation.

---

## The kernel trick

The circles dataset from the non-linearity lesson needed a lift into a higher
dimension. Doing that explicitly through $\phi(x)$ is expensive. Instead, note
that the SVM only ever needs *inner products* between points — so replace them:

$$
K(x, x') = \phi(x)^T \phi(x')
$$

and never compute $\phi(x)$ at all.

![Lifting with a kernel](assets/s3-models-and-tuning/support-vector-machines/kernel-lift.png)

| Kernel | Formula |
|---|---|
| Linear | $K(x, x') = x^T x'$ |
| Polynomial | $K(x, x') = (x^T x' + c)^d$ |
| RBF (Gaussian) | $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ |

The RBF kernel corresponds to an *infinite-dimensional* $\phi$, which you could
never compute directly and never need to.

---

## In scikit-learn

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

svm_linear = SVC(kernel='linear', C=1.0)
svm_rbf    = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_s, y_train)
y_pred = svm_rbf.predict(X_test_s)
```

```python
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_rbf.fit(X_train_s, y_train)
y_pred = svr_rbf.predict(X_test_s)
```

> **SVM needs scaled features** — like KNN, for the same reason: the kernel is a
> function of distances and inner products.

---

## Pros and cons

| Pros | Cons |
|---|---|
| Effective in high dimensions | Slow for large datasets: $O(n^2)$ to $O(n^3)$ |
| Memory-efficient (only support vectors) | Sensitive to feature scaling |
| Flexible via kernel choice | No native probability output |

---

## Models so far

| Model | Type | Parametric? | Scaling? | Key hyperparameters |
|---|---|---|---|---|
| Linear / Ridge / Lasso | Reg / Clf | Yes | Recommended | $\lambda$ |
| KNN | Reg / Clf | No | **Required** | $K$ |
| Decision Tree | Reg / Clf | No | Not needed | `max_depth` |
| SVM | Clf (Reg) | Yes | **Required** | $C$, kernel |
