
## Support vector machines

- Find the hyperplane that **maximises the margin** between classes
- Margin = distance between the hyperplane and the nearest data points
- Those nearest points are the **support vectors** — they alone define the
  boundary


![1_nUpw5agP-Vefm4Uinteq-A.png](assets/tabular/1_nUpw5agP-Vefm4Uinteq-A.png)


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
