# Models

A model is a parameterised function (non parametric models have also parameters). Choosing one means choosing its input
space, its output space, and how many numbers it is allowed to learn — all three
before you have seen a row of data.

<!-- notes: 30 minutes. The parameter-count figures are the spine of this
lesson: same template, more θ. The parametric/non-parametric split and the
parameter/hyperparameter split are the two distinctions worth the time. -->

---

## Definition

A model is a map from a feature vector to a prediction, indexed by a parameter
vector:

$$
f_\theta : \mathbb{R}^p \longrightarrow \mathcal{Y}
\qquad
\theta \in \mathbb{R}^{d}
\qquad
\hat{y}_i = f_\theta(x_i)
$$

Three things are fixed the moment you name a model, and only the third is
learned:

| | | |
|---|---|---|
| **Input** | $\mathbb{R}^p$ | the feature space — $p$ is set by your data |
| **Output** | $\mathcal{Y}$ | $\mathbb{R}$ for regression, $[0,1]$ or $\Delta^{K-1}$ for classification |
| **Parameters** | $\theta \in \mathbb{R}^d$ | the $d$ numbers training has to find |

The hat on $\hat{y}$ means *estimated*. A **model family** $\mathcal{F} =
\{f_\theta : \theta \in \mathbb{R}^d\}$ is the set of functions you are willing
to consider; training picks one element of it.

---


## The output space is a modelling decision

It is the constraint that does the most work for the least effort.

| Task | $\mathcal{Y}$ | Why |
|---|---|---|
| Regression | $\mathbb{R}$ | unbounded, a price or a mass |
| Regression, positive quantity | $\mathbb{R}_{+}$ | often via $\log y$ |
| Binary classification | $[0, 1]$ | read as $P(y = 1 \mid x)$ |
| $K$-class classification | $\Delta^{K-1}$ | the simplex: $K$ probabilities summing to 1 |

---

## Size: $d$ is the number of values the model must learn

Picking a family fixes $d$ before you see any data. The five that follow are the
same template with more $\theta$.

---

## Linear — 2 parameters

![Linear functions](/api/academic_courses/assets/lessons/167/family-linear.png)

$\mathbb{R} \rightarrow \mathbb{R}$, with a slope $a$ and an intercept $b$.

---

## Polynomial of degree 2 — 3 parameters

![Polynomial functions](/api/academic_courses/assets/lessons/167/family-polynomial.png)

$\mathbb{R} \rightarrow \mathbb{R}$, with $a$, $b$ and $c$. One more parameter
buys one bend.

---

## Logistic — 2 parameters, and a bounded output

![Logistic functions](/api/academic_courses/assets/lessons/167/family-logistic.png)

$\mathbb{R} \rightarrow [0, 1]$. Note the codomain: this one *cannot* output
anything outside $[0,1]$, which is exactly why it is used for probabilities.

---

## Quadratic surface — 5 parameters

![Quadratic surfaces](/api/academic_courses/assets/lessons/167/family-quadratic-surface.png)

$\mathbb{R}^2 \rightarrow \mathbb{R}$: two inputs, one output, five numbers to
learn.

---

## A small network — 47 parameters

![A small neural network](/api/academic_courses/assets/lessons/167/neural-network-diagram.png)

$\mathbb{R}^6 \rightarrow \mathbb{R}$. Modern ones have $10^9$ or more, and
nothing about the template has changed.

---


## The interface

Two different models, one import, one prediction call — the class name is the
only thing that changes:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

reg = LinearRegression()      # regression:     y in R
clf = LogisticRegression()    # classification: y in {0, 1}

y_hat = reg.predict(X_test)   # real values
y_cls = clf.predict(X_test)   # labels
y_cls = clf.predict_proba(X_test)   # y in [0, 1]
```

Choosing the class is choosing $f_\theta$; `predict` applies it. Run this as
is and `scikit-learn` refuses — the object has no $\theta$ yet.
