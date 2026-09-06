# Models

A model is a parameterised function. Choosing one means choosing its input
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

## Training picks one function out of the family

![Fitting a function to data](/api/academic_courses/assets/lessons/167/regression.png)

Every curve the family allows is a candidate. The data and the loss decide which
one you end up with.

---

## The output space is a modelling decision

It is the constraint that does the most work for the least effort.

| Task | $\mathcal{Y}$ | Why |
|---|---|---|
| Regression | $\mathbb{R}$ | unbounded, a price or a mass |
| Regression, positive quantity | $\mathbb{R}_{+}$ | often via $\log y$ |
| Binary classification | $[0, 1]$ | read as $P(y = 1 \mid x)$ |
| $K$-class classification | $\Delta^{K-1}$ | the simplex: $K$ probabilities summing to 1 |

A model whose codomain is $\mathbb{R}$ can return $-0.3$ for a probability. You
can either penalise that after the fact, or pick a family that cannot do it —
which is the whole argument of the logistic regression lesson.

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

## What $d$ costs you

In general, a linear model on $p$ features has $d = p + 1$: one coefficient per
feature, plus the intercept. Capacity grows with $d$, and so does the amount of
data you need to pin it down.

---

## Parametric and non-parametric

**Parametric** — $d$ is fixed in advance and does not grow with $n$.

$$
f_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_p x_p = \theta^\top x
$$

Training means finding $\theta$. The fitted model is small, fast to evaluate,
and can be stored without the data. Linear and logistic regression, neural
networks.

**Non-parametric** — the structure grows with the data.

- **$k$-nearest neighbours** — keeps the entire training set; there is no $\theta$
- **Decision trees** — depth and shape determined by the data
- **Random forests, gradient boosting** — ensembles of such trees

More flexible, and more prone to memorising. They typically need more data and
more careful validation. All three are Session 3.

---

## Which to reach for

| Situation | Reach for |
|---|---|
| Tabular, moderate size | gradient boosting (XGBoost / LightGBM) |
| Tabular, need to explain it | linear / logistic regression |
| Images, audio | pretrained neural network |
| Text | pretrained transformer |
| Very little data | the simplest model you can defend |

On tabular data, gradient boosting still beats deep networks most of the time.
That is not a slogan, it is the empirical result you should assume until your
own validation says otherwise.

---

## Parameters are learned. Hyperparameters are chosen.

$\theta$ is found by training. Everything that *configures* the training — or
fixes $d$ in the first place — is a **hyperparameter**, and you set it.

| Model | Hyperparameters |
|---|---|
| Linear / logistic regression | regularisation strength, penalty type |
| Polynomial regression | the degree — which is to say, $d$ itself |
| Decision tree | max depth, min samples per leaf |
| Gradient boosting | learning rate, number of trees, depth |
| Neural network | architecture, learning rate, batch size |

![Hyperparameter search](/api/academic_courses/assets/lessons/167/hyperparam.png)

They cannot be learned by the same procedure that learns $\theta$ — minimising
training error over the polynomial degree just returns the largest degree
available. They are chosen on a **validation** set, which is the opening of
Session 3.

---

## The interface

Every model in `scikit-learn` exposes the same two methods, and the split is
exactly the parametric one: `fit` searches for $\theta$, `predict` applies
$f_\theta$.

```python
model.fit(X_train, y_train)       # training:   find theta
y_pred = model.predict(X_test)    # prediction: apply f_theta
```

For classification, ask for the probability rather than the label where you can
— you can always threshold afterwards, and you cannot recover a probability from
a label:

```python
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba > 0.5).astype(int)
```

`0.5` is a **choice**, not part of the model.
