# Training

The $\arg\min$ of the previous lesson is a wish, not an algorithm. For all but a
handful of families there is no formula for $\theta^{*}$ — you have to walk
downhill towards it. This is the optimiser that trains almost everything you
will use this year.

<!-- notes: 40 minutes. The lesson Session 4 builds on directly. If time is
short, cut the optimiser variants, not the learning-rate figures — students who
have never watched a run diverge do not believe in it. -->

---

## Two routes to $\theta^{*}$

$$
\theta^{*} = \arg\min_{\theta \in \mathbb{R}^{d}} \ \ell(\theta)
$$

| | | |
|---|---|---|
| **Analytical** | set $\nabla_\theta \ell = 0$ and solve | exact, but only for a few $(f, \ell)$ pairs — linear regression with MSE is the one you will meet |
| **Iterative** | start somewhere, step downhill, repeat | always available, needs only that $\ell$ be differentiable |

Everything from logistic regression onward takes the second route.

---

## Derivatives and gradients

The **gradient** collects the partial derivatives of a scalar function of
several variables:

$$
\nabla_\theta \ell = \left[ \frac{\partial \ell}{\partial \theta_1}, \ \frac{\partial \ell}{\partial \theta_2}, \ \ldots, \ \frac{\partial \ell}{\partial \theta_d} \right]^\top
$$

It points in the direction of **steepest increase**, so $-\nabla_\theta \ell$ is
the direction of steepest decrease. That single fact is the whole algorithm.

![A function and its derivative](/api/academic_courses/assets/lessons/169/derivative-and-function.png)

Where the derivative vanishes, the function is flat — a minimum, a maximum, or a
saddle. Every optimiser is chasing $\nabla_\theta \ell = 0$; none of them can
tell you which of the three it found.

---

## The algorithm

**0. Initialise** the parameters, usually at random:

$$
\theta_0
$$

**1. Step** against the gradient:

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \, \ell(\theta_t)
$$

**2. Iterate** until the loss stops falling.

$\eta > 0$ is the **learning rate**. In three lines:

```python
for _ in range(n_steps):
    theta -= eta * grad(theta)
```

![Gradient descent with a well-chosen learning rate](/api/academic_courses/assets/lessons/169/gd-optimal-lr.png)

---

## The learning rate decides whether it works

$\eta$ is a hyperparameter — you choose it, training does not. Get it wrong in
either direction and the run fails, in two very different ways.

![Learning rate too small](/api/academic_courses/assets/lessons/169/gd-small-lr.png)

Too small ($\eta = 0.02$) — it converges, but so slowly you run out of budget.

![Learning rate too large](/api/academic_courses/assets/lessons/169/gd-large-lr.png)

Too large ($\eta = 0.95$) — it oscillates across the valley and can diverge. On
a quadratic $\ell(\theta) = \theta^2$ the step is $\theta \mapsto (1 - 2\eta)
\theta$, so the iteration contracts exactly when $\eta < 1$ and the magnitude
grows every step beyond it. A loss that goes to `nan` after a few epochs is
almost always this; divide $\eta$ by 10.

---

## Convexity is the guarantee

![A non-convex loss surface](/api/academic_courses/assets/lessons/169/gd-non-convex.png)

On a **convex** loss, every local minimum is global, and gradient descent with a
small enough $\eta$ converges to it regardless of where it started. MSE with a
linear model, and cross-entropy with a logistic model, are both convex in
$\theta$ — which is why the two models in this session are safe.

Drop convexity and the guarantee goes with it: you reach *a* stationary point,
and which one depends on the initialisation. The run above starts on the wrong
side of a hill and settles in a local minimum with the global one untouched.
Neural network losses are non-convex, and Session 4 is spent making that work
anyway.

---

## How much data per step

The loss is a sum over observations, so its gradient is too — and you may
evaluate that sum on any subset you like.

| | Gradient computed on | Cost per step | Behaviour |
|---|---|---|---|
| **Batch** | all $n$ observations | high | smooth, exact descent direction |
| **Stochastic (SGD)** | 1 observation | tiny | very noisy, cheap, escapes shallow minima |
| **Mini-batch** | $B$ observations, typically 32–512 | tuneable | the default everywhere |

The noise in mini-batch gradients is not purely a cost: it is unbiased, and it
helps the iterate leave narrow minima. Mini-batch is what PyTorch does in
Session 4.

---

## Optimisers: better use of the same gradient

Plain descent uses only the current gradient. **Momentum** accumulates a running
mean, so steps persist through flat regions and oscillation cancels out:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla \ell(\theta_t)
$$

**Adam** keeps that and a running second moment, which rescales each coordinate
by its own recent gradient magnitude:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \big(\nabla \ell(\theta_t)\big)^2
\qquad
\theta_{t+1} = \theta_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

![SGD versus SGD with momentum](/api/academic_courses/assets/lessons/169/gd-momentum.gif)

Adam is the default you will reach for in PyTorch next session.

---

## Worked example: fitting a line

Nine temperature observations, the two-parameter model $\hat{y} = ax + b$, MSE
loss. Start from a deliberately bad initialisation:

![Initial fit](/api/academic_courses/assets/lessons/169/fit-init-line.png)

| Year | `y_true` | `y_pred` | Error |
|---|---|---|---|
| 1980 | 14.18 | 14.80 | 0.62 |
| 1985 | 14.10 | 14.75 | 0.65 |
| 1990 | 14.35 | 14.70 | 0.35 |
| 1995 | 14.38 | 14.65 | 0.27 |
| 2000 | 14.33 | 14.60 | 0.27 |
| 2005 | 14.55 | 14.55 | 0.00 |
| 2010 | 14.58 | 14.50 | 0.08 |
| 2015 | 14.80 | 14.45 | 0.35 |
| 2020 | 14.92 | 14.40 | 0.52 |

Take the gradient of that total error with respect to $a$ and $b$, step, repeat:

![After updates](/api/academic_courses/assets/lessons/169/fit-updated-line.png)

![Parameters over 50 steps](/api/academic_courses/assets/lessons/169/parameter-evolution.png)

![Total absolute error over 50 steps](/api/academic_courses/assets/lessons/169/error-evolution.png)

The error falls from 3.45 to 0.60 in fifteen steps and then flattens — the shape
of essentially every training curve you will ever plot.

| | Initial | Trained |
|---|---|---|
| Parameters | $a = -0.01$, $b = 34.70$ | $a = 0.019$, $b = -23.47$ |
| Prediction for 2030 | $14.4$ | $15.1$ |

The initial model predicts cooling, the trained one warming. Nothing changed
except 50 gradient steps.
