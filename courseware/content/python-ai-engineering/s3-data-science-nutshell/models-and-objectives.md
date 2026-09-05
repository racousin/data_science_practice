# Models & Objectives

The shared vocabulary. You will meet all of it again in *MS2A - Machine Learning
Practice* — tabular, vision, NLP, reinforcement learning; the point of this lesson
is that none of it is new when you do.

<!-- notes: 35 minutes. Move quickly; this is a levelling lesson, and half the
room already knows it. The parametric/non-parametric distinction and the loss
vs metric distinction are the two that are worth the time. -->

---

## The setup

You have data: $n$ examples, each a pair $(x_i, y_i)$.

- $x_i \in \mathbb{R}^d$ — the **features**, $d$ of them
- $y_i$ — the **target**

You want a function $f$ such that $f(x) \approx y$, including on $x$ you have
never seen. That last clause is the entire difficulty.

---

## Regression vs classification

| | Regression | Classification |
|---|---|---|
| Target | continuous | discrete, from $K$ classes |
| Example | house price | spam / not spam |
| Output | a number | a class, or $K$ probabilities |
| Typical loss | squared error | cross-entropy |

![Regression](assets/ds/regression.png)

---

## Parametric models

A **fixed** number of parameters $\theta$, decided before you see the data.

$$
f_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d
$$

- Training = finding $\theta$
- Model size does not grow with $n$
- Fast to predict, easy to store
- Examples: linear/logistic regression, neural networks

---

## Non-parametric models

Structure grows with the data.

- **k-nearest neighbours** — keeps the entire training set
- **Decision trees** — depth and shape determined by the data
- **Random forests, gradient boosting** — ensembles of such trees

More flexible, and more prone to memorising. They typically need more data and
more careful validation.

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

## Training

Training is optimisation. You define a **loss** measuring how wrong the model
is, and search $\theta$ to make it small:

$$
\theta^* = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n L\big(f_\theta(x_i), y_i\big)
$$

- For linear regression, $L$ is squared error and there is a closed-form answer.
- For neural networks, there is not — you use gradient descent. That is Session 4.

---

## Common losses

| Task | Loss | Formula |
|---|---|---|
| Regression | Mean squared error | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ |
| Regression, outliers | Mean absolute error | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ |
| Binary classification | Binary cross-entropy | $-\frac{1}{n}\sum [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$ |
| Multi-class | Categorical cross-entropy | $-\frac{1}{n}\sum_i \sum_k y_{ik} \log \hat{p}_{ik}$ |

---

## Loss is not metric

This distinction causes more confusion than any other in this session.

| | Loss | Metric |
|---|---|---|
| Used by | the optimiser | you, and the leaderboard |
| Must be | differentiable | anything you can compute |
| Example | cross-entropy | F1, accuracy, AUC |

You optimise cross-entropy because you can differentiate it. You are *judged* on
F1 because that is what the problem cares about. They are not the same number
and they do not always move together.

<!-- notes: The classic consequence: a model with lower loss but worse F1 because
the decision threshold is wrong. Threshold tuning is a metric-side fix. -->

---

## Prediction

Once trained, prediction is a forward pass:

```python
model.fit(X_train, y_train)     # training:   find theta
y_pred = model.predict(X_test)  # prediction: apply f_theta
```

`scikit-learn` gives every model this interface, which is why it is worth
learning once.

---

## Probabilities, not just labels

For classification, ask for probabilities where you can:

```python
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba > 0.5).astype(int)
```

`0.5` is a **choice**, not a law. Moving it trades precision against recall, and
on an imbalanced problem the best threshold is rarely `0.5`. You cannot make
that choice at all if you only kept the labels.

---

## The bias–variance tradeoff

Two ways to be wrong:

- **Bias** — the model is too simple to represent the pattern. Wrong in the same
  direction every time.
- **Variance** — the model is so flexible it fits the noise. A different training
  sample gives a very different model.

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible noise}
$$

Every knob you have — model complexity, regularisation, data quantity — moves
you along this tradeoff.

---

## Hyperparameters

Parameters are learned. **Hyperparameters** are chosen by you, before training.

| Model | Hyperparameters |
|---|---|
| Linear/logistic regression | regularisation strength, penalty type |
| Decision tree | max depth, min samples per leaf |
| Gradient boosting | learning rate, number of trees, depth |
| Neural network | architecture, learning rate, batch size |

![Hyperparameter search](assets/ds/hyperparam.png)

They are chosen on a **validation** set — never the test set. That is the next
lesson.

---

## Check yourself

1. A model with the **lower** cross-entropy loss scores the **worse** F1. Is one
   of the two numbers wrong?

   **Answer.** No. Loss is what the optimiser minimises and must be
   differentiable; the metric is what you and the leaderboard judge on and can be
   anything computable. They do not always move together — the usual culprit is
   the decision threshold, which the loss never sees.

2. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   proba = np.array([0.10, 0.35, 0.48, 0.52, 0.80])
   print((proba > 0.5).astype(int))   # -> [0 0 0 1 1]
   print((proba > 0.4).astype(int))   # -> [0 0 1 1 1]
   ```

   **Answer.** The model did not change between the two lines — only the
   threshold did, and one more example became a positive. `0.5` is a choice, not
   a law, and you cannot make that choice at all if you kept only
   `predict`'s labels instead of `predict_proba`'s probabilities.

3. `max_depth` on a decision tree: parameter or hyperparameter, and which set is
   it chosen on?

   **Answer.** A hyperparameter — you choose it before training, whereas
   parameters are learned by it. It is chosen on the **validation** set, never on
   the test set.

4. You have 4,000 rows of tabular data with 12 mixed numeric and categorical
   columns, and you must be able to explain each prediction to a client. What
   does the "which to reach for" table say, and what is the empirical claim in
   the paragraph under it?

   **Answer.** Logistic regression — "tabular, need to explain it". The claim
   worth remembering from the row above it is that on tabular data gradient
   boosting still beats deep networks most of the time, and you should assume
   that until your own validation says otherwise.
