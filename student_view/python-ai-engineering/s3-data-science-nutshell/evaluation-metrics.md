# Evaluation Metrics

The metric is the definition of "better". Choose it before you model, and
choose it for the problem rather than for convenience.

<!-- notes: 40 minutes. The imbalanced-accuracy example is the one they
remember. Do it with real numbers on the board. -->

---

## Regression — MSE

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

Squaring means large errors dominate. Units are the square of the target's,
which is why RMSE ($\sqrt{MSE}$) is often reported instead.

---

## Regression — MAE

$$
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

Linear in the error, so an outlier counts once rather than enormously. Same
units as the target, so it is directly interpretable.

---

## MSE or MAE?

Predictions `[10, 12, 100]` against truth `[10, 12, 20]` — one bad prediction:

| Metric | Value |
|---|---|
| MAE | $(0 + 0 + 80)/3 = 26.7$ |
| MSE | $(0 + 0 + 6400)/3 = 2133$ |

Ask whether one error of 80 is as bad as eighty errors of 1.

- **Yes** → MAE
- **Much worse** → MSE

For a hospital bed forecast, one huge miss is catastrophic: MSE. For a delivery
time estimate, consistency matters more: MAE.

---

## MAPE — relative error

$$
MAPE = \frac{1}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100
$$

Percentages, so it compares across scales. Undefined at $y_i = 0$ and unstable
near it — check your target before reaching for it.

---

## The confusion matrix

Everything in binary classification comes from four counts.

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP | FN |
| **Actually negative** | FP | TN |

```python
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
```

---

## Accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

The fraction you got right. Intuitive, and misleading exactly when it matters
most.

---

## Why accuracy fails

A disease affecting 1% of the population. This model:

```python
def predict(x):
    return 0        # "nobody is ill"
```

scores **99% accuracy** and finds zero patients.

Whenever the classes are imbalanced — fraud, disease, defects, churn — accuracy
measures the majority class and nothing else.

---

## Precision

$$
Precision = \frac{TP}{TP + FP}
$$

Of everything you flagged, how much was real?

```python
from sklearn.metrics import precision_score
precision_score(y_true, y_pred)
```

High precision = few false alarms. Optimise it when acting on a positive is
expensive: sending a technician, blocking a transaction.

---

## Recall

$$
Recall = \frac{TP}{TP + FN}
$$

Of everything that was real, how much did you find?

```python
from sklearn.metrics import recall_score
recall_score(y_true, y_pred)
```

High recall = few misses. Optimise it when missing a positive is expensive:
cancer screening, safety faults.

---

![Precision and recall](/api/academic_courses/assets/lessons/42/Precisionrecall.png)

---

## They trade off

Lower the decision threshold and you catch more positives — and more false
alarms. Raise it and the reverse.

| Threshold | Precision | Recall |
|---|---|---|
| 0.9 | high | low |
| 0.5 | medium | medium |
| 0.1 | low | high |

There is no threshold that maximises both. Which side you lean to is a decision
about the *problem*, not about the model.

---

## F1

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

The harmonic mean, which — unlike the arithmetic mean — is dragged down by the
weaker of the two. Precision 1.0 with recall 0.0 gives F1 = 0, which is the
honest answer.

```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred)
```

Use it when you need one number and both errors matter.

---

## ROC AUC

Plot true positive rate against false positive rate across **every** threshold;
the area underneath is the AUC.

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_proba)     # probabilities, not labels
```

- 1.0 — perfect ranking
- 0.5 — no better than chance

![ROC curve](/api/academic_courses/assets/lessons/42/roc_auc.png)

---

## Reading AUC correctly

AUC measures **ranking**, not calibration or classification. It answers: given a
random positive and a random negative, how often does the model rank the
positive higher?

It is threshold-free, which makes it good for comparing models — and it stays
optimistic on heavily imbalanced data, where **precision–recall AUC** is the more
honest choice.

---

## Multi-class averaging

With $K$ classes you have $K$ precision and recall values. How you combine them
matters:

```python
f1_score(y_true, y_pred, average="macro")     # unweighted class mean
f1_score(y_true, y_pred, average="weighted")  # weighted by class support
f1_score(y_true, y_pred, average="micro")     # global TP/FP/FN pooling
```

- **macro** — every class counts equally; rare classes can dominate
- **weighted** — frequent classes dominate
- **micro** — equals accuracy in the single-label case

Say which one you used. A reported "F1 = 0.82" without the averaging is not a
number anyone can compare against.

---

## Ranking metrics

When the output is an ordered list rather than a label:

- **MAP** — mean average precision; rewards putting relevant items early
- **NDCG** — discounted cumulative gain, normalised; handles graded relevance

Used for search, recommendation, and retrieval.

---

## Choosing

| Situation | Metric |
|---|---|
| Balanced classification | accuracy |
| Imbalanced classification | F1, PR-AUC |
| Missing a positive is costly | recall |
| A false alarm is costly | precision |
| Comparing models, threshold-free | ROC AUC |
| Regression, outliers matter | MSE / RMSE |
| Regression, outliers are noise | MAE |
| Errors are relative | MAPE |

---

## The rule

> Fix the metric **before** you look at the results.

Choosing the metric after seeing which one flatters your model is how you fool
your supervisor, then your users, then yourself. On ML-Arena the metric is fixed
by the competition — which is exactly the discipline the leaderboard is
enforcing.
