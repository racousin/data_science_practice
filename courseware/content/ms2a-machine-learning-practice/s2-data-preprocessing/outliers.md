# Outliers

The third defect no type check catches. An outlier changes your metric more
than a duplicate or a misspelt category does, and it is the one where "clean the
data" most often means "delete the evidence" — a value you cannot explain is a
claim about how the data was generated, not a number to tidy away.

<!-- notes: 15 minutes. Open with the three kinds and insist that the standard
detectors only find the first. Linger on the pair slide — two values each
inside their own fences, unusual only together — because that is the case a
column-by-column check never sees. Close on the logging rule: a pipeline that
prints how many rows it dropped, and why. -->

---

## Outliers: three kinds

- **Point** — a single value far from the distribution
- **Contextual** — normal in general, impossible here: 25 °C in Oslo in January
- **Collective** — no single point is extreme, the *pattern* is

![A point outlier in a scatter plot](assets/preprocessing/point-outlier-scatter.png)

The first is what the standard detectors find. The second and third need domain
knowledge, and a detector that flags them is usually flagging the wrong rows.

---

## Same statistics, four datasets

![Anscombe's quartet: four scatter plots with identical summary statistics and the same fitted line](assets/preprocessing/anscombes-quartet.png)

Same mean, variance, correlation and fitted line in all four. Bottom left, one
point tilts the line; bottom right, one point *is* the line.

---

## Detecting point outliers

Z-score, for roughly symmetric data:

$$
z_i = \frac{x_i - \bar{x}}{s}
$$

```python
z = (df["revenue"] - df["revenue"].mean()) / df["revenue"].std()
outliers = df[z.abs() > 3]
```

The mean and the standard deviation are themselves moved by the outliers, so on
a heavy-tailed column this under-detects. The IQR rule is not:

```python
q1, q3 = df["revenue"].quantile([0.25, 0.75])
iqr = q3 - q1
mask = df["revenue"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
```

---

## IQR fences and σ

![IQR on a boxplot](assets/preprocessing/iqr-fences-on-normal-distribution.png)

On normal data the 1.5 × IQR fences fall at ±2.698σ, just inside the |z| > 3 cut-off.

---
## Outliers only as a pair

![Seating capacity against revenue](assets/preprocessing/seating-capacity-vs-revenue-joint-outliers.png)

The low-revenue cluster and the 36-seat restaurant near 1.2M sit inside the IQR
fences of both columns; only the combination is unusual.

---

## Multivariate outliers

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.01, random_state=0)
flags = iso.fit_predict(X_num)
```

A 40-year-old is normal. A 40-year-old with 45 years of professional experience
is not, and no single-column rule sees it. Isolation Forest isolates points with
random splits and scores how few splits it takes.

---

## Every detector assumes a shape

![Five outlier detectors on five toy datasets, each drawing a different boundary around the same inliers](assets/preprocessing/anomaly-detection-comparison.png)

Five boundaries around the same points. `contamination` is an assumption you are
making, not one the algorithm discovers — state it, and check what it flagged.

---





## Remove, clip, or keep

| Situation | Action |
|---|---|
| Physically impossible (age 300, negative price) | remove, and fix the source |
| Genuine extreme, model is linear or distance-based | clip to a percentile |
| Genuine extreme, model is a tree | keep — it splits around it |
| The extreme *is* the target (fraud, failure, churn) | keep, obviously |
| Heavy right tail across the column | transform, do not clip |

```python
lo, hi = df["revenue"].quantile([0.01, 0.99])
df["revenue"] = df["revenue"].clip(lo, hi)
```

Clipping bounds the influence without inventing a value. Compute `lo` and `hi`
on the training rows only — they are fitted parameters like any other.
