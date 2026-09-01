# Lab 3 — Baseline on a Real Dataset

Build a complete, leak-free pipeline on a dataset you have not seen. The model
quality matters less than the honesty of the number you report.

**Time:** 45 minutes. **Deliverable:** a notebook or script in your repository,
plus a written `RESULTS.md`.

---

## The dataset

Use the UCI Adult (Census Income) dataset — binary classification, tabular,
imbalanced (~24% positive), with a mix of numeric and categorical features.

```python
from sklearn.datasets import fetch_openml

data = fetch_openml("adult", version=2, as_frame=True)
X, y = data.data, (data.target == ">50K").astype(int)
```

If you prefer another dataset, it must be tabular, imbalanced, and have at least
one categorical column. Clear it with your instructor.

---

## Part A — Frame it (5 min)

Write the top of `RESULTS.md` **before modelling**:

```markdown
## Problem
Predicting: <what, exactly>
Available at prediction time: <which features, and why they qualify>

## Metric
Chosen metric: <one metric>
Why this one: <one sentence tied to the cost of each error type>

## Baseline
Strategy: <e.g. always predict the majority class>
Expected score: <the number>
```

You may not change the metric after seeing any result.

---

## Part B — Baseline (5 min)

```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dummy = DummyClassifier(strategy="most_frequent")
print(cross_val_score(dummy, X, y, cv=cv, scoring="f1").mean())
```

Record it. Every later number is measured against this one.

Note what accuracy the dummy achieves, and put that in `RESULTS.md` too — it is
the argument for your metric choice.

---

## Part C — A leak-free pipeline (15 min)

Requirements:

- **All** preprocessing inside a `Pipeline` — no `fit_transform` on the full `X`
- Numeric: impute + scale. Categorical: impute + one-hot encode
- Stratified 5-fold cross-validation
- Report mean **and** standard deviation

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

num = X.select_dtypes("number").columns
cat = X.select_dtypes(exclude="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler())]), num),
    ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
])

pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
```

---

## Part D — Beat the baseline (10 min)

Train a second model — gradient boosting is the obvious choice — and compare.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
```

Add to `RESULTS.md`:

| Model | Metric (mean ± std) | Beats baseline? |
|---|---|---|
| Dummy | | — |
| Logistic regression | | |
| Gradient boosting | | |

---

## Part E — Threshold (10 min)

Your metric depends on a threshold that defaults to `0.5`. Find a better one on
the **validation** folds:

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
import numpy as np

proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
for t in np.arange(0.1, 0.9, 0.05):
    print(f"{t:.2f}  {f1_score(y, proba > t):.3f}")
```

Report the best threshold and the score it gives. State in one sentence what
moving it did to precision and recall.

---

## Grading

| Criterion | Weight |
|---|---|
| Metric chosen and justified **before** results | 20% |
| Baseline measured and reported | 15% |
| Zero leakage: all preprocessing inside the Pipeline | 30% |
| Stratified CV, mean **and** spread reported | 20% |
| Threshold analysis with a stated tradeoff | 15% |

---

## Automatic fail

Any of these zeroes the leakage component:

- `fit` or `fit_transform` called on the full `X`
- an unstratified split on this imbalanced target
- a reported score from a metric you changed after seeing results

---

## If you finish early

Add one engineered feature you can defend, and show whether it helps. "It
improved by 0.001" is a real answer — most engineered features do nothing, and
noticing that is the skill.
