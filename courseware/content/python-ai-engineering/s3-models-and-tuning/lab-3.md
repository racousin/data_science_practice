# Lab 3 — Baseline on a Real Dataset

Build a complete, leak-free pipeline on a dataset you have not seen. The model
quality matters less than the honesty of the number you report.

**Time:** 45 minutes in class, plus ~5 minutes to submit.
**Deliverable:** a notebook or script in your repository, a written `RESULTS.md`,
and a leaderboard entry on competition **181**.

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

Record it. It is the argument for your metric choice, not a bar to clear — an F1
of `0.000` is beaten by anything that finds one true positive. The bar is the
logistic-regression pipeline you build in Part C.

Note what accuracy the dummy achieves, and put that in `RESULTS.md` too. The
distance between those two numbers is the whole reason this dataset is here.

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

`HistGradientBoostingClassifier` is scikit-learn's own gradient booster, so there
is no XGBoost or LightGBM to install; all three behave the same way here. Unlike
`LogisticRegression` it **cannot read sparse input**, and the Part C
`OneHotEncoder` emits sparse by default — so rebuild the transformer for this
pipeline with one argument changed:

```python
pre_dense = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler())]), num),
    ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("o", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False))]), cat),
])

gb = Pipeline([("pre", pre_dense),
               ("clf", HistGradientBoostingClassifier(random_state=42))])
```

If you forget, `cross_val_score` reports `ValueError: All the 5 fits failed. It
is very likely that your model is misconfigured.` The real error underneath is
`TypeError: Sparse data was passed for X, but dense data is required.` Expect
the five folds to take tens of seconds rather than the few the logistic
regression took — budget for it inside the ten minutes.

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

## Expected numbers

If your pipeline is leak-free and stratified, 5-fold F1 on this dataset lands
close to these. They were measured with the code above, on scikit-learn 1.8.0,
with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

| Model | F1 (mean ± std) |
|---|---|
| `DummyClassifier(most_frequent)` | 0.000 (accuracy 0.761) |
| Logistic regression, threshold 0.50 | 0.657 ± 0.006 |
| Logistic regression, best threshold ≈ 0.35 | 0.689 |
| `HistGradientBoostingClassifier`, threshold 0.50 | 0.713 ± 0.006 |

A logistic-regression F1 much **above** 0.68 at threshold 0.50 means you leaked
— go back to the checklist in *Validation & Overfitting*. Much **below** 0.60
means your preprocessing is dropping a column: run `pipe.fit(X, y)` once and
print `pipe.named_steps["pre"].get_feature_names_out().shape`. On this dataset it
is `(105,)` — 6 numeric columns and 99 one-hot levels. If yours is materially
smaller, print `len(num), len(cat)`: they must be `6, 8`, and a column that
reached neither list reached neither transformer.

---

## Part F — Submit (5 min)

The competition **PAIE S3 — Adult Census Income** (`181`) is this lab with the
last step attached: the same data, but the test labels are held back, so the
number you get is one you cannot have tuned against.

```bash
uv pip install mlarena-sdk
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.download_dataset(181, dest_dir="data")      # X_train.csv y_train.csv X_test.csv
```

Refit your Part C pipeline (or your Part D one) on `data/X_train.csv` against
`data/y_train.csv`, predict `data/X_test.csv`, and write `submission.csv` with
columns `id,prediction`, where `1` means >50K:

```python
import pandas as pd

X_tr = pd.read_csv("data/X_train.csv").set_index("id")   # id as index, not a feature
y_tr = pd.read_csv("data/y_train.csv").set_index("id")["prediction"]
X_te = pd.read_csv("data/X_test.csv").set_index("id")

pipe.fit(X_tr, y_tr.loc[X_tr.index])
pred = pipe.predict(X_te)
pd.DataFrame({"id": X_te.index, "prediction": pred}).to_csv(
    "submission.csv", index=False)

client.submit(competition_id=181, files=["submission.csv"])
print(client.leaderboard(181).head())
```

Setting `id` as the index is not cosmetic — it keeps the row identifier out of
the feature matrix. Re-derive `num` and `cat` from a frame where `id` is still a
column and `select_dtypes(exclude="number")` hands the identifier to the one-hot
encoder as a categorical feature with 39,073 levels, one per training row, every
one of them unknown at test time.

Always predicting the majority class scores **F1 = 0.000** on this split. The
Part C logistic-regression pipeline, refit exactly as above, scores
**F1 = 0.656** — that is the bar, and it is on the board under `__benchmark__`.
Tuning the threshold to 0.38 reaches 0.692, and gradient boosting 0.715. Higher
is better. The leaderboard F1 is the number your `RESULTS.md` has to explain: if
it is far below your cross-validated number, say why.

---

## Grading

| Criterion | Weight |
|---|---|
| Metric chosen and justified **before** results | 15% |
| Baseline measured and reported | 15% |
| Zero leakage: all preprocessing inside the Pipeline | 30% |
| Stratified CV, mean **and** spread reported | 20% |
| Threshold analysis with a stated tradeoff | 10% |
| Submitted to ML-Arena competition 181 | 10% |

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

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone, with the Lab 3 script committed
- [ ] `git log` shows the Part A metric section of `RESULTS.md` committed **before** any result
- [ ] `RESULTS.md` records the dummy's F1 **and** its accuracy (0.000 and 0.76)
- [ ] `grep -n fit_transform` on my script returns nothing — every transformer is fitted by the `Pipeline`
- [ ] My logistic-regression 5-fold F1 is between 0.60 and 0.68, reported as mean ± std
- [ ] My gradient-boosting `cross_val_score` returned 5 numbers, not `All the 5 fits failed`
- [ ] `RESULTS.md` names the best threshold, its F1, and what moving it did to precision and recall
- [ ] `submission.csv` has 9,770 lines (header + 9,769 test ids) and no duplicate id
- [ ] My submission is on the leaderboard of PAIE S3 — Adult Census Income (#181)
- [ ] My score beats the baseline: **F1 ≥ 0.656**

If the last two are not ticked you have not finished the lab, however good the
code is.
