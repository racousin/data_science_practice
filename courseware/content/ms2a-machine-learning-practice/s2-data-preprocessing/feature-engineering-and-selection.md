# Feature Engineering and Selection

Feature engineering is where domain knowledge enters a model. Feature selection
is where you take back the columns that engineering added and could not justify.
The two are one loop, and both run inside the Pipeline.

<!-- notes: 40 minutes, the longest lesson of the session. Spend real time on
cyclical encoding and on the group-aggregation leak; the selection half can move
faster because most of it is one sklearn call each. -->

---

## The highest-leverage feature is arithmetic

```python
df["price_per_m2"] = df["price"] / df["surface_m2"]
df["days_since_signup"] = (df["order_ts"] - df["signup_ts"]).dt.days
df["debt_to_income"] = df["debt"] / df["income"].clip(lower=1)
```

A ratio, a difference, a rate. Three lines that a gradient-boosted tree would
need many splits to approximate and a linear model could never represent at all.
`clip(lower=1)` is not cosmetic: a zero denominator produces `inf`, which is not
`NaN` and survives every missing-value check you wrote this morning.

---

## Interactions and polynomials

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
```

`PolynomialFeatures` generates every product up to `degree`;
`interaction_only=True` keeps the cross-terms and drops the squares. It is also
a combinatorial trap: 50 columns at degree 2 become 1,325. Apply it to a handful
of columns you chose, never to the whole matrix, and never in front of a tree
model — trees build interactions by construction.

---

## Datetime decomposition

```python
ts = pd.to_datetime(df["datetime"])
df["hour"] = ts.dt.hour
df["dayofweek"] = ts.dt.dayofweek
df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
```

A timestamp as an integer is nearly useless; its components are not. Electricity
demand depends on the hour and on whether the offices are open, and neither is
recoverable from a Unix epoch by any split a tree can make.

Add the domain calendar too — holidays, paydays, school terms, promotion weeks.
That is the part no library gives you.

---

## Cyclical encoding

Hour 23 and hour 0 are one hour apart, and the integers say 23. December and
January are adjacent, and the integers say 11. Project the value onto a circle:

$$
\sin\left(\frac{2 \pi h}{24}\right), \quad \cos\left(\frac{2 \pi h}{24}\right)
$$

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

Both are needed: the sine alone maps 03:00 and 09:00 to the same value (0.7071
each), and the cosine alone maps 06:00 and 18:00 to the same value (0.0 each).
Do this for hour, day of week and month whenever the model is linear or a
network. Trees can carve the discontinuity out with extra splits, so the gain
there is smaller.

---

## Binning

```python
df["age_band"] = pd.cut(df["age"], bins=[0, 25, 40, 60, 120])
df["income_q"] = pd.qcut(df["income"], q=5, labels=False)
```

`cut` uses the edges you supply — legal ages, tax brackets, tariff bands. `qcut`
uses quantiles and produces balanced bins. Binning trades information for
robustness to outliers and gives a linear model a non-linear response.

Before a tree it is almost always a loss: the tree was going to find the
threshold anyway, and it would have found a better one.

---

## Group aggregations, and the leak inside them

```python
agg = train.groupby("customer_id")["amount"].agg(["mean", "std", "count"])
train = train.join(agg, on="customer_id")
test = test.join(agg, on="customer_id")     # train statistics, applied to test
```

Per-group summaries — a customer's mean basket, a station's median temperature,
a store's order count — are among the strongest tabular features there are.

They leak in two ways. Computing `agg` over train **and** test puts test rows
into a training feature. Aggregating the *target* per group puts the label into
the feature, exactly as in target encoding: it needs the same out-of-fold
treatment, or a strictly past-only window when the rows are ordered in time.

---

## Text and counts, briefly

```python
df["desc_len"] = df["description"].str.len()
df["n_words"] = df["description"].str.split().str.len()
```

Length and count features carry a surprising share of the signal in a short
free-text column. Beyond them, TF-IDF over the top few hundred terms is a
reasonable feature block inside a tabular problem:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=500, min_df=5, ngram_range=(1, 2))
```

Anything more is representation learning, and it is Sessions 7 and 8.

---

## Why selection: the curse of dimensionality

Adding a column adds a dimension. In high dimensions volume grows faster than
any sample can fill it: points become equidistant, distance-based methods
degenerate, and the rows needed for the same density grow exponentially. The
practical symptoms arrive long before the theoretical ones:

- training slows in proportion to the width
- variance rises — the model fits noise in columns that carry none
- collinear columns split an effect between them and destabilise coefficients
- nobody can explain what the model uses

---

## Three families

| Family | How it decides | Cost | Blind spot |
|---|---|---|---|
| **Filter** | a statistic per feature against the target | one pass | interactions |
| **Wrapper** | train the model on candidate subsets | one fit per step | overfits on small data |
| **Embedded** | selection happens during fitting | nearly free | tied to that model |

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV

filt = SelectKBest(mutual_info_classif, k=20)
wrap = RFECV(estimator=LogisticRegression(), cv=5, scoring="roc_auc")
emb = LassoCV(cv=5)                     # L1 drives coefficients to exactly zero
```

Use `f_classif` / `f_regression` for linear association, `mutual_info_*` for any
dependence, `chi2` for non-negative counts.

All three are fitted steps and belong **inside** the pipeline. Selecting
features on the full dataset and cross-validating the survivors is the
second-largest leak in this course, and it buys several points of imaginary AUC.

---

## Correlation and VIF

```python
corr = X.corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print(pairs[pairs > 0.95])
```

Two columns correlated at 0.98 carry one piece of information and two
coefficients. Drop one — the cheaper to collect, or the easier to explain.

Pairwise correlation misses a column that is a linear combination of three
others. The variance inflation factor catches it, by regressing each feature on
all the rest:

$$
VIF_j = \frac{1}{1 - R_j^2}
$$

Above 10 is the usual alarm. It matters for linear models, where it makes
coefficients unstable and their signs arbitrary; trees are indifferent.

---

## Permutation importance

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(pipe, X_val, y_val, n_repeats=10, random_state=0)
```

Shuffle one column in the validation set, re-score, measure how far the score
fell. Model-agnostic, computed on held-out data, and it answers the question you
actually asked. Prefer it to `feature_importances_`, which is computed on the
training set and biased towards high-cardinality continuous columns.

Both mislead under correlation: shuffling one of a correlated pair leaves the
information available through the other, and neither looks important.

---

## PCA, and when it is the wrong tool

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95).fit(X_tr_scaled)   # keep 95% of the variance
```

PCA rotates the data onto orthogonal axes ordered by variance and keeps the
first few. It needs scaled inputs — variance has units — and is fitted on train
like anything else. Right tool: decorrelating a wide block of measurements,
compressing before a distance-based method, visualising at two components.

Wrong tool when you need to know which columns matter, because every output
mixes every input; when the structure is non-linear; and when the signal has low
variance, which PCA discards first — being unsupervised, it has never seen the
target.

> Prefer selection to projection whenever anyone will ask why the model made a
> decision.

---

## The loop

1. engineer features from domain knowledge, not from a library
2. put every one of them in the Pipeline
3. cross-validate: does the score move outside the noise?
4. drop what does not earn its place — width is a cost you pay every run
5. re-check the leak: could this value be computed at prediction time?

Step 5 is the one to write on the wall. A feature that will not exist at
inference — tomorrow's price, a field back-filled by an operator, an aggregate
over the full dataset — makes a model that validates beautifully and predicts
nothing.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np

   for h in (3, 9):
       print(h, round(float(np.sin(2 * np.pi * h / 24)), 4),
                round(float(np.cos(2 * np.pi * h / 24)), 4))
   # -> 3 0.7071 0.7071
   # -> 9 0.7071 -0.7071
   ```

   **Answer.** The sine alone cannot tell 03:00 from 09:00; the cosine can. That
   is why cyclical encoding always ships both columns — either one on its own
   folds two different hours onto the same value.

2. A per-customer mean basket is one of the strongest tabular features there is.
   Name the two distinct ways it leaks.

   **Answer.** Computing the aggregate over train **and** test puts test rows
   into a training feature. Aggregating the *target* per group puts the label
   into the feature, exactly as target encoding does — it needs the same
   out-of-fold treatment, or a strictly past-only window when the rows are
   ordered in time.

3. Name two situations in which PCA is the wrong tool, and say why.

   **Answer.** When you need to know which columns matter — every component mixes
   every input, so the explanation is gone. When the signal has low variance —
   PCA discards low-variance directions first, and being unsupervised it has
   never seen the target. (A third: when the structure is non-linear.)
