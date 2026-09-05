# Missing Values

Missing data is the most common defect in a real table and the one with the
widest range of defensible answers. The mistake is not choosing badly; it is
choosing without asking *why* the value is absent.

<!-- notes: 35 minutes. Spend the first ten on MCAR/MAR/MNAR — it is the only
part they cannot look up in the sklearn docs. The income example lands: ask what
happens to a salary model when high earners decline to answer. -->

---

## First, look at it

```python
df.isna().mean().sort_values(ascending=False).head(10)
df.isna().sum(axis=1).value_counts().head()
```

The first line ranks columns by how empty they are. The second counts rows by
how many fields each is missing. A flat scatter of single gaps and a spike at
"eleven fields missing" are different problems: the spike is a failed batch, an
unjoined source or a schema change, none of which imputation fixes.

---

## Missing does not always look missing

```python
df = pd.read_csv("raw.csv",
                 na_values=["", "NA", "N/A", "-", "unknown", "-999"])
```

A sentinel is a missing value the loader did not recognise. `-999`, `0`,
`1900-01-01`, `"unknown"`, an empty string: each is a value pandas will happily
average.

Print `df.min()` for every numeric column and `value_counts()` for every
categorical one before imputing anything. A mean age of 41 in a table holding
three hundred `-999` rows is a mean age of nothing.

---

## Three mechanisms

| | Meaning | Example |
|---|---|---|
| **MCAR** | absence unrelated to any value | a sensor dropped packets |
| **MAR** | absence explained by *other observed* columns | older users skip the web form |
| **MNAR** | absence depends on the missing value itself | high earners decline to state income |

The consequences are not symmetric:

- MCAR — dropping is unbiased, merely wasteful
- MAR — imputable from the columns that explain it
- MNAR — every imputation biases the result, and so does dropping

---

## Diagnosing the mechanism

You cannot test for MNAR — the evidence is the data you do not have. You can
test for MAR:

```python
flag = df["income"].isna()
df.groupby(flag)[["age", "tenure", "n_visits"]].mean()
```

If the two groups differ, the missingness is explained by observed columns and
an imputer that uses them beats a column mean. If they do not differ, you are
looking at MCAR or MNAR, and only domain knowledge separates the two — ask
whoever collected the data.

---

## Dropping

```python
df = df.dropna(subset=["target"])             # always: no label, no row
df = df.drop(columns=["field_97pct_empty"])   # sometimes
```

A row with no target is not training data, whatever else it contains. Drop it
first and record how many.

Dropping a *column* is the right call above roughly 60–70% missing, unless the
missingness itself is predictive. Dropping *rows* on a feature is rarely right:
at 5% missing in ten columns you can lose 40% of the table.

---

## Central tendency, and a more honest constant

```python
from sklearn.impute import SimpleImputer

num = SimpleImputer(strategy="median")
cat = SimpleImputer(strategy="constant", fill_value="MISSING")
```

Median rather than mean for numeric columns: one outlier moves a mean and moves
no median, and columns with outliers are exactly the ones people forget to
check. Every constant-fill imputation shrinks the column's variance and weakens
its correlation with everything else — a defensible default, not a free one.

For a categorical column prefer `"MISSING"` over the mode. It is a category, it
carries the information that the field was empty, and it survives one-hot
encoding as its own column.

---

## Time series: fill forward, never backward

```python
df = df.sort_values("ts")
df["price"] = df["price"].ffill(limit=3)
df["temp"] = df["temp"].interpolate(method="time")
```

Forward fill carries the last observed value forward: at time $t$ you only use
information available at time $t$. Backward fill copies the future into the
past. In a time-ordered dataset that is not an imputation, it is a leak, and it
will produce a beautiful backtest.

`limit=3` is the honesty parameter: a sensor silent for two days should read
missing, not "the same as Monday".

---

## KNN imputation

```python
from sklearn.impute import KNNImputer

imp = KNNImputer(n_neighbors=5, weights="distance")
X_tr = imp.fit_transform(X_tr)
```

Each gap is filled with the weighted mean of the `k` most similar rows over the
other columns. It exploits MAR structure that a column median throws away.

Two costs: it needs scaled features, or the column with the largest units
defines "similar"; and it stores the training set, so inference is expensive.
`n_neighbors` is a hyperparameter — tune it in the pipeline, not by eye.

---

## Iterative imputation

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=0, max_iter=10)
```

Each column with gaps is regressed on the others, round-robin, until the
estimates stop moving. It is the strongest single-value imputer in sklearn and
the easiest to over-trust: the imputed values arrive with no uncertainty
attached, and any model downstream treats them as measured.

Use it when the gaps are genuinely predictable from the other columns.
Otherwise the median plus an indicator is more honest and ten times faster.

---

## The missingness indicator

```python
num = SimpleImputer(strategy="median", add_indicator=True)
```

`add_indicator=True` appends a binary column per feature that had gaps. The
value is filled, and the fact that it was filled is preserved.

This is the highest-value line in the lesson. For MNAR data the indicator is
often a better predictor than the column it accompanies — "declined to state
income" is itself a signal — and it lets a tree route imputed rows separately
instead of trusting the median.

---

## In the pipeline, not in the dataframe

```python
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])
pre = ColumnTransformer([("num", num_pipe, NUM_COLS),
                         ("cat", cat_pipe, CAT_COLS)], remainder="drop")
```

That is the whole object, both branches, and it is the one to copy: `cat_pipe`
appears again in the encoding lesson and in Lab 2 and it always means these two
steps. Print `pre.fit_transform(X_tr).shape` the first time you build one — a
shape you cannot account for column by column is a column that fell into neither
list.

`df.fillna(df.median())` computed on the full frame is a leak, computed on the
train frame is code you have to remember to repeat at inference, and computed in
a notebook is code that does not exist.

Inside the pipeline the median is fitted once, stored, serialised with the model
and applied identically to a million-row batch or to one row arriving over HTTP.

---

## Choosing

| Situation | Do this |
|---|---|
| Target is missing | drop the row |
| Column > 70% empty | drop the column, keep an indicator |
| Numeric, MCAR, few gaps | median + indicator |
| Categorical | constant `"MISSING"` |
| Ordered by time | `ffill` with a `limit`, or `interpolate` |
| MAR, columns clearly related | KNN or iterative, inside the pipeline |
| Model is LightGBM / XGBoost / HistGB | leave the `NaN` — they split on it |

> Impute with the simplest method you can defend, and always keep the indicator.
> A model that is beaten by a better imputer will tell you so in cross-validation.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.impute import SimpleImputer

   imp = SimpleImputer(strategy="median", add_indicator=True)
   print(imp.fit_transform([[1.0], [np.nan], [3.0], [5.0]]).tolist())
   # -> [[1.0, 0.0], [3.0, 1.0], [3.0, 0.0], [5.0, 0.0]]
   ```

   **Answer.** One column went in, two came out. The gap was filled with the
   median of the observed values (3.0), and the second column records *which* row
   was filled — the fact that the value was absent survives the imputation.

2. Run this. You should get exactly the output shown — it is the complete
   `ColumnTransformer`, both branches.

   ```python
   import numpy as np, pandas as pd
   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import StandardScaler, OneHotEncoder

   df = pd.DataFrame({"age":    [25, np.nan, 41, 60],
                      "income": [30000.0, 42000.0, np.nan, 91000.0],
                      "city":   ["paris", "lyon", np.nan, "paris"],
                      "plan":   ["free", "pro", "pro", np.nan]})
   NUM_COLS, CAT_COLS = ["age", "income"], ["city", "plan"]

   num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
   cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("encode", OneHotEncoder(handle_unknown="ignore"))])
   pre = ColumnTransformer([("num", num_pipe, NUM_COLS),
                            ("cat", cat_pipe, CAT_COLS)], remainder="drop")

   print(pre.fit_transform(df).shape)          # -> (4, 10)
   ```

   **Answer.** Ten columns from four: `age`, `income`, their two missingness
   indicators, and six one-hot columns — three levels each for `city` and `plan`,
   because `"MISSING"` is a level of its own. If you cannot account for the shape
   column by column, a column fell into neither list.

3. Your rows are ordered in time and a sensor went silent for an afternoon. Why
   is `ffill` acceptable and `bfill` never?

   **Answer.** `ffill` carries the last observed value forward, so at time *t*
   you only use information available at *t*. `bfill` copies the future into the
   past — that is not an imputation, it is a leak, and it produces a beautiful
   backtest.
