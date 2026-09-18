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

![Missingness in the penguins table](assets/preprocessing/penguins-missingness-map.png)

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

- Missing Completely At Random — dropping is unbiased, merely wasteful
- Missing At Random — imputable from the columns that explain it
- Missing Not At Random — every imputation biases the result, and so does dropping


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

![Missing values per column and per row](assets/preprocessing/missing-values-per-column-and-row.png)

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


![ts2_11.jpg](assets/preprocessing/ts2_11.jpg)


---

## KNN imputation

```python
from sklearn.impute import KNNImputer

imp = KNNImputer(n_neighbors=5, weights="distance")
X_tr = imp.fit_transform(X_tr)
```

Each gap is filled with the weighted mean of the `k` most similar rows over the
other columns. It exploits MAR structure that a column median throws away.

![KNN imputation from the nearest rows](assets/preprocessing/knn-imputation.png)

Two costs: it needs scaled features, or the column with the largest units
defines "similar"; and it stores the training set, so inference is expensive.
`n_neighbors` is a hyperparameter — tune it in the pipeline, not by eye.

---

## Iterative imputation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=0, max_iter=10) # BayesianRidge default estimator
```

Each column with gaps is regressed on the others, round-robin, until the
estimates stop moving. It is the strongest single-value imputer in sklearn and
the easiest to over-trust: the imputed values arrive with no uncertainty
attached, and any model downstream treats them as measured.

Use it when the gaps are genuinely predictable from the other columns.
Otherwise the median plus an indicator is more honest and ten times faster.


![799631_m_z8E4HrFtCnHBoDANauTQ.png](assets/preprocessing/799631_m_z8E4HrFtCnHBoDANauTQ.png)


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
