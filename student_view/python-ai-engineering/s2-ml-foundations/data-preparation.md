# Data Preparation

Every model in this session is a function on $\mathbb{R}^p$. Real tables are not
in $\mathbb{R}^p$ — they have holes and they have words. Getting from one to the
other is where most of the project's time goes, and every choice you make along
the way is a modelling assumption whether you declare it or not.

<!-- notes: 40 minutes. The domain-knowledge table on missing data is the one to
slow down for — it is the difference between a student who imputes the mean
everywhere and one who thinks. -->

---

## The constraint

$$
f_\theta : \mathbb{R}^{p} \longrightarrow \mathcal{Y}
$$

Read the domain literally. $\theta^\top x$ is not defined when $x_j$ is `NaN`,
and it is not defined when $x_j$ is `"Torgersen"`. Before any model:

- **no missing entries** — every cell has a number
- **no non-numeric entries** — categories are mapped into $\mathbb{R}$

Neither requirement has a canonical answer. Both are decisions, and both change
what the model can learn.

One consequence to keep in view. Supervised learning means predicting on rows
whose target you do not have — so whatever mapping you build here must be
applicable to those rows too. An encoding that needs the target to compute, or a
category-to-integer map you did not keep, is not a preparation step you can use.

---

## Missing values

![A table with NaNs](/api/academic_courses/assets/lessons/152/missing-values-table.png)

Two ordinary causes:

- **Not collected** — through oversight, or because it was not available.
- **Corrupted** — entries lost in transfer or storage.

### Look at the pattern first

```python
df.isna().sum()
sns.heatmap(df.isna(), cbar=False)
```

![Missingness in the penguins table](/api/academic_courses/assets/lessons/152/penguins-missing.png)

The penguins table from the data lesson, with its 344 rows across. Two things
are visible that `df.isna().sum()` alone does not tell you: the four
measurement columns go missing *together*, in exactly 2 rows — two birds that
were never measured — while `sex` is missing in 11 rows scattered
independently. Those are two different mechanisms and they deserve two different
treatments.

The pattern matters more than the count. Blocks mean something systematic;
scatter means something closer to random.

![Missing-value patterns](/api/academic_courses/assets/lessons/152/missing-values-heatmap.png)

The formal version of the distinction — MCAR, MAR, MNAR — is the one you already
know from your statistics courses, and it is exactly what decides whether the
strategies below are unbiased.

---

### Strategy 1 — drop

```python
df.dropna()                    # drop rows with any NaN
df.dropna(subset=['age'])      # only where 'age' is NaN
```

| When to use | Risk |
|---|---|
| Few missing values (<5%) | You lose data |
| Missing completely at random | Bias, if the missingness is not random |

---

### Strategy 2 — impute a central value

Mean, median or mode. Simple, and reasonable when the column has low variance.

![Mean imputation](/api/academic_courses/assets/lessons/152/mean-imputation.png)

```python
from sklearn.impute import SimpleImputer
df[['age']] = SimpleImputer(strategy='mean').fit_transform(df[['age']])
```

Note what it costs: imputing the mean leaves the mean unchanged and shrinks the
variance and every covariance involving that column. You have not added
information, you have added confidence you do not have.

---

### Strategy 3 — forward fill (time series)

Propagate the last observed value into the gap.

![Forward fill](/api/academic_courses/assets/lessons/152/forward-fill.png)

```python
df['temp'] = df['temp'].ffill()
df['temp'] = df['temp'].interpolate(method='linear')
```

`interpolate` uses the future as well as the past. On a time series that is a
statement about what you will know at prediction time — be sure it is true.

---

### Strategy 4 — predict the missing value

Use the similarity between rows: more accurate than a central value on complex
tables, and considerably more expensive.

![KNN imputation](/api/academic_courses/assets/lessons/152/knn-imputation.png)

```python
from sklearn.impute import KNNImputer
df_imputed = KNNImputer(n_neighbors=5).fit_transform(df)
```

---

### Strategy 5 — domain knowledge

Often the missingness *is* the signal, and imputing destroys it:

| Domain | Missing ≠ random — it means something | Fix |
|---|---|---|
| Medicine | Lab not ordered → the patient was fine | Impute normal, not mean-of-sick |
| Finance | No price → the market was closed | Forward-fill, never mean |
| Sensors | Gap → the sensor died, not random | Use a neighbouring sensor |
| Credit | No mortgage history → never had one | Encode missingness as a feature |
| E-commerce | No rating → didn't care enough | Implicit feedback, do not ignore |

The last column is worth more than the four techniques above it. When in doubt,
add an explicit `was_missing` indicator column and let the model decide — it
costs one binary feature and it preserves the information that imputation
throws away.

---

## Outliers

![An outlier in a scatter plot](/api/academic_courses/assets/lessons/152/outlier-scatter.png)

The first question is never "how do I remove it".

| Errors | Real signal |
|---|---|
| Sensor malfunction | Legitimate extreme values |
| Data entry mistake | Rare but real events |
| ETL bug | Important for the model |

Delete a fraud case as an outlier and you have deleted the thing you were hired
to predict.

---

### Detection: the IQR rule

![IQR on a boxplot](/api/academic_courses/assets/lessons/152/iqr-boxplot.png)

```python
q1, q3 = df['col'].quantile([0.25, 0.75])
outliers = df[~df['col'].between(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))]
```

The interquartile range measures the spread of the middle 50%. Because it is
built from quantiles, the extreme points cannot move the threshold that is being
used to judge them.

---

### Detection: the z-score

![Z-scores on a normal distribution](/api/academic_courses/assets/lessons/152/z-score-normal.png)

```python
from scipy import stats
outliers = df[np.abs(stats.zscore(df['col'])) > 3]
```

Two failure modes you should expect. It **assumes approximate normality** — on a
skewed column it flags the whole tail. And it is not robust: the outlier
inflates the $\hat{\sigma}$ it is then measured against. On
$\{1, 2, 3, 4, 100\}$ the z-score of 100 is 2.0 — comfortably under 3, so the
test **misses the one point it exists to find** — while the IQR fence sits at 7
and catches it immediately. Prefer IQR, or a robust z-score built on the median
and the MAD.

---

### Strategy

![Ways to handle outliers](/api/academic_courses/assets/lessons/152/outlier-strategies.png)

| Strategy | Code | When |
|---|---|---|
| Remove | `df = df[~mask]` | Clear errors, few outliers |
| Clip | `df['col'].clip(lo, hi)` | Limit extremes without dropping rows |
| Transform | `np.log1p(df['col'])` | Reduce skewness |
| Keep | do nothing | Legitimate extremes |

---

## Categorical values

![Nominal versus ordinal](/api/academic_courses/assets/lessons/152/categorical-types.png)

- **Nominal** — no inherent order (`island ∈ {Torgersen, Biscoe, Dream}`).
- **Ordinal** — a logical order (`{bad, good, excellent}`).

The distinction decides the encoding, and getting it wrong invents structure the
data never had.

```python
sns.countplot(x='color', data=df)
```

![Categorical distribution](/api/academic_courses/assets/lessons/152/categorical-barplot.png)

---

### Ordinal encoding

![Label encoding](/api/academic_courses/assets/lessons/152/label-encoding.png)

```python
from sklearn.preprocessing import OrdinalEncoder
df[['size']] = OrdinalEncoder(categories=[['bad','good','excellent']]).fit_transform(df[['size']])
```

Map each category to an integer. This embeds the categories in $\mathbb{R}$ with
their order *and their spacing*, so it asserts both that `excellent > good` and
that the gap `good → excellent` equals the gap `bad → good`. The first claim is
usually what you want; the second rarely is, and a linear model acts on it.

Pass `categories=` explicitly. `LabelEncoder` sorts alphabetically, which on
`{bad, good, excellent}` gives exactly the wrong order — and `LabelEncoder` is
meant for the target column, not for features.

On a **nominal** column this is simply false: it tells the model
`Dream (1) < Torgersen (2)`, and the model will use it.

---

### One-hot encoding

![Label versus one-hot encoding](/api/academic_courses/assets/lessons/152/label-vs-onehot.png)

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_enc = ohe.fit_transform(df[['island', 'sex']])
```

One binary column per category: no order, no spacing, nothing asserted. This is
the default for nominal features.

Three practical notes. `handle_unknown='ignore'` matters — the default raises on
a category that appears only at prediction time, which is a class of failure you
meet in production and not in the notebook. The columns are exactly collinear
with the intercept, so drop one level (`drop='first'`) for a linear model, and
leave them all for a tree. And the cost is width: a column with 500 categories
becomes 500 mostly-empty columns, pushing $p$ towards $n$ — which is where the
curse of dimensionality and mandatory regularisation both start, in Session 3.
