# Data Preparation

Every model in this session is a function on $\mathbb{R}^p$. Real tables are not
in $\mathbb{R}^p$ — they have holes, they have words, and they have values no
sensor ever produced. This lesson is those three: how to see each one, and the
simplest thing that fixes it.

Then a fourth part that is not repair but design — **feature engineering**,
where you stop asking what the model needs and start asking what it cannot work
out for itself.

`pandas` is how you get from a CSV to a rectangle of numbers. `seaborn` is how
you look at what you have before you commit to it.

<!-- notes: 55 minutes. Merged with the old pandas & seaborn reference on
2026-09-08 — the reference was self-study nobody did, and its content is only
useful next to the problem it solves. The three problem sections are
deliberately the same shape: a picture, one line to catch it, one line to fix
it. Feature engineering was added the same day; the pptx lists it as step 6 of
preprocessing and has no slide behind it. -->

---

## The constraint

$$
f_\theta : \mathbb{R}^{p} \longrightarrow \mathcal{Y}
$$

Read the domain literally. $\theta^\top x$ is not defined when $x_j$ is `NaN`,
and it is not defined when $x_j$ is `"Torgersen"`. Before any model:

- **no missing entries** — every cell has a number
- **no non-numeric entries** — categories are mapped into $\mathbb{R}$

Neither has a canonical answer. Both are decisions, and both change what the
model can learn.

---

## The DataFrame

A `DataFrame` is a dictionary of **typed columns** sharing one index.

- **Typed columns** — each column has one dtype, not each cell. This is why a
  single bad value turns a whole numeric column into text.
- **A shared index** — two columns, or two frames, align by index and not by
  position. Most surprising pandas behaviour is that alignment doing its job.

```python
import pandas as pd, seaborn as sns
df = sns.load_dataset("penguins")     # 344 rows x 7 columns
```

---

## Read and inspect

Four calls, in the order you should always run them.

```python
df = pd.read_csv("X_train.csv")
df.shape        # -> (344, 7)   the (n, p) of the data lesson
df.head(3)      # do you believe the columns?
df.dtypes       # what pandas decided each column is
df.describe()   # count/mean/std/quartiles — numeric columns only
```

`dtypes` is the one people skip and the one that bites: a numeric column with a
stray `"N/A"` comes back as `str`. And `describe` **drops missing values**, so
its `count` row is your first evidence they exist — 342 against 344 rows means
two are gone.

---

## Selecting

One bracket gives a **Series**, two give a **DataFrame**.

```python
df["body_mass_g"]                                  # Series
df[df["body_mass_g"] > 5000]                       # rows, by boolean mask
df[(df["sex"] == "Male") & (df["island"] == "Dream")]   # & and |, not and/or
X = df.drop(columns=["species"])                   # NOT df.drop("species")
df.select_dtypes("number")                         # the numeric block
```

The parentheses are required, and `and` genuinely does not work: `&` operates
element-wise, `and` tries to collapse a column to one truth value and raises.

---

## seaborn: pass the frame, name the columns

Every plot takes tidy data — you hand it the whole frame and name columns, and
seaborn does the grouping, the aggregation and the legend. `hue=` splits any
plot by a third column.

```python
sns.histplot(df["body_mass_g"])                        # one distribution
sns.boxplot(data=df, x="species", y="body_mass_g")     # y across the levels of x
sns.countplot(data=df, x="island")                     # rows per level
sns.pairplot(df, hue="species")                        # every numeric pair
```

`pairplot` is the first thing to run and the most expensive ($p^2$ panels);
restrict it with `vars=[...]` on a wide frame.

---

## Problem 1 — missing values

![A table with NaNs](assets/s2-ml-foundations/data-preparation/missing-values-table.png)

Two ordinary causes: **not collected**, or **corrupted in transfer**. Which one
it is decides what you should do about it.

---

### See them

```python
df.isna().sum()                      # count per column
sns.heatmap(df.isna(), cbar=False)   # where the holes are
```

![Missingness in the penguins table](assets/s2-ml-foundations/data-preparation/penguins-missing.png)

The four measurement columns go missing *together*, in exactly 2 rows — two
birds never measured — while `sex` is missing in 11 scattered rows. Two
mechanisms, and the picture is what tells them apart. A block means something
systematic; scatter means something closer to random.

---

### Replace them

```python
df["sex"] = df["sex"].fillna("unknown")            # a category: a word
df["bill_length_mm"] = df["bill_length_mm"].fillna(df["bill_length_mm"].median())
df = df.dropna()                                   # or drop the rows entirely
```

For a category, a literal `"unknown"` is usually the honest fix: it keeps the
row and says what is true. For a number, the median — it does not move when the
column is skewed, and unlike the mean it is not dragged by the extremes you are
about to go looking for.

Both cost you something. Filling shrinks the variance of that column: you have
not added information, you have added confidence you do not have.

---

### When the hole *is* the signal

A lab test not ordered means the patient was fine. No mortgage history means
they never had one. Imputing the mean there destroys the thing you wanted.

```python
df["income_missing"] = df["income"].isna().astype(int)
```

One binary column, and the model decides for itself. When in doubt, do this
before you fill.

---

## Problem 2 — categories

![Nominal versus ordinal](assets/s2-ml-foundations/data-preparation/categorical-types.png)

- **Nominal** — no inherent order (`island ∈ {Torgersen, Biscoe, Dream}`).
- **Ordinal** — a logical order (`{bad, good, excellent}`).

Getting the distinction wrong invents structure the data never had.

---

### See them

```python
df.select_dtypes(exclude="number").columns   # which columns are words
df["island"].value_counts()                  # the levels, and how many of each
sns.countplot(data=df, x="island")
```

![Categorical distribution](assets/s2-ml-foundations/data-preparation/categorical-barplot.png)

Count the levels before you encode. Two levels cost you one column; five hundred
cost you five hundred.

---

### Replace them

```python
X = pd.get_dummies(df)      # island -> island_Biscoe, island_Dream, island_Torgersen
```

One binary column per level: no order, no spacing, nothing asserted. This is the
default for nominal columns, and it is one call for the whole frame — numeric
columns pass through untouched.

![Label versus one-hot encoding](assets/s2-ml-foundations/data-preparation/label-vs-onehot.png)

---

### The alignment trap

Encode train and test separately and a level present in one and absent in the
other gives you two matrices of different widths — fed to the model without an
error, and wrong.

```python
X_train_enc = pd.get_dummies(X_train)
X_test_enc = pd.get_dummies(X_test).reindex(columns=X_train_enc.columns, fill_value=0)
assert list(X_train_enc.columns) == list(X_test_enc.columns)
```

`reindex` forces the second frame onto the first's exact column list. Assert it
once and never think about it again.

---

### If the order is real

```python
from sklearn.preprocessing import OrdinalEncoder
df[["size"]] = OrdinalEncoder(categories=[["bad", "good", "excellent"]]).fit_transform(df[["size"]])
```

Pass `categories=` explicitly, or the levels are sorted alphabetically — which
on `{bad, good, excellent}` is exactly the wrong order. Integers assert order
*and* spacing, so this says the gap `good → excellent` equals `bad → good`. On a
nominal column it is simply false, and a linear model will use it.

---

## Problem 3 — outliers

![An outlier in a scatter plot](assets/s2-ml-foundations/data-preparation/outlier-scatter.png)

The first question is never "how do I remove it". A sensor fault and a real rare
event look identical in the plot and are opposite in meaning — delete a fraud
case as an outlier and you have deleted the thing you were hired to predict.

---

### See them

```python
sns.boxplot(data=df, x="species", y="body_mass_g")

q1, q3 = df["col"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[~df["col"].between(q1 - 1.5*iqr, q3 + 1.5*iqr)]
```

![IQR on a boxplot](assets/s2-ml-foundations/data-preparation/iqr-boxplot.png)

The interquartile range is the spread of the middle 50%. Because it is built
from quantiles, an extreme point cannot move the threshold that is judging it —
which is exactly what a z-score lets it do.

---

### Replace them

```python
lo, hi = df["col"].quantile([0.01, 0.99])
df["col"] = df["col"].clip(lo, hi)        # keep the row, cap the value
df["col"] = np.log1p(df["col"])           # or squash a long right tail
```

Clipping keeps the row and its other columns, which is why it is usually better
than dropping. Dropping is for values that cannot exist — a negative age, a
timestamp from 1900 — and those you should fix at the source.

---

## Beyond the three problems — feature engineering

The three above are what the model *requires*: without them `fit` raises. This
one is what the model **cannot do for itself**.

$f_\theta$ is fixed once you have chosen the family — a linear model can only
ever add up its inputs. $x$ is not fixed. So when the shape you need is not in
the family, you put it in the columns instead.

<!-- notes: the pptx lists "Feature Engineering" as step 6 of preprocessing and
has no slide behind it. This is that slide. Do the hour example live — it is
worth more than the rest of the session's modelling advice put together. -->

---

### The column a linear model cannot use

![hour as a number versus hour as 24 categories](assets/s2-ml-foundations/data-preparation/hour-numeric-vs-onehot.png)

`hour` in the bike-demand data runs 0–23, and demand climbs to a commute peak at
08:00, falls, climbs again at 17:00, falls. A linear model gets **one**
coefficient for `hour`, so the only two statements it can make are "later is
busier" and "later is quieter". Both are wrong, and the fitted line above is the
compromise between them: nearly flat, and useless.

---

### Fixing it is one line

```python
X["hour"] = X["hour"].astype(str)      # 24 unordered levels, not a quantity
X = pd.get_dummies(X)
```

Twenty-four columns instead of one, and the model can put a different number on
every hour — the right-hand panel. On the Session 2 challenge that single change
takes −MAE from **−138.9 to −100.3** with the same `LinearRegression`, which is a
larger gain than any model in Session 3 buys you.

The same argument applies to `month` and `weekday`, and to every integer code
that names a thing rather than counting one.

---

### Cyclical columns

`astype(str)` throws away one true fact: 23:00 is next to 00:00. If that
adjacency matters — and it does for wind direction, day of year, angle — encode
the circle instead.

```python
X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
```

Two columns instead of twenty-four, and midnight sits beside 23:00 where it
belongs. The cost is that one sine can only bend once per cycle, so it fits a
single daily hump and not a double commute peak. Dummies when you have the rows
to spare; sine and cosine when $p$ has to stay small.

---

### The other three moves

| Move | Example | Why it helps |
|---|---|---|
| **decompose** | timestamp → year, month, hour, weekday | the parts carry the signal; the timestamp is one huge integer |
| **combine** | `price / surface`, `debt / income` | ratios are what the domain actually talks in |
| **interact** | `temp × workingday` | the effect of one column depends on another, and no linear model can discover that on its own |

```python
X["price_per_m2"] = X["price"] / X["surface"]
X["temp_x_working"] = X["temp"] * X["workingday"].astype(int)
```

Each one is a hypothesis about the problem, written as a column. A domain expert
is worth more here than a bigger model.

---

### Scale

`get_dummies` gives you 0/1 columns next to a `windspeed` in the tens. Linear
regression does not care — it just learns a smaller coefficient. Anything that
measures a **distance** does: KNN, SVM, and every neural network in Session 4.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)         # mean and std, from train
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
```
