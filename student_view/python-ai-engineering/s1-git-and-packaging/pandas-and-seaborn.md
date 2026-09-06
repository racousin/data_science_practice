# Reference — pandas & seaborn

Self-study, never lectured. Every session from 2 onward reads a CSV, reshapes
it, and plots it; this page is the subset of `pandas` and `seaborn` those
sessions actually use, and nothing else.

If you already work with data frames, skim to *Categories into numbers* — the
`reindex` alignment idiom in that section is the one thing here that reliably
catches people who know pandas.

<!-- notes: Reference lesson, in_deck: false. Pointed at from Session 2's "The
Data". The scope is deliberately the intersection of "what the course uses" and
nothing more — it was derived from the actual API calls in the lessons and the
four challenge notebooks. -->

---

## What each one is for

The chain every challenge in this course runs:

$$
\text{CSV} \xrightarrow{\ \text{pandas}\ } \text{a table you understand}
\xrightarrow{\ \text{seaborn}\ } \text{a picture}
\xrightarrow{\ \text{pandas}\ } X \in \mathbb{R}^{n \times p}
$$

`scikit-learn` will not accept anything but the last object — a rectangle of
numbers. `pandas` is how you get there and `seaborn` is how you look at what you
have before you commit to it.

There is a **runnable companion notebook** to this page. Open it in Colab and
execute it before Session 2; reading about a data frame is much less useful than
having one in front of you.

---

## The DataFrame

A `DataFrame` is a dictionary of **typed columns** sharing one index. That is
the whole model, and the two halves both matter:

- **Typed columns** — each column has one dtype (`float64`, `int64`, `bool`,
  `str`). Not each cell. This is why a single bad value turns a whole numeric
  column into text.
- **A shared index** — the row labels. Two columns from the same frame align by
  index, not by position, and so do two *frames*. Most surprising pandas
  behaviour is this alignment doing its job when you expected positions.

```python
import pandas as pd
import seaborn as sns
df = sns.load_dataset("penguins")     # 344 rows x 7 columns
```

---

## Read and inspect

Four calls, in the order you should always run them.

```python
df = pd.read_csv("X_train.csv")
df.shape        # -> (344, 7)   the (n, p) of the data lesson
df.head(3)      # the first rows — do you believe the columns?
df.dtypes       # what pandas decided each column is
```

`dtypes` is the one people skip and it is the one that bites. `pandas` infers
types on read: a numeric column with a stray `"N/A"` comes back as `str`, and
every later arithmetic on it fails or, worse, silently concatenates.

```python
df.describe()   # count/mean/std/quartiles — numeric columns only
```

`describe` **silently drops missing values**, so its `count` row is your first
evidence they exist: 342 where the frame has 344 rows means two are missing.

---

## Selecting

One bracket gives a **Series** (one column, a 1-D labelled array); two give a
**DataFrame**. Losing track of which you hold is the most common beginner error.

```python
df["body_mass_g"]              # Series
df[["species", "body_mass_g"]] # DataFrame
```

Rows come out with a boolean mask — a Series of `True`/`False`, which you build
with ordinary comparisons:

```python
df[df["body_mass_g"] > 5000]                       # one condition
df[(df["sex"] == "Male") & (df["island"] == "Dream")]   # & and |, not `and`/`or`
```

The parentheses are required and `and` genuinely does not work here: `&`
operates element-wise on the whole column, `and` tries to collapse it to one
truth value and raises.

Dropping is by keyword, and the keyword matters:

```python
X = df.drop(columns=["species"])   # NOT df.drop("species") — that drops a ROW
```

Splitting numeric from categorical is a step every challenge notebook performs:

```python
df.select_dtypes("number")            # the 4 measurement columns
df.select_dtypes(exclude="number")    # species, island, sex
```

---

## Missing values

```python
df.isna().sum()      # count per column: sex 11, the measurements 2 each
df.dropna()          # 333 of 344 rows survive
```

`isna()` returns a frame of booleans the same shape as `df`, which is why
`.sum()` counts per column and why `sns.heatmap(df.isna())` draws you a map of
where the holes are. What to *do* about them is the data-preparation lesson;
this is only how to find them.

---

## Deriving and summarising

`assign` adds a column and returns a new frame, which keeps a chain readable and
leaves the original alone:

```python
d = df.assign(mass_kg=df["body_mass_g"] / 1000)
```

`groupby` is the one worth real attention, because it answers "does this feature
carry signal?" before any model does:

```python
df.groupby("species")["body_mass_g"].mean()
df.groupby("species")["body_mass_g"].agg(["count", "mean", "std"])
```

Read it as **split → apply → combine**: split the rows by `species`, take
`body_mass_g` in each group, apply the statistic, stack the results back into a
frame indexed by group.

For a **0/1 target this is the single most useful line in exploratory work**:

```python
df.groupby("island")["target"].mean()   # the positive RATE per level
```

The mean of a 0/1 column is its rate. Counts tell you which levels are common;
rates tell you which are predictive, and those are different questions.

Correlation, for the numeric block only:

```python
df.corr(numeric_only=True)
```

Remember what it measures: the **linear** part of a relationship. A column can
drive the target hard and still show a correlation near zero if the shape is not
a line — which is exactly what happens to `hour` in the bike challenge.

---

## Categories into numbers

`scikit-learn` needs $\mathbb{R}^{n \times p}$. One call converts every
non-numeric column into indicator columns and leaves the numeric ones untouched:

```python
X = pd.get_dummies(df)     # island -> island_Biscoe, island_Dream, island_Torgersen
```

### The alignment trap

This is the part to actually read. Encode train and test **independently** and
you can get different columns — a category present in one and absent in the
other produces a matrix of a different width, or the same width with columns in
a different order. Either way the model is fed nonsense, and usually without an
error.

```python
X_train_enc = pd.get_dummies(X_train)
X_test_enc = pd.get_dummies(X_test).reindex(columns=X_train_enc.columns, fill_value=0)
```

`reindex(columns=...)` forces the second frame onto the first's exact column
list: columns missing from the test set are created and filled with `0`, columns
the test set has and training did not are dropped. The result is guaranteed to
match the matrix the model was fitted on, which is the only thing that matters.

Assert it, once, and never think about it again:

```python
assert list(X_train_enc.columns) == list(X_test_enc.columns)
```

---

## Writing the submission

```python
pd.DataFrame({"id": X_test["id"], "prediction": preds}).to_csv("submission.csv", index=False)
```

`index=False` is not cosmetic. Without it pandas writes its row labels as a
leading unnamed column, the file gains a column the scorer did not ask for, and
the submission is rejected.

---

## seaborn: the shape of the API

Two things explain nearly all of it.

**One, every plot takes tidy data.** You pass the whole frame and name the
columns — seaborn does the grouping, the aggregation and the legend.

```python
sns.barplot(data=df, x="species", y="body_mass_g")
```

`hue=` splits any plot by a third column and builds the legend. It is the
highest-value single argument in the library:

```python
sns.lineplot(data=df, x="hour", y="count", hue="workingday")
```

**Two, there are two kinds of plot function**, and mixing them up is the usual
source of "why is my figure blank":

| | draws onto | takes `ax=` | examples |
|---|---|---|---|
| **axes-level** | one subplot you give it | yes | `histplot`, `barplot`, `boxplot`, `lineplot`, `regplot`, `heatmap`, `countplot` |
| **figure-level** | a whole figure it creates itself | **no** | `pairplot`, `relplot`, `catplot`, `displot` |

So `sns.pairplot(...)` cannot be placed inside a subplot grid you made; it *is*
the grid.

---

## The plots this course uses

| call | answers |
|---|---|
| `sns.histplot(df["y"])` | what does my target look like — skewed, bounded, bimodal? |
| `sns.boxplot(data=df, x="cat", y="y")` | does `y` differ across the levels of a category? |
| `sns.barplot(data=df, x="cat", y="y")` | the mean of `y` per level, with a confidence bar |
| `sns.countplot(data=df, x="cat")` | how many rows per level — the counts, not the rates |
| `sns.lineplot(data=df, x="t", y="y", hue="g")` | how does `y` move along an ordered axis, per group? |
| `sns.regplot(data=df, x="a", y="b")` | scatter plus the fitted line — is this relationship linear? |
| `sns.heatmap(df.corr(numeric_only=True), annot=True)` | which columns are redundant with which |
| `sns.pairplot(df, hue="target")` | every numeric pair at once — the first thing to run |

`pairplot` is expensive on wide frames ($p^2$ panels). Restrict it:
`sns.pairplot(df, vars=["bmi", "bp", "glucose"], hue="species")`.

---

## Just enough matplotlib

seaborn draws on matplotlib, so laying figures out is matplotlib's job.

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["body_mass_g"], ax=axes[0])
sns.boxplot(data=df, x="species", y="body_mass_g", ax=axes[1])
plt.tight_layout()
plt.show()
```

- `plt.subplots(rows, cols)` returns the figure and an array of axes.
- `ax=` sends an **axes-level** seaborn plot to a particular panel.
- `plt.tight_layout()` stops labels overlapping.
- `plt.show()` renders it. In a notebook the last expression often displays on
  its own, which is why you will see it omitted — call it anyway and the
  behaviour stops depending on where the line sits in the cell.

Titles and labels hang off the axes, not off seaborn:

```python
axes[0].set_title("Body mass")
axes[0].set_xlabel("grams")
```

---

## The habit worth keeping

Four lines, on every new dataset, before any modelling:

```python
df.shape          # how much data, how wide
df.dtypes         # what pandas thinks each column is
df.isna().sum()   # where the holes are
df.describe()     # the scale of each numeric column
```

then one plot of the target and one `pairplot`. It costs a minute and it is the
difference between modelling the data you have and modelling the data you
assumed you had.
