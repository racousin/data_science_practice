# Categorical Encoding

A category is a name. A model needs a number. Every encoding is a claim about
what arithmetic on that number is allowed to mean, and most of the damage in
this lesson comes from claims nobody noticed making.

<!-- notes: 35 minutes. The target-encoding leak is the centrepiece — write the
naive version on the board, ask the room whether it leaks, then show the
out-of-fold version. Half of them will have shipped the naive one. -->

---

## Two kinds, one question

| | Definition | Examples |
|---|---|---|
| **Nominal** | no order | city, colour, payment method |
| **Ordinal** | a real order | `low < medium < high`, education level |

The question to ask of any category column: **does the difference between two
codes mean anything?** For `red = 0, blue = 1, green = 2` it does not, and any
model that computes `blue - red` is computing noise.

Ordinality is a property of the world, not of the data. Only you know whether
`"medium"` sits between `"low"` and `"high"`.

---

## One-hot: the default for nominal

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                    min_frequency=10)
```

One binary column per category, no order implied, no ordering invented. This is
the correct default for nominal columns and for every linear model, neural
network or distance-based method.

`min_frequency=10` collapses everything rarer than ten occurrences into a single
`infrequent` column — high-cardinality control with one argument.

---

## What one-hot costs

| Column | Distinct values | Columns produced |
|---|---|---|
| `payment_method` | 4 | 4 |
| `country` | 195 | 195 |
| `city` | 12,000 | 12,000 |
| `product_sku` | 400,000 | 400,000 |

Past a few dozen categories the matrix is mostly zeros, trees split on each
column in isolation and find nothing, and the memory is real. Keep
`sparse_output=True` for linear models, which handle sparse matrices natively.
`drop="first"` avoids perfect collinearity — it matters for ordinary least
squares and unregularised logistic regression, and for nothing else.

---

## Ordinal encoding, and when it lies

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(categories=[["low", "medium", "high"]],
                     handle_unknown="use_encoded_value", unknown_value=-1)
```

Pass `categories` explicitly. The default is alphabetical, which orders
`"high" < "low" < "medium"` and hands the model a fluent lie.

`LabelEncoder` is not this. It takes a 1-D array and is meant for the *target*
column; using it on features one at a time silently re-derives an alphabetical
order per column, and it has no `handle_unknown`. Use `OrdinalEncoder`.

Trees tolerate an arbitrary ordinal code better than linear models do — a tree
can carve `{0, 3, 7}` out with three splits. It still wastes depth doing it.

---

## Target encoding

Replace the category with the mean target it is associated with, smoothed
towards the global mean:

$$
\hat{y}_c = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}
$$

One column instead of 12,000, and it carries the signal that matters. $m$ is the
smoothing weight: a category seen twice is pulled almost entirely to the global
mean, a category seen ten thousand times keeps its own.

This is the strongest encoding for high-cardinality features and the most
dangerous transformer in scikit-learn.

---

## Why the naive version leaks

```python
means = df.groupby("city")["target"].mean()          # wrong
df["city_enc"] = df["city"].map(means)
```

Row 7's encoded value was computed from a group that contains row 7's own label.
For a city seen once, `city_enc` *is* the target. The model learns to read the
answer off the feature, cross-validation confirms it, and production does not.

The symptom is unmistakable: a feature with implausible importance and a
validation score that collapses on genuinely new data.

---

## Out of fold, or not at all

```python
from sklearn.preprocessing import TargetEncoder

enc = TargetEncoder(smooth="auto", cv=5, random_state=0)
```

sklearn's `TargetEncoder` computes each training row's encoding from the *other*
folds, so no row contributes to its own feature. At `transform` time it uses the
full-train statistics, which is correct — the test rows never contributed.

Put it in the `ColumnTransformer` and never compute a target aggregate with
`groupby` on a dataframe you are about to train on. The same argument applies to
any group-level feature built from the label.

---

## Frequency and hashing

```python
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)               # fitted on train only
```

Frequency encoding replaces a category with how often it occurs. One column, no
target involved, no leak — and surprisingly strong when rarity is itself
informative (rare SKU, rare browser, rare error code).

```python
from sklearn.feature_extraction import FeatureHasher

h = FeatureHasher(n_features=256, input_type="string")
```

Hashing maps categories into a fixed number of columns, trading collisions for
a bounded width and no fitted vocabulary — which is why streaming systems use
it, where new categories appear continuously.

---

## High cardinality: pick one

| Strategy | Keeps signal | Leak risk | Note |
|---|---|---|---|
| One-hot with `min_frequency` | partly | none | first thing to try |
| Group rare into `"OTHER"` | partly | none | needs a threshold you defend |
| Frequency encoding | if rarity matters | none | one column, cheap |
| Target encoding, out of fold | most | high if hand-rolled | use `TargetEncoder` |
| Hashing | some | none | fixed width, unreadable |
| Native categorical | most | none | LightGBM, CatBoost |
| Learned embedding | most | none | needs a network, Session 8 |

LightGBM and CatBoost accept category columns directly and split on subsets of
levels. On tabular data that is frequently the best answer and requires no
encoding at all.

---

## The category that appears at inference

Training saw 12,000 cities. Production sends a 12,001st on the third day.

| Encoder | Argument | Behaviour |
|---|---|---|
| `OneHotEncoder` | `handle_unknown="ignore"` | all-zero row for that feature |
| `OrdinalEncoder` | `handle_unknown="use_encoded_value"` | the `unknown_value` code |
| `TargetEncoder` | built in | the global target mean |

Without those arguments the default is to raise — correct for a batch job, an
outage for an online service. Choose deliberately, and log a counter of unknown
categories either way: a rate climbing from 0.1% to 12% is the first visible
sign of drift.

---

## Choosing

| Column | Encoding |
|---|---|
| Binary (`yes`/`no`) | a single 0/1 column |
| Nominal, under ~30 levels | one-hot |
| Genuinely ordinal | `OrdinalEncoder` with explicit `categories` |
| Nominal, hundreds of levels, tree model | native categorical, or target encoding |
| Nominal, hundreds of levels, linear model | target or frequency encoding |
| Open-ended and growing | hashing |
| Free text | not a category — Session 7 |

> One-hot until it hurts, then target-encode out of fold. Anything that touches
> the label goes inside the Pipeline, where cross-validation can see it.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   from sklearn.preprocessing import OrdinalEncoder

   enc = OrdinalEncoder().fit([["low"], ["medium"], ["high"]])
   print(enc.categories_[0].tolist())                            # -> ['high', 'low', 'medium']
   print(enc.transform([["low"], ["medium"], ["high"]]).ravel().tolist())
   # -> [1.0, 2.0, 0.0]
   ```

   **Answer.** The default order is alphabetical, so the encoder has just told
   the model that `high < low < medium`. Pass `categories=[["low", "medium",
   "high"]]` explicitly, every time.

2. Why does `df.groupby("city")["target"].mean()` leak when you map it back onto
   the same frame, and what does sklearn's `TargetEncoder` do differently?

   **Answer.** Row 7's encoded value was computed from a group that contains row
   7's own label — for a city seen once, the feature *is* the target.
   `TargetEncoder(cv=5)` computes each training row's encoding from the *other*
   folds, so no row contributes to its own feature.

3. Run this. You should get exactly the output shown.

   ```python
   from sklearn.preprocessing import OneHotEncoder

   oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
   oh.fit([["paris"], ["lyon"]])
   print(oh.categories_[0].tolist())      # -> ['lyon', 'paris']
   print(oh.transform([["berlin"]]).tolist())   # -> [[0.0, 0.0]]
   ```

   **Answer.** The unseen category becomes an all-zero block instead of an
   exception. Without `handle_unknown="ignore"` the default is to raise —
   correct for a batch job, an outage for an online service.
