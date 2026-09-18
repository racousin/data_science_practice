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

## Nominal and ordinal

![Nominal versus ordinal](assets/preprocessing/nominal-vs-ordinal.png)

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

## One-hot in a linear model

![hour as a number versus hour as 24 categories](assets/preprocessing/hour-integer-vs-onehot-linear-fit.png)

A linear regression on bike rentals, with `hour` as one integer column and as 24
one-hot columns.

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
