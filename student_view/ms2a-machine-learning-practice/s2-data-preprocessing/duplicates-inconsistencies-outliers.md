# Duplicates, Inconsistencies, Outliers

Three defects that share a property: none of them is detected by a type check,
and all of them change your metric. They are also the three where "clean the
data" most often means "delete the evidence".

<!-- notes: 30 minutes. The duplicate-across-the-split point is the one to make
loudly — a duplicated row that lands in both train and test is a free correct
prediction and it inflates every score in the room. -->

---

## Three kinds of duplicate

| Kind | Definition | Usual cause |
|---|---|---|
| **Exact** | identical in every column | a re-run of the ingestion job |
| **Partial** | identical in the key, different elsewhere | two systems merged |
| **Approximate** | not identical, same entity | typos, free-text entry |

Only the first is unambiguous. The other two require a decision about what
identity means in your table, which is the "what is one row?" question from
Session 1 arriving with consequences.

---

## Detecting

```python
df.duplicated().sum()
df.duplicated(subset=["station_id", "date"], keep=False).sum()
dupes = df[df.duplicated(subset=["station_id", "date"], keep=False)]
```

`subset` restricts the comparison to the columns that define identity.
`keep=False` marks *every* member of a group rather than the copies only — what
you want for inspection, never for deletion. Look at `dupes` before dropping: if
the rows disagree on a measurement, you have a conflict, not a duplicate.

---

## Removing

```python
df = df.sort_values("ingested_at")
df = df.drop_duplicates(subset=["station_id", "date"], keep="last")
```

`keep` is `"first"`, `"last"` or `False`. On an unsorted frame `"first"` means
"whichever row the ingestion happened to write first" — a coin flip promoted to
a business rule. Sort explicitly on the column that encodes recency, then keep
`"last"`. If no such column exists, that is the bug to fix.

---

## The duplicate that costs you the most

A row present in both the training and the test split is a free correct
prediction. Near-duplicates do the same thing more quietly: two records of the
same customer, one on each side of the split.

```python
assert set(train["customer_id"]) & set(test["customer_id"]) == set()
```

De-duplicate **before** splitting, and split on the entity, not on the row —
`GroupShuffleSplit` in sklearn. A model that scores 0.97 on a leaky split and
0.78 on a grouped one was never a 0.97 model.

---

## Approximate duplicates

```python
from rapidfuzz import process, fuzz

matches = process.extract("John Smith", names, scorer=fuzz.token_sort_ratio,
                          limit=5, score_cutoff=88)
```

Pairwise comparison of `n` names is `n^2` operations, unusable past a few
thousand rows. Real entity resolution blocks first: compare only records sharing
a postcode, a birth year or the first three characters of a surname.

Fuzzy matching produces candidates, not decisions. Above 95 auto-merge, below 85
ignore, in between build a review list — and record the threshold you used.

---

## Inconsistencies

The same fact written several ways in one column:

| Kind | Example |
|---|---|
| Casing and whitespace | `"Paris"`, `" paris"`, `"PARIS "` |
| Units | metres and feet, °C and °F, EUR and USD |
| Dates | `2023-01-01`, `01/02/2023`, `04-05-2023` |
| Spellings | `"USA"`, `"U.S.A."`, `"United States"` |
| Mixed types | `1`, `2`, `"three"` in one column |

Every one of these silently multiplies a category, and a one-hot encoder will
give you three columns where the world has one thing.

---

## Standardising

```python
s = df["city"].str.strip().str.lower()
df["city"] = s.replace({"u.s.a.": "usa", "united states": "usa"})
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
```

`errors="raise"` rather than `errors="coerce"`. Coercion turns an unparseable
date into `NaT` and you discover in Session 3 that 8% of your rows have no
timestamp. Raising tells you today, with the offending string in the traceback.

Units have no distinguishing type — 67.97 °F and 19.98 °C are both floats — so
the only defence is a range assertion in the check function from Session 1:

```python
assert df["temp_c"].between(-60, 60).all(), "temperature out of range"
```

The mapping dictionaries and the bounds belong in the repository, versioned,
next to a test that asserts every raw value maps to something.

---

## Outliers: three kinds

- **Point** — a single value far from the distribution
- **Contextual** — normal in general, impossible here: 25 °C in Oslo in January
- **Collective** — no single point is extreme, the *pattern* is

The first is what the standard detectors find. The second and third need domain
knowledge, and a detector that flags them is usually flagging the wrong rows.

---

## Detecting point outliers

Z-score, for roughly symmetric data:

$$
z_i = \frac{x_i - \bar{x}}{s}
$$

```python
z = (df["revenue"] - df["revenue"].mean()) / df["revenue"].std()
outliers = df[z.abs() > 3]
```

The mean and the standard deviation are themselves moved by the outliers, so on
a heavy-tailed column this under-detects. The IQR rule is not:

```python
q1, q3 = df["revenue"].quantile([0.25, 0.75])
iqr = q3 - q1
mask = df["revenue"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
```

---

## Multivariate outliers

```python
from sklearn.ensemble import IsolationForest

flags = IsolationForest(contamination=0.01, random_state=0).fit_predict(X_num)
```

A 40-year-old is normal. A 40-year-old with 45 years of professional experience
is not, and no single-column rule sees it. Isolation Forest isolates points with
random splits and scores how few splits it takes. `contamination` is an
assumption you are making, not one the algorithm discovers — state it, and check
what it flagged.

---

## Remove, clip, or keep

| Situation | Action |
|---|---|
| Physically impossible (age 300, negative price) | remove, and fix the source |
| Genuine extreme, model is linear or distance-based | clip to a percentile |
| Genuine extreme, model is a tree | keep — it splits around it |
| The extreme *is* the target (fraud, failure, churn) | keep, obviously |
| Heavy right tail across the column | transform, do not clip |

```python
lo, hi = df["revenue"].quantile([0.01, 0.99])
df["revenue"] = df["revenue"].clip(lo, hi)
```

Clipping bounds the influence without inventing a value. Compute `lo` and `hi`
on the training rows only — they are fitted parameters like any other.

---

## The rule

> An outlier is a claim about the data-generating process, not a property of a
> number. Removing one you cannot explain is deleting evidence.

Log every row you drop, with the reason, in a counter the pipeline prints. A
cleaning step that silently removes 4% of the table is the failure mode of this
lesson, and nothing downstream will ever tell you it happened.
