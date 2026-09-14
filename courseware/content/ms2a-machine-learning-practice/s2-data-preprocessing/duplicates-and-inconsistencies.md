# Duplicates and Inconsistencies

Two defects that share a property: neither is detected by a type check, and
both change your metric. A duplicate multiplies a row; an inconsistency
multiplies a category. They are also the two where "clean the data" most often
means "delete the evidence".

<!-- notes: 15 minutes. The duplicate-across-the-split point is the one to make
loudly — a duplicated row that lands in both train and test is a free correct
prediction and it inflates every score in the room. Keep fuzzy matching short:
candidates, not decisions. On inconsistencies, push the errors="raise" habit —
a coerced NaT is a defect you meet three sessions later with no traceback. -->

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

## Same tokens, different order

![Three product descriptions with the same words in a different order, matching words joined by colour](assets/preprocessing/fuzzy-matching-token-order.png)

One product, three descriptions, the same tokens in three orders. `token_sort_ratio`
sorts the tokens before comparing, so all three score as one entity.

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

## One date, two readings

![World map of the date order each country writes, with a legend for day-month-year, month-day-year and year-month-day](assets/preprocessing/date-format-by-country.png)

`01/02/2023` is 1 February in Paris and 2 January in Chicago. A column fed by two
sources carries both conventions, and nothing in the string says which.

---

## One country, many strings

![Country counts before cleaning](assets/preprocessing/country-casing-whitespace-variants.png)

The 20 most frequent of 26 distinct `Country` strings; repeated labels differ only
by surrounding whitespace, and stripping it and normalising case leaves 5.

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

## The rule

> A duplicate is a decision about what one row is; an inconsistency is a
> decision about what one value is. Neither decision belongs in a notebook cell.

Count every row you drop and every string you rewrite, and make the pipeline
print the counts. A step that silently removes 4% of the table, or silently
merges two categories into one, is the failure mode of this lesson — and the
next lesson applies the same rule to outliers.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import pandas as pd
   d = pd.DataFrame({"station_id":  ["s1", "s1", "s2"],
                     "date":        ["2026-01-01"] * 3,
                     "temp_c":      [3.0, 4.5, 9.0],
                     "ingested_at": ["2026-01-02", "2026-01-03", "2026-01-02"]})
   kept = (d.sort_values("ingested_at")
             .drop_duplicates(subset=["station_id", "date"], keep="last")
             .set_index("station_id"))
   print(kept.loc["s1", "temp_c"])   # -> 4.5
   print(len(kept))                  # -> 2
   ```

   **Answer.** 4.5 is the later ingestion. Drop the `sort_values` and `"first"`
   or `"last"` returns whichever row the ingestion job happened to write first —
   a coin flip promoted to a business rule.

2. Why must de-duplication happen *before* the train/test split, and what does
   splitting on the row rather than the entity cost you?

   **Answer.** A row present in both splits is a free correct prediction. Split
   on the entity — `GroupShuffleSplit` — because two records of the same customer
   land on either side otherwise. A model scoring 0.97 on a leaky split and 0.78
   on a grouped one was never a 0.97 model.

3. `pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")` runs clean on a column
   where 8% of the strings are unparseable. When do you find out, and what does
   `errors="raise"` do instead?

   **Answer.** With `coerce` you find out in Session 3, when 8% of your rows have
   a `NaT` timestamp and no traceback points at the cause. `errors="raise"`
   raises a `ValueError` today, with the offending string in it.
