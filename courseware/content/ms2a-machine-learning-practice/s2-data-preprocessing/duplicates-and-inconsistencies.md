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
