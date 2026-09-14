# Collection Strategy and Data Quality

Two decisions frame every collection: *when* you collect, and *what you check
before you believe any of it*.

<!-- notes: 25 minutes. The quality checks are the part they will reuse every
week — push them to write the check function into their project repo today, not
later. -->

---

## Batch or stream

| | Batch | Streaming |
|---|---|---|
| Cadence | scheduled — hourly, nightly | continuous, as events arrive |
| Unit | a large chunk | one record |
| Latency | minutes to a day | seconds |
| Reprocessing | easy — re-run the job | hard — the past has gone by |
| Tooling | cron, Airflow, a Makefile | Kafka, webhooks, queues |
| Example | end-of-day financial reports | sentiment on posts as they appear |

![Animation: batch processing counts a full car park in one pass; stream processing counts each car as it drives past](assets/collect/batch-stream.gif)

---

## Choose batch

Unless you have a stated latency requirement, batch is correct.

- historical analysis and model training
- complex aggregations over full history
- anything you will want to recompute after fixing a bug
- resource efficiency: one scheduled job, not an always-on consumer

Every project in this course trains on a fixed snapshot. That is batch.

---

## Choose streaming

- a decision must be taken within seconds of the event
- patterns must be detected as they occur — fraud, anomalies
- monitoring, alerting, live and interactive dashboards
- the volume genuinely cannot be stored before processing

Streaming buys latency and costs you reproducibility: you cannot re-run last
Tuesday. Do not pay that price for a nightly report.

---

## The same count, both ways

```python
# batch: one run over a dated snapshot
df = pd.read_parquet("data/raw/posts_2026-09-14.parquet")
counts = df.groupby("topic").size()
```

```python
# streaming: state updated one event at a time
counts = collections.Counter()
for event in consumer:              # e.g. a Kafka topic
    counts[event["topic"]] += 1
```

The batch version can be re-run on the same file and gives the same answer. The
streaming version holds its answer in memory; if the process restarts, the
state is gone unless you checkpoint it.

---

## Snapshot your data

```text
data/raw/orders_2026-09-14.parquet
```

Date the file. A model trained on "the data" is not reproducible; a model trained
on `orders_2026-09-14.parquet` is.

This is the same argument as pinning dependency versions, applied to the input
that matters most.

---

## Quality is checked at collection

![The six key data quality dimensions: accuracy, completeness, consistency, timeliness, validity and uniqueness](assets/collect/quality.png)

Six dimensions, all cheap to check and expensive to discover late:

| Dimension | Question |
|---|---|
| Completeness | how much is missing, and where? |
| Uniqueness | is the key actually unique? |
| Validity | do values fall in their allowed range? |
| Consistency | do sources agree with each other? |
| Timeliness | how old is the freshest record? |
| Accuracy | does it match reality? |

---

## The first look

```python
df.info()
df.describe(include="all")
df.isna().mean().sort_values(ascending=False).head(10)
df.duplicated(subset=["order_id"]).sum()
```

Four lines. Run them on every dataset you ever load, before any modelling
thought enters your head.

The `isna().mean()` line ranks columns by how missing they are — a column that is
97% empty is a column you should be arguing about, not imputing.

---

## Completeness and consistency, as checks

```python
# completeness: required fields present, and no gap in coverage
assert df[["order_id", "amount"]].notna().all().all()
print(df.resample("MS", on="ts").size())    # a month at 0 is a gap
```

A month with zero rows is almost never a quiet month; it is a month the
collection missed.

```python
# consistency: one spelling, one unit per column
print(df["country"].value_counts())         # "FR", "France", "fr"
```

Consistency also means naming conventions and units agree across sources: a
weight in grams in one file and in kilograms in another joins without an error.

---

## Validity and accuracy, as checks

```python
# validity: types, patterns, and relationships between fields
assert df["zipcode"].str.fullmatch(r"\d{5}").all()
assert (df["delivered_at"] >= df["ordered_at"]).all()
```

Validity is conformance to rules you can write down. Accuracy is agreement with
the real world, and needs something outside the dataset:

- the source system's own row count or revenue total for the period
- a sample of records checked by hand against the original
- known facts — no store sells before its opening year

Record the error rate you find. It is part of the dataset's documentation.

---

## Make the checks executable

```python
def check(df: pd.DataFrame) -> None:
    assert df["order_id"].is_unique, "order_id is not unique"
    assert df["amount"].between(0, 1e6).all(), "amount out of range"
    assert df["ts"].max() > pd.Timestamp("2026-01-01"), "data is stale"
```

A function in the repository, called by the ingestion step and by a test. Not a
notebook cell, and not a comment.

When the source changes in November, this crashes on the day it changes rather
than on the day you present.

---

## Missing is not one thing

| Pattern | Meaning | Consequence |
|---|---|---|
| MCAR | missing for no reason related to the data | dropping is safe, just wasteful |
| MAR | missingness explained by other columns | imputable from them |
| MNAR | missingness depends on the missing value itself | dropping introduces bias |

Income missing because high earners decline to answer is MNAR. Dropping those
rows biases every conclusion you draw. Imputation handles the mechanics; the
*diagnosis* belongs at collection, while you can still ask the source.

---

## Align types before combining

```python
a["item_code"].dtype, b["item_code"].dtype      # int64, and text
a.merge(b, on="item_code")                       # ValueError
b["item_code"] = pd.to_numeric(b["item_code"])   # raises on "N/A"
```

The same key read from a CSV and from a JSON API rarely arrives with the same
type. pandas refuses to merge an integer key with a text one — a clean crash.

Convert explicitly: `pd.to_numeric`, `pd.to_datetime(format="%d/%m/%Y")`,
`astype`. Each raises on a value it cannot convert, and that value is the one
you needed to see. Check the result with `df.dtypes`.

---

## Stacking sources

```python
frames = [pd.read_csv(p).assign(store=p.stem)
          for p in sorted(Path("data/raw").glob("store_*.csv"))]
df = pd.concat(frames, ignore_index=True)
```

`concat` stacks tables that describe the same thing — one export per store, one
file per month. `assign(store=...)` keeps the provenance of every row.

It aligns columns **by name**. `PRODUCT_ID` in one file and `product_id` in
another become two half-empty columns, without an error. Normalise the names
first: `df.columns = df.columns.str.lower()`.

---

## Joining sources

```python
before = len(orders)
df = orders.merge(customers, on="customer_id", how="left", validate="m:1")
assert len(df) == before
```

`validate="m:1"` makes pandas raise if the right-hand key is not unique. It is
the single most useful argument in `merge` and almost nobody uses it.

The row-count assertion catches the rest.

---

## Which rows a join keeps

| `how=` | Keeps |
|---|---|
| `inner` | only keys present on both sides |
| `left` | every row of the left table |
| `right` | every row of the right table |
| `outer` | every key from either side |

```python
df = a.merge(b, on="item_code", how="outer", indicator=True)
print(df["_merge"].value_counts())     # both, left_only, right_only
```

An inner join drops unmatched rows without a word. `indicator=True` shows how
many there were before you choose. `a.join(b)` is the same operation, matched
on the index instead of a column.

---

## Document the dataset

A short `DATASET.md` next to the data:

- what one row is
- one line per column: meaning, unit, source
- how it was collected, and when
- known problems and how they were handled
- licence and terms

Two hundred words. It is what makes the dataset usable by your teammate, by the
grader, and by you in six weeks.

---

## In one line

> Collect deliberately, store the raw response, check at the boundary, and write
> down where it came from.

A model is only as defensible as the dataset under it.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import pandas as pd

   orders    = pd.DataFrame({"customer_id": [1, 1, 2], "amount": [10, 20, 30]})
   customers = pd.DataFrame({"customer_id": [1, 1, 2],       # 1 is duplicated
                             "segment": ["A", "A", "B"]})

   print(len(orders.merge(customers, on="customer_id", how="left")))   # -> 5
   orders.merge(customers, on="customer_id", how="left", validate="m:1")
   # -> pandas.errors.MergeError: Merge keys are not unique in right dataset;
   #    not a many-to-one merge
   ```

   **Answer.** Three rows became five and nothing complained. `validate="m:1"`
   turns that silent duplication into an exception at the line that caused it.

2. You have no stated latency requirement. Batch or streaming, and what does the
   other one cost you?

   **Answer.** Batch. Streaming buys latency and costs reproducibility: you
   cannot re-run last Tuesday, so any bug you fix is a bug you cannot repair
   retroactively.

3. Income is missing because high earners decline to answer. Which of MCAR, MAR
   and MNAR is that, and why is dropping those rows not a safe default?

   **Answer.** MNAR — the absence depends on the missing value itself. Dropping
   the rows removes exactly the high earners, so it biases every conclusion you
   draw; unlike MCAR, where dropping is merely wasteful.

---

## Worked case study

Predict the number of items sold by StoreE from the data of four other stores,
spread over three kinds of source:

- **Files** — one export per store: two CSV layouts, a two-sheet Excel
  workbook, and JSON records, stacked with `concat`
- **API** — item volumes, behind an authentication call
  ([documentation](https://www.raphaelcousin.com/module4/api-doc))
- **Scraping** — ratings and review counts, on a
  [JavaScript-rendered page](https://www.raphaelcousin.com/module4/scrapable-data)

Each source is joined on the product id, and the same baseline model is scored
after each join, so the change in error measures what that source adds.

[Open the case study in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s1-case-study.ipynb)
