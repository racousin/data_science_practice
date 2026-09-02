# Collection Strategy and Data Quality

Two decisions close the session: *when* you collect, and *what you check before
you believe any of it*.

<!-- notes: 25 minutes. The quality checks are the part they will reuse every
week — push them to write the check function into their project repo today, not
in Session 2. -->

---

## Batch or stream

| | Batch | Streaming |
|---|---|---|
| Cadence | scheduled — hourly, nightly | continuous, as events arrive |
| Unit | a large chunk | one record |
| Latency | minutes to a day | seconds |
| Reprocessing | easy — re-run the job | hard — the past has gone by |
| Tooling | cron, Airflow, a Makefile | Kafka, webhooks, queues |

![Batch versus streaming](assets/collect/batch-stream.gif)

---

## Choose batch

Unless you have a stated latency requirement, batch is correct.

- historical analysis and model training
- complex aggregations over full history
- anything you will want to recompute after fixing a bug

Every project in this course trains on a fixed snapshot. That is batch.

---

## Choose streaming

- a decision must be taken within seconds of the event
- monitoring, alerting, live dashboards
- the volume genuinely cannot be stored before processing

Streaming buys latency and costs you reproducibility: you cannot re-run last
Tuesday. Do not pay that price for a nightly report.

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

![Data quality](assets/collect/quality.png)

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
rows biases every conclusion you draw. Session 2 handles the mechanics; the
*diagnosis* belongs here, at collection, while you can still ask the source.

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

## The session in one line

> Collect deliberately, store the raw response, check at the boundary, and write
> down where it came from.

Everything after this session assumes you have a dataset you can defend. Building
one is the lab.
