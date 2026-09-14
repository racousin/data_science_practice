# Data and Its Shapes

Before you can model anything you have to get the data, and the shape it
arrives in decides most of what happens next.

<!-- notes: 20 minutes. Set the frame: every model assumes a dataset exists;
this lesson is about where it comes from. Ask the room what data they will use
for their project — most have not thought about it. -->

---

## The only question that matters first

> What is one sample ? one row?

Answer it out before you write any code. One customer? One customer-month?
One click? One sensor reading? One image?

![A CUSTOMER table in which one row is one customer and each column one attribute](assets/collect/structured.jpg)

Everything downstream — the split, the metric, the leakage risk — follows from
that answer.

---

## Three shapes

| Shape | What it means | Typical source |
|---|---|---|
| **Structured** | Fixed schema, rows and columns | SQL tables, CSV, Parquet |
| **Semi-structured** | Self-describing, nested, no fixed schema | JSON, XML, logs |
| **Unstructured** | No schema at all | text, images, audio, video |

Structured data is the easiest to search and aggregate. Semi-structured data
trades that for flexibility: each record carries its own field names.
Unstructured data is by far the most abundant — and the most work to use.

---

## The same facts, three shapes

![The same student records as free text, as XML elements, and as a table with ID, Name, Age and Degree columns](assets/collect/data.png)

Three students, written three ways. The table can be filtered immediately. The
XML has to be parsed, but its tags name every field. From the free text, a
program has to *extract* the age and the degree — and extraction makes errors.


---

## Volume changes the tooling

| Size | Approach |
|---|---|
| < 1 GB | pandas, in memory |
| 1–100 GB | Parquet + column selection, chunked reads, Polars/DuckDB |
| > 100 GB | Distributed |

```python
# Read three columns of a 40 GB table without loading the other 200
df = pd.read_parquet("events.parquet", columns=["user_id", "ts", "amount"])
```

The single biggest performance lever in data collection is *not reading what you
do not need*.

---

## Where data comes from

Four sources:

1. **Files** — someone already exported it
2. **Databases** — the source of truth, queryable
3. **APIs** — a contract, rate-limited, versioned
4. **Scraping** — no contract, breaks silently, legally loaded

Prefer the highest ones on that list that gives you the data. Scraping is the
last resort.

---

## Data has a provenance

Every dataset you build should record, in the repository:

- **where** each field came from (source, endpoint, table, URL)
- **when** it was pulled (a timestamp, not "last week")
- **what** transformation was applied between the source and the file
- **which** licence or terms govern its use

```text
data/
  raw/          <- exactly what the source returned, never edited
  interim/      <- parsed, still one row per source record
  processed/    <- model-ready
```

`raw/` is immutable. If you find a bug, you fix the code that produces
`interim/`, and re-run.

---

## The cost you cannot see

A model is a week of work. A dataset is a month.

Every hour spent on collection design — the right grain, the right keys, a
recorded provenance — is repaid three times over during preprocessing and
modelling, when you discover that the join you did silently duplicated 12% of
your rows.

![0_xrw_SEDZvUKjU0KS.png](assets/collect/0_xrw_SEDZvUKjU0KS.png)
