# Data and Its Shapes

Before you can model anything you have to get the data, and the shape it
arrives in decides most of what happens next.

<!-- notes: 20 minutes. Set the frame: every model assumes a dataset exists;
this lesson is about where it comes from. Ask the room what data they will use
for their project — most have not thought about it. -->

---

## The only question that matters first

> What is one row?

Answer it out loud before you write any code. One customer? One customer-month?
One click? One sensor reading? One image?

![A CUSTOMER table in which one row is one customer and each column one attribute](assets/collect/structured.jpg)

Everything downstream — the split, the metric, the leakage risk — follows from
that answer, and getting it wrong is not a bug you find in a unit test.

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

## Why the shape matters

Structured data is ready for a model after cleaning. Unstructured data needs a
*representation* step first — a CNN, a tokenizer, an embedding model — and
building that representation is a modelling problem in its own right.

Semi-structured is the trap in the middle: it looks tabular in the first hundred
records and stops being tabular at record 4,000, when a field that was a string
turns into a list.

---

## Modalities you will actually meet

- **Tabular** — the default, and still where gradient-boosted trees win
- **Text** — free-form, needs tokenization before any model sees it
- **Images** — tensors of pixels, height × width × channels
- **Time series** — rows are *ordered*, which breaks the usual random split
- **Interaction traces** — states, actions, rewards, logged by an agent

---

## Volume changes the tooling

| Size | Approach |
|---|---|
| < 1 GB | pandas, in memory, no ceremony |
| 1–50 GB | Parquet + column selection, chunked reads, Polars/DuckDB |
| > 50 GB | out-of-core or distributed; push the work into the database |

```python
# Read three columns of a 40 GB table without loading the other 200
df = pd.read_parquet("events.parquet", columns=["user_id", "ts", "amount"])
```

The single biggest performance lever in data collection is *not reading what you
do not need*.

---

## Where data comes from

Four sources, in rough order of how much you should like them:

1. **Files** — someone already exported it
2. **Databases** — the source of truth, queryable
3. **APIs** — a contract, rate-limited, versioned
4. **Scraping** — no contract, breaks silently, legally loaded

Prefer the highest one on that list that gives you the data. Scraping is the
last resort, not the first exercise.

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

---

## One question per source

| Source | Question to answer |
|---|---|
| Files | It is already a file. How do I read it correctly? |
| Databases | It is in a database. How do I query it without melting it? |
| APIs | Someone exposes it. How do I pull it reliably? |
| Web scraping | Nobody exposes it. What are my options and my obligations? |
| Any source | Batch or stream, and what do I check before I trust it? |

---

## Check yourself

1. Before you write any code against a new dataset, which single question do you
   answer first, and what depends on the answer?

   **Answer.** *What is one row?* The split, the metric and the leakage risk all
   follow from the grain, and getting it wrong is not a bug a unit test finds.

2. Run this. You should get exactly the output shown.

   ```python
   import pandas as pd

   df = pd.DataFrame({"customer_id": ["c1", "c1", "c2"],
                      "month": ["2026-01", "2026-02", "2026-01"],
                      "spend": [10, 20, 30]})
   print(len(df), df["customer_id"].nunique())          # -> 3 2
   print(df.duplicated(subset=["customer_id"]).any())   # -> True
   ```

   **Answer.** One row here is not one customer, it is one customer-month —
   three rows, two customers. Anything that treats this table as one row per
   customer is already wrong.

3. The table you want is published as an HTML page, exposed through a documented
   JSON API, and also shipped as a monthly CSV export. Which do you take, and why
   is the scrape last on the list?

   **Answer.** The CSV export. The order of preference is files, databases, APIs,
   scraping; scraping has no contract, no versioning, breaks silently on a
   redesign and is legally loaded.
