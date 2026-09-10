# Databases

The database is usually the source of truth. Reading from it well means pushing
the work to the server and pulling only the answer.

<!-- notes: 25 minutes. Most students only know `SELECT *`. The point to land is
that filtering and aggregation belong in SQL, not in pandas after the fact. -->

---

## Two families

| | SQL (relational) | NoSQL |
|---|---|---|
| Model | tables, fixed schema | documents, key-value, graph |
| Schema | declared up front | implicit, per record |
| Joins | built in | done in your code |
| Scaling | vertical, mostly | horizontal |
| Guarantees | ACID transactions | usually eventual consistency |

For data science work you will meet PostgreSQL far more often than anything else.
Learn SQL properly; treat the rest as it comes.

---

## Connecting

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/db")
df = pd.read_sql("SELECT * FROM orders LIMIT 5", engine)
```

SQLAlchemy gives one interface over Postgres, MySQL, SQLite and others, and
pandas speaks to it directly.

---

## Never put the password in the code

```python
import os
engine = create_engine(os.environ["DATABASE_URL"])   # KeyError if unset
```

`os.environ[...]`, not `os.getenv(..., "postgres://localhost")`. A default
connection string is how you end up reading an empty local database and
reporting that the table is missing.

Add `.env` to `.gitignore`. A credential in git history is a credential you have
to rotate.

---

## Push the work down

```python
# Reads 40 million rows to keep 12,000
df = pd.read_sql("SELECT * FROM orders", engine)
df = df[df.created_at >= "2025-01-01"]
```

```python
# Reads 12,000
df = pd.read_sql(
    "SELECT * FROM orders WHERE created_at >= '2025-01-01'", engine
)
```

The database has indexes, a query planner, and more memory than your laptop.
Filter, join and aggregate there.

---

## Parameters, not f-strings

```python
from sqlalchemy import text

q = text("SELECT * FROM orders WHERE country = :c AND amount > :a")
df = pd.read_sql(q, engine, params={"c": "FR", "a": 100})
```

String interpolation into SQL is an injection bug even when the input "comes from
a config file". It also breaks on any value containing a quote.

---

## Aggregate server-side

```sql
SELECT customer_id,
       COUNT(*)      AS n_orders,
       SUM(amount)   AS total,
       MAX(created_at) AS last_order
FROM orders
WHERE created_at >= '2025-01-01'
GROUP BY customer_id
```

This is a feature table. Computing it in SQL moves a hundred million rows of work
off your machine and returns one row per customer — which, if your grain is "one
customer", is exactly the dataframe you wanted.

---

## Chunked reads when the answer is still big

```python
chunks = pd.read_sql("SELECT * FROM events", engine, chunksize=100_000)
totals = sum(c["amount"].sum() for c in chunks)
```

`chunksize` returns an iterator instead of a dataframe. Use it when the result
genuinely does not fit in memory — and ask first whether an aggregate would have
answered the question.

On PostgreSQL it bounds the dataframe, not the transfer: the driver receives
the whole result first. Pass a connection opened with
`execution_options(stream_results=True)` to stream it.

---

## Joins duplicate rows

```sql
SELECT o.*, c.segment
FROM orders o
JOIN customers c ON o.customer_id = c.id
```

If `customers.id` is not unique, this join silently multiplies your rows. It is
the most common way a dataset quietly gains 12% more records than it should.

```python
assert len(df) == n_orders_before_join
```

Assert the row count across every join. Every time.

---

## NoSQL, briefly

```python
from pymongo import MongoClient
col = MongoClient(os.environ["MONGO_URL"]).app.events
docs = list(col.find({"status": "active"}, {"_id": 0, "user": 1, "ts": 1}))
df = pd.json_normalize(docs)
```

Same discipline, different syntax: filter in the query, project only the fields
you need, and expect the schema to vary between documents — because nothing
enforced it.

---

## Reading a schema you did not write

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'orders';
```

Before querying an unfamiliar database: list the tables, read the column types,
count the rows, and check what the primary and foreign keys actually are.

Ten minutes of that prevents a join on a column that looked like a key and was
not.

---

## Checklist

- credentials from the environment, never in the file
- filter, join and aggregate in SQL
- parameters, never f-strings
- assert the row count after every join
- write the result to Parquet once, then work from that

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import sqlite3, pandas as pd

   con = sqlite3.connect(":memory:")
   con.execute("CREATE TABLE orders (customer_id TEXT, amount REAL, created_at TEXT)")
   con.executemany("INSERT INTO orders VALUES (?,?,?)", [
       ("c1", 10.0, "2025-03-01"), ("c1", 20.0, "2025-04-01"),
       ("c2", 45.0, "2025-05-01"), ("c3",  5.0, "2024-12-01")])

   print(pd.read_sql("""
       SELECT customer_id, COUNT(*) AS n_orders, SUM(amount) AS total
       FROM orders WHERE created_at >= '2025-01-01'
       GROUP BY customer_id
   """, con).to_string(index=False))
   ```

   ```text
   customer_id  n_orders  total
            c1         2   30.0
            c2         1   45.0
   ```

   **Answer.** Four order rows went in and two customer rows came out: the filter
   and the aggregation both ran in the database. If your grain is "one customer",
   that is already the dataframe you wanted.

2. Why `os.environ["DATABASE_URL"]` and never
   `os.getenv("DATABASE_URL", "postgresql://localhost/db")`?

   **Answer.** The default silently connects you to an empty local database, and
   you report that the table is missing. `os.environ[...]` raises `KeyError` at
   the boundary instead.

3. After a `JOIN` on `customers`, your order table has 12% more rows than before.
   What happened, and which single line would have caught it?

   **Answer.** `customers.id` is not unique, so the join multiplied rows.
   `assert len(df) == n_orders_before_join`, after every join, every time.
