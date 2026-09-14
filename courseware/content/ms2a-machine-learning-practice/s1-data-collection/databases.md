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
| Relationships | foreign keys and JOINs | denormalised, or joined in your code |
| Scaling | vertical, mostly | horizontal |
| Guarantees | ACID transactions | usually eventual consistency (BASE) |
| Suited to | complex queries, transactions | high throughput, evolving records |
| Examples | PostgreSQL, MySQL, SQLite | MongoDB, Redis, Neo4j |

For data science work you will meet PostgreSQL far more often than anything else.
Learn SQL properly; treat the rest as it comes.

---

## ACID and BASE

A relational database runs changes inside **transactions**, which are ACID:

- **Atomic** — every statement in the transaction applies, or none does
- **Consistent** — constraints (keys, types, `NOT NULL`) hold before and after
- **Isolated** — concurrent transactions do not see each other's half-work
- **Durable** — once committed, the change survives a crash

Distributed NoSQL stores usually relax this to **BASE** — *basically
available, soft state, eventually consistent*. Two reads a second apart can
disagree until the replicas converge. For collection, that means a count taken
from a replica is an estimate of the moment, not a fixed fact.

---

## Connecting

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/db")
df = pd.read_sql("SELECT * FROM orders LIMIT 5", engine)
```

SQLAlchemy gives one interface over Postgres, MySQL, SQLite and others, and
pandas speaks to it directly.

SQLite has no server at all: the database is one file, opened with
`create_engine("sqlite:///shop.db")`. It is the quickest way to get a real SQL
database for a project.

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

## Writing: the rest of CRUD

`SELECT` is the *Read* of CRUD. `INSERT`, `UPDATE` and `DELETE` are the others,
and each belongs in a transaction:

```python
with engine.begin() as conn:     # commits on exit, rolls back on error
    conn.execute(text("UPDATE accounts SET balance = balance - 100"
                      " WHERE id = :a"), {"a": 1})
    conn.execute(text("UPDATE accounts SET balance = balance + 100"
                      " WHERE id = :b"), {"b": 2})
```

If the second statement fails, the first is undone: no money disappears.

```python
df.to_sql("predictions", engine, if_exists="replace", index=False)
```

`to_sql` writes a whole dataframe as a table. `if_exists` is `"fail"` by
default — say explicitly whether you mean to replace or to append.

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

## Querying documents

```python
col.find({"preferences.theme": "dark", "age": {"$gt": 25}})

col.aggregate([
    {"$match": {"status": "active"}},
    {"$group": {"_id": "$country", "n_users": {"$sum": 1}}},
])
```

Dot notation reaches into nested fields; `$gt`, `$in` and `$exists` are
query operators.

The aggregation pipeline is MongoDB's `WHERE ... GROUP BY`: `$match` filters,
`$group` aggregates, both on the server. Pushing the work down applies here too.

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

## Try it on the course database

```python
engine = create_engine(os.environ["DATABASE_URL"])
stores = pd.read_sql("SELECT * FROM retail.stores", engine)
print(stores[["store_name", "city", "weekly_footfall"]])
```

Five rows, one per store of the Lab 1.3 challenge. `DATABASE_URL` is in the
*Access — today's sandbox* section of the Lab 1.1 page: put it in Colab's
*Secrets* panel or a gitignored `.env`, never in the code.

`retail.data_dictionary` describes every column of that challenge and the
source it comes from: read it before joining anything.
