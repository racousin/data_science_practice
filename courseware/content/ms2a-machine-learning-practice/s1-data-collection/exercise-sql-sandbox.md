# Exercise — Query a Shared Database

A PostgreSQL server the whole class shares: a million orders, fifty thousand
customers, and a segment history with a trap in it. Read its schema, make it do
the work, catch a join that quietly adds 140,543 rows — then write somewhere
you are allowed to.

**Time:** 40 minutes. **Deliverable:** `data/processed/customer_features_<date>.parquet`,
and the numbers in the checklist at the end.

<!-- notes: Everyone queries the same seeded data, so every number below is
exact for every student — a different number means different SQL, and that is
the conversation to have. Part B is the moment: whoever pulls the whole table
while the room does the same waits, or is cancelled at 30 s, and both make the
point. If the server is down, hand out the offline copy: it loses only Part B's
timings and Part G. -->

---

## The database: `shop`

Schema `public`, generated from a fixed seed: every student has the same rows,
so every query has the same answer.

| table | rows | columns | key |
|---|---|---|---|
| `customers` | 50,000 | `id`, `country`, `signup_date` | `id` |
| `customer_segments` | 57,068 | `customer_id`, `segment` (`new`/`regular`/`vip`), `valid_from`, `valid_to` | `(customer_id, valid_from)` |
| `orders` | 1,000,000 | `id`, `customer_id`, `created_at`, `amount_cents`, `currency` (`EUR`/`GBP`/`CHF`), `status` (`paid`/`refunded`/`cancelled`) | `id` |

Orders run from 2024-01-01 to 2026-08-31, denser as the shop grows. Indexes:
`orders (created_at, id)` and `orders (customer_id)`.

---

## Three traps, planted on purpose

- **A history table.** `customer_segments` holds one row per customer *per
  period*, with `valid_to` NULL on the current one. Joining on `customer_id`
  alone multiplies orders.
- **An apostrophe.** One country is `Côte d'Ivoire`: an SQL query assembled
  with an f-string breaks on it.
- **A million orders.** Pulling them all to filter in pandas is visibly slow —
  72 MB on the wire for a 4 MB answer.

Each one is a part of this exercise.

---

## Two accounts

| | `shop_reader` | `playground_writer` |
|---|---|---|
| read the shop tables and `retail` | yes | yes |
| create, fill and drop tables | no | in schema `playground` only |
| default schema | `public` | `playground`, then `public` |
| connections, whole class | 60 | 30 |

`DATABASE_URL` connects as `shop_reader`, `WRITER_URL` as `playground_writer`.
Use the reader for everything that only reads.

---

## Limits, for both accounts

- **30 s per statement.** Longer statements are cancelled, and any statement
  older than 60 s is ended even if you set the timeout to 0.
- **Idle connections.** An idle transaction is closed after 30 s, an idle
  session after 15 min.
- **Blocked outright:** temporary tables and large objects.
- **The playground** is one schema for the whole class, capped at **200 MB and
  200 tables**. Over either cap, the largest tables are dropped within ten
  seconds, whoever they belong to.

Prefix every table you create with your name, and drop it when you are done.

---

## The `retail` schema

- `retail.stores` (one row per store, including `weekly_footfall`) and
  `retail.data_dictionary` (each column of the store-sales dataset and its
  source) are the database part of the store-sales challenge.
- `retail.leaderboard` is its predictions board: a `playground` table named
  `<your name>_predictions` (`item_code`, `quantity_sold`) is scored there
  within ten to twenty seconds, on half the items — a hint, not the grade.

---

## Setup — three values

| variable | what it is |
|---|---|
| `DATABASE_URL` | the reader account's connection string |
| `WRITER_URL` | the writer account's connection string |
| `STUDENT` | your name: a-z, 0-9 and _, starting with a letter |

The two URLs are in the *Today's sandbox* section the teacher posts on the
Lab 1 page. They are credentials, so they go in `.env` (gitignored) or Colab's
*Secrets* panel, never in a cell.

In Colab: `from google.colab import userdata`, then
`os.environ["DATABASE_URL"] = userdata.get("DATABASE_URL")`, and the same for the
other two.

---

## Setup — connect

```bash
uv pip install "psycopg[binary]" sqlalchemy pandas pyarrow
```

In Colab, `pip install "psycopg[binary]"` is enough.

```python
import os
import pandas as pd
from sqlalchemy import create_engine, inspect, text

engine = create_engine(os.environ["DATABASE_URL"])     # KeyError if unset
```

The URLs use port 443 on purpose. Some networks inspect PostgreSQL's usual port,
5432, and stall its encrypted handshake. The same server also listens on 5432.

If the server is down, the teacher provides an offline copy of the database.

---

## Part A — Read the schema before you query (5 min)

```python
insp = inspect(engine)
for table in sorted(insp.get_table_names()):
    pk = insp.get_pk_constraint(table)["constrained_columns"]
    refs = [fk["referred_table"] for fk in insp.get_foreign_keys(table)]
    print(f"{table:18} pk={pk} references={refs}")
```

```text
customer_segments  pk=['customer_id', 'valid_from'] references=['customers']
customers          pk=['id'] references=[]
orders             pk=['id'] references=['customers']
```

---

## Part A — Count the rows

```python
print(pd.read_sql("""
    SELECT 'customers' AS tbl, count(*) AS n_rows FROM customers
    UNION ALL SELECT 'customer_segments', count(*) FROM customer_segments
    UNION ALL SELECT 'orders', count(*) FROM orders""", engine)
      .to_string(index=False))
```

```text
              tbl  n_rows
        customers   50000
customer_segments   57068
           orders 1000000
```

One table has a `customer_id` that is not what its name suggests. Which, and
how do the two outputs tell you? Write it down: it is Part D.

---

## Part B — Push the work down (6 min)

Run this **once**.

```python
import time

t = time.time()
everything = pd.read_sql("SELECT * FROM orders", engine)
august = everything[everything["created_at"] >= "2026-08-01"]
print(len(everything), len(august), f"{time.time() - t:.1f} s")
del everything
```

```text
1000000 62879 <your time> s
```

---

## Part B — Filter in SQL instead

```python
t = time.time()
august = pd.read_sql(
    "SELECT * FROM orders WHERE created_at >= '2026-08-01'", engine)
print(len(august), f"{time.time() - t:.1f} s")
```

```text
62879 <your time> s
```

---

## Part B — What the first query cost

The same 62,879 rows. The first query sent **72 MB** across the network to keep
**4 MB** of it (measured at the server), and your machine built a million-row
dataframe to throw 94% of it away.

On the shared server, with the room doing the same, the first query takes tens
of seconds — or the server cancels it at 30 s with *canceling statement due to
statement timeout*. That is not the server failing. It is the point of this
part, said by the database.

---

## Part C — Parameters, not f-strings (5 min)

```python
country = "Côte d'Ivoire"
query = f"""SELECT count(*) AS n_orders FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE c.country = '{country}'"""
pd.read_sql(query, engine)             # raises DatabaseError
```

PostgreSQL answers *syntax error at or near "Ivoire"*: the apostrophe closed the
string early. The data did it, not an attacker.

---

## Part C — Bind the value

```python
query = text("""SELECT count(*) AS n_orders FROM orders o
                JOIN customers c ON o.customer_id = c.id
                WHERE c.country = :country""")
print(pd.read_sql(query, engine, params={"country": country})
      .to_string(index=False))
```

```text
 n_orders
    20114
```

The f-string version, run with `country = "x' OR '1'='1"`, does not fail. What
does it count, and why is that worse than the syntax error?

---

## Part D — The join that adds 140,543 rows (9 min)

```python
def count(sql):
    with engine.connect() as con:
        return con.execute(text(sql)).scalar_one()

naive = count("""SELECT count(*) FROM orders o
    JOIN customer_segments s ON o.customer_id = s.customer_id""")
print(naive, naive - 1_000_000)
```

```text
1140543 140543
```

`customer_segments` holds one row per customer *per period*: when a customer
moves from `new` to `regular`, one row closes (`valid_to` is set) and the next
opens. Joining on `customer_id` alone repeats each order once per period its
customer has had.

---

## Part D — Two ways back to one row per order

```python
current = count("""SELECT count(*) FROM orders o
                   JOIN customer_segments s ON o.customer_id = s.customer_id
                   WHERE s.valid_to IS NULL""")
as_of = count("""SELECT count(*) FROM orders o
                 JOIN customer_segments s ON o.customer_id = s.customer_id
                  AND o.created_at >= s.valid_from
                  AND (s.valid_to IS NULL OR o.created_at < s.valid_to)""")
print(current, as_of)
```

```text
1000000 1000000
```

Both pass the row-count assertion.

---

## Part D — They do not give the same answer

```python
print(count("""SELECT count(*) FROM orders o
    JOIN customer_segments cur ON cur.customer_id = o.customer_id
                              AND cur.valid_to IS NULL
    JOIN customer_segments hist ON hist.customer_id = o.customer_id
     AND o.created_at >= hist.valid_from
     AND (hist.valid_to IS NULL OR o.created_at < hist.valid_to)
    WHERE cur.segment <> hist.segment"""))
```

```text
50404
```

50,404 orders carry, under the first fix, a segment their customer did not have
yet when they ordered — `vip` stamped on a purchase made while they were `new`.
Fed to a model, that is information from the future — leakage — and no row
count will ever show it. The *as-of* join is the right one.

---

## Part E — A feature table, computed where the data is (6 min)

```python
features = pd.read_sql("""
    SELECT customer_id, COUNT(*) AS n_orders,
           SUM(amount_cents) AS total_cents, MAX(created_at) AS last_order
    FROM orders WHERE status = 'paid'
    GROUP BY customer_id ORDER BY customer_id""",
    engine, parse_dates=["last_order"])

assert features["customer_id"].is_unique
assert features["n_orders"].sum() == count(
    "SELECT count(*) FROM orders WHERE status = 'paid'")
print(features.shape)
print(features.head(3).to_string(index=False))
```

```text
(48972, 4)
 customer_id  n_orders  total_cents          last_order
           1        38       196466 2026-08-18 23:23:31
           2        35       179282 2026-08-23 16:23:54
           3        26       121190 2026-08-15 19:34:20
```

---

## Part E — Save it

```python
from datetime import date

os.makedirs("data/processed", exist_ok=True)
features.to_parquet(
    f"data/processed/customer_features_{date.today()}.parquet",
    index=False)
```

One row per customer who has paid at least once — 48,972 of the 50,000. The
million orders never left the server; 48,972 rows did.

---

## Part F — Chunked reads, and what they do not do (4 min)

```python
paid = "SELECT amount_cents FROM orders WHERE status = 'paid'"
with engine.connect().execution_options(stream_results=True) as con:
    total = sum(chunk["amount_cents"].sum()
                for chunk in pd.read_sql(paid, con, chunksize=100_000))
print(total,
      count("SELECT SUM(amount_cents) FROM orders WHERE status = 'paid'"))
```

```text
4337694870 4337694870
```

---

## Part F — Bounding the dataframe is not bounding the transfer

`chunksize` on its own bounds the dataframe, not the transfer: psycopg has
received the whole result before pandas asks for the first chunk.
`stream_results=True` is what makes the rows arrive as you consume them.

Measured pulling all million orders, as the peak memory of the Python process:
no chunks **862 MB**, `chunksize` alone **392 MB**, both **296 MB** — and the
first chunk arrived after 0.45 s without streaming, 0.13 s with it.

Then ask whether you needed the rows at all: this answer was one number, and
`SUM` sent it as one row.

---

## Part G — Write where you are allowed to (5 min)

The account you have used so far can read and nothing else:

```python
# requires WRITER_URL
with engine.begin() as con:                        # the read-only account
    con.execute(text("CREATE TABLE t (x int)"))    # raises ProgrammingError
```

*permission denied for schema public.* Least privilege: the job that reads the
data does not need to be able to damage it. Writing goes through a second
account, into one schema, `playground`.

---

## Part G — Your table in the playground

```python
# requires WRITER_URL
import re

me = os.environ["STUDENT"]                  # your name: a-z, 0-9 and _
# it becomes part of a table name
assert re.fullmatch(r"[a-z][a-z0-9_]{1,31}", me)
writer = create_engine(os.environ["WRITER_URL"])

features.to_sql(f"features_{me}", writer, schema="playground",
                if_exists="replace", index=False)
print(len(pd.read_sql_table(f"features_{me}", writer, schema="playground")))
```

```text
48972
```

A table name cannot be a bound parameter the way a value can, so the name is
checked against a strict pattern *before* it goes into any SQL — the Part C
rule, applied to identifiers.

---

## Part G — Share the playground

It belongs to the whole class and is capped at **200 MB and 200 tables**. Over
either, the largest tables are dropped within ten seconds, whoever they belong
to. How full is it?

```python
# requires WRITER_URL
print(pd.read_sql("""
    SELECT count(*) AS tables,
      pg_size_pretty(coalesce(sum(pg_total_relation_size(c.oid)), 0))
        AS used
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'playground' AND c.relkind IN ('r', 'm')""",
    writer).to_string(index=False))
```

Drop yours when you are done:

```python
# requires WRITER_URL
with writer.begin() as con:
    con.execute(text(f"DROP TABLE playground.features_{me}"))
```
