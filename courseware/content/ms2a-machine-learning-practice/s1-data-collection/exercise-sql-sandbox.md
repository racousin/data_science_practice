# Lab 1.1 — Query a Shared Database

A PostgreSQL server the whole class shares: a million orders, fifty thousand
customers, and a segment history with a trap in it. Read its schema, make it do
the work, catch a join that quietly adds 140,543 rows — then write somewhere
you are allowed to.

**Time:** 40 minutes. **Deliverable:** `data/processed/customer_features_<date>.parquet`,
and the numbers in the checklist at the end.

<!-- notes: Everyone queries the same seeded data, so every number is exact for
every student — a different number means different SQL, and that is the
conversation to have. Part B is the moment: whoever pulls the whole table while
the room does the same waits, or is cancelled at 30 s, and both make the point.
The credentials below are today's (make info in sql_api_sandbox writes
LAB.local.md): replace them before each session. If the server is down, hand
out the offline copy: it loses only Part B's timings and Part G. -->

---

## The concept

A shared database is not a file. Other people query it at the same time, it
enforces who may do what, and every query you send costs everyone something.

- **Read the schema first.** Primary and foreign keys say what one row of each
  table is — before any `SELECT`.
- **Push the work down.** Filter, join and aggregate in SQL; bring back the
  answer, not the table.
- **Parameters, not f-strings.** A value spliced into SQL breaks on an
  apostrophe and opens the door to injection.

---

## The concept (continued)

- **Joins multiply rows.** Joining a history table on its id alone repeats
  rows; the fix that restores the row count can still be wrong.
- **Point in time.** Join the record that was valid *when the event happened*,
  or the features carry information from the future.
- **Streaming bounds memory, not need.** Chunked reads cap the dataframe; an
  aggregate in SQL removes the transfer.
- **Least privilege.** Read with an account that can only read; write with a
  second one, into one schema.

---

## The database: `shop`

| table | rows | key |
|---|---|---|
| `customers` | 50,000 | `id` |
| `customer_segments` | 57,068 | `(customer_id, valid_from)` |
| `orders` | 1,000,000 | `id` |

Two accounts: `shop_reader` reads the shop and the `retail` schema;
`playground_writer` can also create tables, in schema `playground` only.

Every statement is cancelled after **30 s**. The playground is shared by the
class and capped at **200 MB and 200 tables**, largest tables dropped first:
prefix yours with your name and drop them when you are done.

---

## Access — today's sandbox

These values are valid for this session only; the passwords change after it.

| variable | value |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://shop_reader:__READER_PASSWORD__@35.241.161.46:443/shop?sslmode=require` |
| `WRITER_URL` | `postgresql+psycopg://playground_writer:__WRITER_PASSWORD__@35.241.161.46:443/shop?sslmode=require` |
| `STUDENT` | your name: `a-z`, `0-9` and `_`, starting with a letter |

Put the two URLs in Colab's *Secrets* panel (the key icon on the left), or in
a gitignored `.env` — never in a cell. The database listens on port 443 because
some networks stall PostgreSQL's usual 5432; if 443 fails where you are, try
`:5432`.

---

## Open the notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s1-lab1-1-sql-sandbox.ipynb)

Seven parts, in order: read the schema (A), push the work down (B), bind
parameters (C), the join that adds rows (D), a feature table (E), chunked reads
(F), and writing to the playground (G). Every cell prints a number you can check
against the notebook's text.

---

## Did you validate this lab?

- [ ] `DATABASE_URL` and `WRITER_URL` come from Secrets or `.env`, and neither password is in your code, notebooks or git history
- [ ] Part A: you can name the table whose `customer_id` is not unique, and point at the two outputs that show it
- [ ] Part B: both queries return 62,879 rows, and you can say which one moved 72 MB
- [ ] Part C: the f-string query fails on `Côte d'Ivoire`; the parameterised one returns 20,114
- [ ] Part D: 1,140,543, then 1,000,000 and 1,000,000 — and the two fixes disagree on 50,404 orders; you can say which fix is right, and why no assertion catches the wrong one

---

## Did you validate this lab? (continued)

- [ ] Part E: your Parquet file has 48,972 rows, a unique `customer_id`, and `n_orders` summing to 899,914
- [ ] Part F: the chunked sum and SQL's `SUM` both print 4,337,694,870
- [ ] Part G: the reader account is refused; your table in `playground` reads back 48,972 rows, and you dropped it

Part D's as-of join is the pattern for every "what did we know at the time"
feature you build. Keep it.
