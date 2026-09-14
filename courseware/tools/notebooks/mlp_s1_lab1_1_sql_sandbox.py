"""Lab 1.1 — Query a Shared Database: the Colab notebook.

The lesson page (`s1-data-collection/exercise-sql-sandbox.md`) introduces the
ideas and hands out the sandbox credentials; this notebook is the practical.
Its SQL and every number it prints come from the SQL sandbox's walkthrough
(`tests-dummy/catalog/sql_api_sandbox/course/sql.md` in the platform repo),
whose `make docs-test` checks them against the live database: a change there
comes here.

Writes one file and nothing else — unlike `build_notebooks.py`, running this
script is safe:

    uv run python courseware/tools/notebooks/mlp_s1_lab1_1_sql_sandbox.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "website" / "public" / "modules" / "ms2a-machine-learning-practice" / "challenges"
NAME = "mlp-s1-lab1-1-sql-sandbox.ipynb"
COLAB = (f"https://colab.research.google.com/github/{nb.GITHUB}/blob/{nb.BRANCH}/"
         f"website/public/modules/ms2a-machine-learning-practice/challenges/{NAME}")

md, code = nb.md, nb.code


def build() -> dict:
    cells = [
        md(
            "# Lab 1.1 — Query a Shared Database",
            "",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})",
            "",
            "A PostgreSQL server the whole class shares: a million orders, fifty",
            "thousand customers, and a segment history with a trap in it. Read its",
            "schema, make it do the work, catch a join that quietly adds 140,543",
            "rows — then write somewhere you are allowed to.",
            "",
            "**Time:** 40 minutes. **Deliverable:**",
            "`data/processed/customer_features_<date>.parquet`, and the numbers in",
            "the checklist at the end.",
            "",
            "Everyone queries the same seeded data, so every number below is exact",
            "for every student. A different number means different SQL.",
        ),
        md(
            "---",
            "",
            "## The database: `shop`",
            "",
            "| table | rows | columns | key |",
            "|---|---|---|---|",
            "| `customers` | 50,000 | `id`, `country`, `signup_date` | `id` |",
            "| `customer_segments` | 57,068 | `customer_id`, `segment` (`new`/`regular`/`vip`), `valid_from`, `valid_to` | `(customer_id, valid_from)` |",
            "| `orders` | 1,000,000 | `id`, `customer_id`, `created_at`, `amount_cents`, `currency`, `status` (`paid`/`refunded`/`cancelled`) | `id` |",
            "",
            "Orders run from 2024-01-01 to 2026-08-31. Indexes: `orders (created_at, id)`",
            "and `orders (customer_id)`.",
            "",
            "Two accounts: `DATABASE_URL` connects as `shop_reader`, which can only",
            "read; `WRITER_URL` as `playground_writer`, which can also create tables",
            "in schema `playground`. Every statement is cancelled after **30 s**.",
        ),
        md(
            "---",
            "",
            "## 0. Setup",
            "",
            "The notebook needs two secrets and your name. Never paste their values",
            "into a cell: a notebook is shared with its code and its outputs.",
            "",
            "| name | where it comes from |",
            "|---|---|",
            "| `DATABASE_URL` | the *Access — today's sandbox* section of the Lab 1.1 page |",
            "| `WRITER_URL` | the same section |",
            "",
            "In Colab, add both to the *Secrets* panel (the key icon on the left)",
            "and allow this notebook to access them, then write your name in the",
            "second cell below. Outside Colab, set the two variables and `STUDENT`",
            "in your environment before starting Jupyter; the cell raises if one is",
            "missing.",
        ),
        code(
            "import os",
            "import re",
            "import subprocess",
            "import sys",
            "import time",
            "from datetime import date",
            "",
            'IN_COLAB = "google.colab" in sys.modules',
            "if IN_COLAB:",
            '    subprocess.run([sys.executable, "-m", "pip", "install", "-q",',
            '                    "psycopg[binary]"], check=True)',
            "",
            "import pandas as pd",
            "from sqlalchemy import create_engine, inspect, text",
            "from sqlalchemy.exc import ProgrammingError",
        ),
        code(
            'SECRETS = ["DATABASE_URL", "WRITER_URL"]',
            "",
            "if IN_COLAB:",
            "    from google.colab import userdata",
            "    for name in SECRETS:",
            "        os.environ[name] = userdata.get(name)",
            "    # Your name: a-z, 0-9 and _, starting with a letter.",
            '    os.environ["STUDENT"] = "your_name"',
            "",
            'missing = [name for name in SECRETS + ["STUDENT"] if not os.environ.get(name)]',
            "if missing:",
            '    raise KeyError(f"not set: {missing}")',
            'STUDENT = os.environ["STUDENT"]',
            'if STUDENT == "your_name" or not re.fullmatch(r"[a-z][a-z0-9_]{1,31}",',
            "                                              STUDENT):",
            '    raise ValueError("STUDENT must be your own name: a-z, 0-9 and _, "',
            '                     "starting with a letter")',
            "",
            'engine = create_engine(os.environ["DATABASE_URL"])    # the reader',
            "with engine.connect() as con:",
            '    print(con.execute(text("SELECT current_user")).scalar_one())',
        ),
        md(
            "You should see `shop_reader`.",
            "",
            "The URLs use port 443 on purpose: some networks inspect PostgreSQL's",
            "usual port, 5432, and stall its encrypted handshake. If 443 fails where",
            "you are, replace `:443` by `:5432` in both secrets.",
        ),
        # ------------------------------------------------------------ Part A
        md(
            "---",
            "",
            "## Part A — Read the schema before you query (5 min)",
            "",
            "Primary keys and foreign keys say what one row of each table is. Read",
            "them before writing a single `SELECT`.",
        ),
        code(
            "insp = inspect(engine)",
            "for table in sorted(insp.get_table_names()):",
            '    pk = insp.get_pk_constraint(table)["constrained_columns"]',
            '    refs = [fk["referred_table"] for fk in insp.get_foreign_keys(table)]',
            '    print(f"{table:18} pk={pk} references={refs}")',
        ),
        md(
            "You should see:",
            "",
            "```text",
            "customer_segments  pk=['customer_id', 'valid_from'] references=['customers']",
            "customers          pk=['id'] references=[]",
            "orders             pk=['id'] references=['customers']",
            "```",
        ),
        code(
            'print(pd.read_sql("""',
            "    SELECT 'customers' AS tbl, count(*) AS n_rows FROM customers",
            "    UNION ALL SELECT 'customer_segments', count(*) FROM customer_segments",
            '    UNION ALL SELECT \'orders\', count(*) FROM orders""", engine)',
            "      .to_string(index=False))",
        ),
        md(
            "You should see 50,000 customers, 57,068 segment rows and 1,000,000",
            "orders.",
            "",
            "**Question.** One table has a `customer_id` that is not what its name",
            "suggests. Which one, and how do the two outputs above tell you?",
            "",
            "*Your answer:*",
        ),
        # ------------------------------------------------------------ Part B
        md(
            "---",
            "",
            "## Part B — Push the work down (6 min)",
            "",
            "Pull the whole table and filter in pandas. Run this **once**: with the",
            "room doing the same, the server may cancel it at 30 s, and that",
            "cancellation is the answer this part is about.",
        ),
        code(
            "t = time.time()",
            "try:",
            '    everything = pd.read_sql("SELECT * FROM orders", engine)',
            '    august = everything[everything["created_at"] >= "2026-08-01"]',
            '    print(len(everything), len(august), f"{time.time() - t:.1f} s")',
            "    del everything",
            "except pd.errors.DatabaseError as err:     # the 30 s statement timeout",
            '    print(f"failed after {time.time() - t:.1f} s:",',
            "          str(err.__cause__.orig).splitlines()[0])",
        ),
        md(
            "You should see `1000000 62879` and your time — or *canceling statement",
            "due to statement timeout*. Now let the database do the filtering:",
        ),
        code(
            "t = time.time()",
            "august = pd.read_sql(",
            "    \"SELECT * FROM orders WHERE created_at >= '2026-08-01'\", engine)",
            'print(len(august), f"{time.time() - t:.1f} s")',
        ),
        md(
            "The same 62,879 rows. The first query sent **72 MB** across the network",
            "to keep **4 MB** of it (measured at the server), and your machine built",
            "a million-row dataframe to throw 94% of it away.",
        ),
        # ------------------------------------------------------------ Part C
        md(
            "---",
            "",
            "## Part C — Parameters, not f-strings (5 min)",
            "",
            "One country in the table is `Côte d'Ivoire`. Assemble the query with an",
            "f-string:",
        ),
        code(
            "country = \"Côte d'Ivoire\"",
            'query = f"""SELECT count(*) AS n_orders FROM orders o',
            "            JOIN customers c ON o.customer_id = c.id",
            "            WHERE c.country = '{country}'\"\"\"",
            "try:",
            "    pd.read_sql(query, engine)",
            "except pd.errors.DatabaseError as err:",
            "    print(str(err.__cause__.orig).splitlines()[0])",
        ),
        md(
            "PostgreSQL answers *syntax error at or near \"Ivoire\"*: the apostrophe",
            "closed the string early. The data did it, not an attacker. Bind the",
            "value instead — the driver sends it separately from the SQL, so no",
            "character in it can change the query:",
        ),
        code(
            'query = text("""SELECT count(*) AS n_orders FROM orders o',
            "                JOIN customers c ON o.customer_id = c.id",
            '                WHERE c.country = :country""")',
            'print(pd.read_sql(query, engine, params={"country": country})',
            "      .to_string(index=False))",
        ),
        md(
            "You should see `20114`.",
            "",
            "**Question.** Run the f-string version with",
            "`country = \"x' OR '1'='1\"`. It does not fail. What does it count, and",
            "why is that worse than the syntax error?",
            "",
            "*Your answer:*",
        ),
        code(
            "# Try the f-string query with country = \"x' OR '1'='1\" here.",
        ),
        # ------------------------------------------------------------ Part D
        md(
            "---",
            "",
            "## Part D — The join that adds 140,543 rows (9 min)",
        ),
        code(
            "def count(sql):",
            "    with engine.connect() as con:",
            "        return con.execute(text(sql)).scalar_one()",
            "",
            'naive = count("""SELECT count(*) FROM orders o',
            '    JOIN customer_segments s ON o.customer_id = s.customer_id""")',
            "print(naive, naive - 1_000_000)",
        ),
        md(
            "You should see `1140543 140543`.",
            "",
            "`customer_segments` holds one row per customer *per period*: when a",
            "customer moves from `new` to `regular`, one row closes (`valid_to` is",
            "set) and the next opens. Joining on `customer_id` alone repeats each",
            "order once per period its customer has had.",
            "",
            "Two ways back to one row per order — keep only the current period, or",
            "join the period that was valid when the order was placed (*as-of*):",
        ),
        code(
            'current = count("""SELECT count(*) FROM orders o',
            "                   JOIN customer_segments s ON o.customer_id = s.customer_id",
            '                   WHERE s.valid_to IS NULL""")',
            'as_of = count("""SELECT count(*) FROM orders o',
            "                 JOIN customer_segments s ON o.customer_id = s.customer_id",
            "                  AND o.created_at >= s.valid_from",
            '                  AND (s.valid_to IS NULL OR o.created_at < s.valid_to)""")',
            "print(current, as_of)",
            "assert current == as_of == 1_000_000",
        ),
        md(
            "Both pass the row-count assertion. Do they give the same answer? Count",
            "the orders on which they disagree:",
        ),
        code(
            'print(count("""SELECT count(*) FROM orders o',
            "    JOIN customer_segments cur ON cur.customer_id = o.customer_id",
            "                              AND cur.valid_to IS NULL",
            "    JOIN customer_segments hist ON hist.customer_id = o.customer_id",
            "     AND o.created_at >= hist.valid_from",
            "     AND (hist.valid_to IS NULL OR o.created_at < hist.valid_to)",
            '    WHERE cur.segment <> hist.segment"""))',
        ),
        md(
            "You should see `50404`.",
            "",
            "**Question.** Which of the two fixes is right for building features, and",
            "why can no row-count assertion catch the wrong one?",
            "",
            "*Your answer:*",
        ),
        # ------------------------------------------------------------ Part E
        md(
            "---",
            "",
            "## Part E — A feature table, computed where the data is (6 min)",
            "",
            "One row per customer, aggregated by the database. Assert the grain and",
            "the total before you trust it.",
        ),
        code(
            'features = pd.read_sql("""',
            "    SELECT customer_id, COUNT(*) AS n_orders,",
            "           SUM(amount_cents) AS total_cents, MAX(created_at) AS last_order",
            "    FROM orders WHERE status = 'paid'",
            '    GROUP BY customer_id ORDER BY customer_id""",',
            '    engine, parse_dates=["last_order"])',
            "",
            'assert features["customer_id"].is_unique',
            'assert features["n_orders"].sum() == count(',
            "    \"SELECT count(*) FROM orders WHERE status = 'paid'\")",
            "print(features.shape)",
            "features.head(3)",
        ),
        md(
            "You should see `(48972, 4)`, and customer 1 with 38 orders and",
            "196,466 cents. Save it:",
        ),
        code(
            'os.makedirs("data/processed", exist_ok=True)',
            'path = f"data/processed/customer_features_{date.today()}.parquet"',
            "features.to_parquet(path, index=False)",
            "print(path, len(pd.read_parquet(path)))",
        ),
        md(
            "One row per customer who has paid at least once — 48,972 of the 50,000.",
            "The million orders never left the server; 48,972 rows did.",
        ),
        # ------------------------------------------------------------ Part F
        md(
            "---",
            "",
            "## Part F — Chunked reads, and what they do not do (4 min)",
        ),
        code(
            "paid = \"SELECT amount_cents FROM orders WHERE status = 'paid'\"",
            "with engine.connect().execution_options(stream_results=True) as con:",
            '    total = sum(chunk["amount_cents"].sum()',
            "                for chunk in pd.read_sql(paid, con, chunksize=100_000))",
            "print(total,",
            "      count(\"SELECT SUM(amount_cents) FROM orders WHERE status = 'paid'\"))",
        ),
        md(
            "You should see `4337694870 4337694870`.",
            "",
            "`chunksize` on its own bounds the dataframe, not the transfer: psycopg",
            "has received the whole result before pandas asks for the first chunk.",
            "`stream_results=True` is what makes the rows arrive as you consume them.",
            "Measured pulling all million orders, as peak memory of the Python",
            "process: no chunks **862 MB**, `chunksize` alone **392 MB**, both",
            "**296 MB**.",
            "",
            "**Question.** This answer was one number. Did you need the rows at all?",
            "",
            "*Your answer:*",
        ),
        # ------------------------------------------------------------ Part G
        md(
            "---",
            "",
            "## Part G — Write where you are allowed to (5 min)",
            "",
            "The account you have used so far can read and nothing else:",
        ),
        code(
            "try:",
            "    with engine.begin() as con:                     # the reader",
            '        con.execute(text("CREATE TABLE t (x int)"))',
            "except ProgrammingError as err:",
            "    print(str(err.orig).splitlines()[0])",
        ),
        md(
            "*permission denied for schema public.* Least privilege: the job that",
            "reads the data does not need to be able to damage it. Writing goes",
            "through the second account, into one schema, `playground`.",
            "",
            "A table name cannot be a bound parameter the way a value can, so your",
            "name was checked against a strict pattern in Setup *before* it goes into",
            "any SQL — the Part C rule, applied to identifiers.",
        ),
        code(
            'writer = create_engine(os.environ["WRITER_URL"])',
            'table = f"features_{STUDENT}"',
            'features.to_sql(table, writer, schema="playground",',
            '                if_exists="replace", index=False)',
            'print(len(pd.read_sql_table(table, writer, schema="playground")))',
        ),
        md(
            "You should see `48972`.",
            "",
            "The playground belongs to the whole class and is capped at **200 MB and",
            "200 tables**. Over either, the largest tables are dropped within ten",
            "seconds, whoever they belong to. How full is it?",
        ),
        code(
            'print(pd.read_sql("""',
            "    SELECT count(*) AS tables,",
            "      pg_size_pretty(coalesce(sum(pg_total_relation_size(c.oid)), 0))",
            "        AS used",
            "    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace",
            "    WHERE n.nspname = 'playground' AND c.relkind IN ('r', 'm')\"\"\",",
            "    writer).to_string(index=False))",
        ),
        md("Drop yours when you are done:"),
        code(
            "with writer.begin() as con:",
            '    con.execute(text(f"DROP TABLE playground.{table}"))',
            "print(inspect(writer).has_table(table, schema=\"playground\"))",
        ),
        md("You should see `False`."),
        # ------------------------------------------------------------ checklist
        md(
            "---",
            "",
            "## Did you validate this lab?",
            "",
            "- [ ] `DATABASE_URL` and `WRITER_URL` come from Secrets or the environment, and neither password is in your code, notebooks or git history",
            "- [ ] Part A: you can name the table whose `customer_id` is not unique, and point at the two outputs that show it",
            "- [ ] Part B: both queries return 62,879 rows, and you can say which one moved 72 MB",
            "- [ ] Part C: the f-string query fails on `Côte d'Ivoire`; the parameterised one returns 20,114",
            "- [ ] Part D: 1,140,543, then 1,000,000 and 1,000,000 — and the two fixes disagree on 50,404 orders; you can say which fix is right, and why no assertion catches the wrong one",
            "- [ ] Part E: your Parquet file has 48,972 rows, a unique `customer_id`, and `n_orders` summing to 899,914",
            "- [ ] Part F: the chunked sum and SQL's `SUM` both print 4,337,694,870",
            "- [ ] Part G: the reader account is refused; your table in `playground` reads back 48,972 rows, and you dropped it",
            "",
            "Part D's as-of join is the pattern for every \"what did we know at the",
            "time\" feature you build. Keep it.",
        ),
    ]
    return nb.notebook(cells)


def main() -> None:
    nb._COUNTER = itertools.count()     # byte-identical on every regeneration
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / NAME
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)}\n      {COLAB}")


if __name__ == "__main__":
    main()
