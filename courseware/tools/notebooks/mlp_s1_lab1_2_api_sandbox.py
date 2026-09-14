"""Lab 1.2 of MS2A Machine Learning Practice, Session 1 — the API sandbox notebook.

    uv run python courseware/tools/notebooks/mlp_s1_lab1_2_api_sandbox.py

writes website/public/modules/ms2a-machine-learning-practice/challenges/
mlp-s1-lab1-2-api-sandbox.ipynb, and nothing else. It borrows the cell helpers of
`build_notebooks.py` by import: running that file instead would regenerate
every notebook it owns.

The walkthrough is the one on the lesson page before it was cut down to a
brief (`s1-data-collection/exercise-api-sandbox`), itself a copy of
tests-dummy/catalog/sql_api_sandbox/course/api.md in the platform repo, whose
`make docs-test` checks the printed numbers against the live sandbox. A change
to the numbers goes to all three.

The credentials are never in the notebook: they come from Colab's Secrets
panel, or from the environment outside Colab, and the session's key rotates.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import build_notebooks as nb  # noqa: E402  (the helpers; main() is not run)
from build_notebooks import code, md, notebook  # noqa: E402

NAME = "mlp-s1-lab1-2-api-sandbox.ipynb"
REL = f"website/public/modules/ms2a-machine-learning-practice/challenges/{NAME}"
OUT = nb.REPO / REL
COLAB = f"https://colab.research.google.com/github/{nb.GITHUB}/blob/{nb.BRANCH}/{REL}"


def build() -> dict:
    cells = [
        md(
            "# Lab 1.2 — A Client That Survives the API",
            "",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})",
            "",
            "A small practice API that misbehaves on a schedule: it rate-limits you,",
            "drops requests, answers slowly, and serves a feed busy enough that offset",
            "pagination repeats itself. Build a client with a timeout, retries with",
            "backoff, cursor pagination and resumable raw writes, watch each piece earn",
            "its place, then ask the API what your client actually did.",
            "",
            "**Time:** 40 minutes. **Deliverable:** `data/raw/orders_since_2026-08-01.jsonl`,",
            "and the numbers in the checklist at the end.",
            "",
            "Run each cell **once**, in order, under one name. The API keeps a history per",
            "name, and the numbers below assume a fresh one.",
        ),
        md(
            "## The API",
            "",
            "`GET /` is the documentation page; `GET /docs` is the OpenAPI reference.",
            "",
            "| endpoint | returns |",
            "|---|---|",
            "| `GET /v1/me` | your name and remaining rate budget |",
            "| `GET /v1/me/stats` | your requests per endpoint and status, and `early_retries_after_429` |",
            "| `GET /v1/orders` | orders, cursor-paginated; filters `since` (inclusive), `until` (exclusive), `status`, `customer_id` |",
            "| `GET /v1/customers/{id}` | one customer and their current segment; 404 if unknown |",
            "| `GET /v1/events` | a feed, newest first, by `offset` or `cursor`; three new events per call |",
            "| `GET /v1/fx/{YYYY-MM-DD}` | EUR→USD/GBP/CHF reference rates for one day |",
            "",
            "**Authentication.** Every `/v1` call sends `Authorization: Bearer <API_KEY>`",
            "(missing or wrong: **401**) and `X-Student: <STUDENT>` (missing or malformed:",
            "**400**).",
            "",
            "**Pagination.** `limit` is 1–500. Pass the response's `next_cursor` back as",
            "`cursor`; it is `null` on the last page. `/v1/orders` supports cursors only;",
            "`/v1/events` takes `offset` *or* `cursor`, not both.",
            "",
            "**Rate limit.** 60 requests per minute, per student. Every metered response",
            "carries `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`",
            "(Unix time); a 429 carries `Retry-After` in seconds. `/v1/me` and",
            "`/v1/me/stats` are free.",
            "",
            "**Failures.** `/v1/fx` misbehaves on a fixed schedule, kept per student:",
            "",
            "- Weekends, and dates outside 2024-01-01 … 2026-08-31, return 404: no rate is",
            "  published.",
            "- Some days fail once with `503` and `Retry-After: 1`; a retry succeeds.",
            "- Some days answer slowly once, after 30 s; a retry is fast.",
            "- Leave a day alone for a minute and its failure comes back.",
        ),
        md(
            "## Setup",
            "",
            "| variable | what it is |",
            "|---|---|",
            "| `SANDBOX_URL` | the API's base URL |",
            "| `API_KEY` | the class key, `sbx_…` |",
            "| `STUDENT` | your name: a-z, 0-9 and _, starting with a letter |",
            "",
            "`SANDBOX_URL` and `API_KEY` are in the *Access — today's sandbox* section of",
            "the Lab 1.2 page on ML-Arena. In Colab, add both in the **Secrets** panel (the",
            "key icon on the left) and allow this notebook to read them. Outside Colab,",
            "export them in your shell or a gitignored `.env`. Never paste them into a",
            "cell: the key is the class's, and it rotates after the session.",
        ),
        code("%pip install --quiet requests"),
        code(
            "import json, os, re, sys, time",
            "from datetime import date, timedelta",
            "",
            "import requests",
            "",
            'if "google.colab" in sys.modules:',
            "    from google.colab import userdata",
            '    for name in ["SANDBOX_URL", "API_KEY"]:',
            "        os.environ[name] = userdata.get(name)",
            "",
            'BASE = os.environ["SANDBOX_URL"]          # KeyError if unset',
            'API_KEY = os.environ["API_KEY"]',
        ),
        md(
            "Your name. It is what keeps your rate limit, your failures and your",
            "statistics apart from everyone else's — pick one and keep it.",
        ),
        code(
            'STUDENT = "your_name"      # a-z, 0-9 and _, starting with a letter',
            "",
            'assert re.fullmatch(r"[a-z][a-z0-9_]*", STUDENT), STUDENT',
            'assert STUDENT != "your_name", "set STUDENT to a name of your own"',
            'HEADERS = {"Authorization": f"Bearer {API_KEY}",',
            '           "X-Student": STUDENT}',
        ),
        md(
            "## Part A — Read the documentation first (3 min)",
        ),
        code("print(requests.get(BASE, timeout=10).text)"),
        md(
            "Find the four things any API's documentation must tell you on that page —",
            "the endpoints, the authentication, the pagination, the rate limit — and",
            "write them down before calling anything else. The last sentence of the page",
            "matters in Part D.",
        ),
        md("## Part B — Status codes (4 min)"),
        code(
            'r = requests.get(f"{BASE}/v1/me", timeout=10)',
            'print(r.status_code, r.json()["error"])                  # no key',
            'r = requests.get(f"{BASE}/v1/me", timeout=10,',
            '                 headers={"Authorization": HEADERS["Authorization"]})',
            'print(r.status_code, r.json()["error"])                  # the key, no name',
            'r = requests.get(f"{BASE}/v1/me", headers=HEADERS, timeout=10)',
            'print(r.status_code, r.json()["rate_limit"]["limit"])    # both',
            'r = requests.get(f"{BASE}/v1/customers/999999", headers=HEADERS, timeout=10)',
            'print(r.status_code, r.json()["error"])                  # nobody by that id',
        ),
        md(
            "You should see:",
            "",
            "```text",
            "401 missing_key",
            "400 missing_student",
            "200 60",
            "404 not_found",
            "```",
            "",
            "401 is about who you are, 400 about what you sent, 404 about what you asked",
            "for. **Which of the three would a retry fix?**",
        ),
        md(
            "## Part C — Pagination on a busy feed (7 min)",
            "",
            "The documentation says three new events arrive between any two of your",
            "calls. Offset pagination first:",
        ),
        code(
            "ids = []",
            "for page in range(10):",
            '    r = requests.get(f"{BASE}/v1/events", headers=HEADERS, timeout=10,',
            '                     params={"offset": 50 * page, "limit": 50})',
            "    r.raise_for_status()",
            '    ids += [event["id"] for event in r.json()["data"]]',
            "print(len(ids), len(set(ids)))",
        ),
        md(
            "You should see `500 473`.",
            "",
            "Now the cursor:",
        ),
        code(
            "ids, cursor = [], None",
            "for page in range(10):",
            '    r = requests.get(f"{BASE}/v1/events", headers=HEADERS, timeout=10,',
            '                     params={"cursor": cursor, "limit": 50})',
            "    r.raise_for_status()",
            "    body = r.json()",
            '    ids += [event["id"] for event in body["data"]]',
            '    cursor = body["next_cursor"]',
            "print(len(ids), len(set(ids)))",
        ),
        md(
            "You should see `500 500`.",
            "",
            "An offset counts from the newest event, and the newest keeps changing. Each",
            "page therefore starts three events early, and its first three are the last",
            "three of the page before: 27 duplicates in ten pages. A cursor says *after",
            "this record*, which no new arrival can move. Where records are also deleted,",
            "offset skips rows as well — and a skipped row, unlike a duplicate, you never",
            "see.",
        ),
        md(
            "## Part D — Retries and timeouts (10 min)",
            "",
            "Reference rates for every day of June 2026, written the way most first",
            "attempts are — no timeout, no retry. The loop is wrapped in `try` only so",
            "that the notebook carries on after it dies.",
        ),
        code(
            "june = [date(2026, 6, 1) + timedelta(days=i) for i in range(30)]",
            "rates = {}",
            "try:",
            "    for day in june:",
            '        r = requests.get(f"{BASE}/v1/fx/{day}", headers=HEADERS)',
            "        if r.status_code == 404:       # a weekend: no rate is published",
            "            continue",
            "        r.raise_for_status()",
            '        rates[day] = r.json()["rates"]',
            "except requests.HTTPError as err:",
            '    print(len(rates), "rates, then:", err)',
        ),
        md(
            "You should see `3 rates, then: 503 Server Error: Service Unavailable for url:",
            "…/v1/fx/2026-06-04`. Three good days, then the upstream is unavailable on",
            "4 June and the loop dies with it.",
            "",
            "### The missing timeout",
            "",
            "This day is slow once:",
        ),
        code(
            "t = time.time()",
            'requests.get(f"{BASE}/v1/fx/2026-06-17", headers=HEADERS)',
            'print(f"{time.time() - t:.0f} s")',
        ),
        md(
            "You should see `30 s`. Thirty seconds for one call. Here the server answers",
            "in the end; a hung connection in production does not, and the overnight job",
            "produces nothing.",
            "",
            "### A session that retries",
            "",
            "A timeout on every call, and backoff on 429 and 5xx:",
        ),
        code(
            "from requests.adapters import HTTPAdapter, Retry",
            "",
            "session = requests.Session()",
            "session.headers.update(HEADERS)",
            "session.mount(BASE, HTTPAdapter(max_retries=Retry(",
            "    total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))",
        ),
        md("July, which you have not touched yet, through that session:"),
        code(
            "july = [date(2026, 7, 1) + timedelta(days=i) for i in range(31)]",
            "rates, skipped, t = {}, [], time.time()",
            "for day in july:",
            '    r = session.get(f"{BASE}/v1/fx/{day}", timeout=5)',
            "    if r.status_code == 404:",
            "        skipped.append(day)",
            "        continue",
            "    r.raise_for_status()",
            '    rates[day] = r.json()["rates"]',
            'print(len(rates), len(skipped), f"{time.time() - t:.0f} s")',
        ),
        md(
            "You should see `23 8` and about 7 s.",
            "",
            "### Ask the server what your client did",
            "",
            "Every weekday arrived. Mounting the adapter on `BASE` rather than on",
            '`"https://"` gives this API\'s retry policy to this API only. Where did the',
            "seconds go?",
        ),
        code(
            'fx = (session.get(f"{BASE}/v1/me/stats", timeout=10)',
            '      .json()["endpoints"]["fx"]["by_status"])',
            'print(fx["503"], fx["404"])',
        ),
        md(
            "You should see `3 8`.",
            "",
            "Three 503s: 4 June, and two in July that your loop never noticed, each",
            "retried after the one second `Retry-After` asked for. Eight weekends. And on",
            "17 July a read timeout at 5 s, retried at once — a timeout counts against",
            "`total` too.",
            "",
            "### The request you gave up on",
            "",
            "Run the next cell now, and again in a minute.",
        ),
        code(
            'print(session.get(f"{BASE}/v1/me/stats", timeout=10)',
            '      .json()["endpoints"]["fx"]["by_status"]["200"])',
        ),
        md(
            "It goes up by one without you calling anything. The request your client gave",
            "up on was still served. The server does the work whether or not you wait for",
            "it.",
        ),
        md(
            "## Part E — Through the rate limit (8 min)",
            "",
            "Every order since 1 August. The limit is 60 requests a minute and this takes",
            "more than 60.",
            "",
            "`collect` appends each page raw, as it arrives, and resumes from the last",
            "page's `next_cursor` when the file already exists.",
        ),
        code(
            "def collect(session, path):",
            '    """Append every page to `path`; resume from the last page\'s',
            '    next_cursor."""',
            "    cursor = None",
            "    if os.path.exists(path):",
            "        with open(path) as f:",
            "            lines = f.read().splitlines()",
            "        if lines:",
            '            cursor = json.loads(lines[-1])["next_cursor"]',
            "            if cursor is None:",
            "                return 0                   # already complete",
            "    pages = 0",
            '    with open(path, "a") as f:',
            "        while True:",
            '            r = session.get(f"{BASE}/v1/orders", timeout=10, params={',
            '                "since": "2026-08-01", "limit": 500, "cursor": cursor})',
            "            r.raise_for_status()",
            "            body = r.json()",
            '            f.write(json.dumps(body) + "\\n")',
            "            pages += 1",
            '            cursor = body["next_cursor"]',
            "            if cursor is None:",
            "                return pages",
        ),
        code(
            'os.makedirs("data/raw", exist_ok=True)',
            'path = "data/raw/orders_since_2026-08-01.jsonl"',
            "t = time.time()",
            'print(collect(session, path), f"{time.time() - t:.0f} s")',
        ),
        md(
            "You should see `126` pages, in between 60 and 120 s.",
            "",
            "126 pages at 60 a minute: partway through, the API answered `429` with a",
            "`Retry-After`, and the adapter waited it out. Nothing in `collect` mentions",
            "429, and nothing needed to.",
            "",
            "### Count what arrived",
        ),
        code(
            'rows = [order for line in open(path) for order in json.loads(line)["data"]]',
            'print(len(rows), len({order["id"] for order in rows}))',
        ),
        md(
            "You should see `62879 62879`.",
            "",
            "The shop database behind this API gives the same 62,879 for",
            "`WHERE created_at >= '2026-08-01'` (Lab 1.1, Part B): two routes to one source,",
            "and they agree. When they do not, one of your two pipelines is wrong — a",
            "*consistency* check.",
        ),
        md("## Part F — Resumable, proved from the other side (5 min)"),
        code(
            'before = session.get(f"{BASE}/v1/me/stats", timeout=10).json()',
            "print(collect(session, path))",
            'after = session.get(f"{BASE}/v1/me/stats", timeout=10).json()',
            'print(after["endpoints"]["orders"]["requests"]',
            '      - before["endpoints"]["orders"]["requests"])',
            'print(after["endpoints"]["orders"]["by_status"]["200"],',
            '      after["early_retries_after_429"])',
        ),
        md(
            "You should see:",
            "",
            "```text",
            "0",
            "0",
            "126 0",
            "```",
            "",
            "The second run made no request at all: resumability, confirmed by the server",
            "rather than by your own code. And `early_retries_after_429` is 0: each time",
            "the API said wait, your client waited. A client that retries before",
            "`Retry-After` is the one providers ban.",
            "",
            "**Your turn.** Delete the last line of the file and run `collect` once more.",
            "How many requests does it make, and why is the file correct afterwards?",
        ),
        code("# delete the last line of `path`, run collect again, and check the stats"),
        md(
            "## Did you validate this walkthrough?",
            "",
            "- [ ] `API_KEY` is read from Colab Secrets or `os.environ[...]`, and is in none of your code, notebooks or git history",
            "- [ ] Part B: 401, 400, 200, 404 — and you can say why none of them is worth a retry",
            "- [ ] Part C: offset gives 500 ids of which 473 are distinct; the cursor gives 500 distinct",
            "- [ ] Part D: the naive loop dies on 2026-06-04 with a 503; the session collects 23 July rates and skips 8 weekend days; your stats show 3 × 503 and 8 × 404",
            "- [ ] Part E: 126 pages and 62,879 distinct orders — the count the shop database gives since 2026-08-01",
            "- [ ] Part F: the second run makes 0 requests, and `early_retries_after_429` is 0",
        ),
    ]
    return notebook(cells)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(nb.REPO)}\n      {COLAB}")


if __name__ == "__main__":
    main()
