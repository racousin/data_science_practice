# APIs

An API is a contract for getting data over HTTP. The contract covers the shape of
the response; it does not cover the network, the rate limit, or the day the
provider changes a field name.

<!-- notes: 30 minutes. Pagination and retries are the parts students skip and
then rediscover at 3 am with a half-written file. Do the retry code live. -->

---

## The four things to find in the documentation

1. **The endpoint** — the URL, and what one call returns
2. **The authentication** — key in a header, OAuth token, nothing
3. **The pagination** — how you get past the first 100 records
4. **The rate limit** — requests per second, per hour, per key

Everything else is detail. If the documentation does not state the rate limit,
assume it is low and find out politely.

---

## A request

```python
import requests

r = requests.get(
    "https://api.example.com/v1/orders",
    headers={"Authorization": f"Bearer {os.environ['API_KEY']}"},
    params={"since": "2025-01-01", "limit": 100},
    timeout=10,
)
r.raise_for_status()
data = r.json()
```

`timeout` is not optional. Without it a hung connection blocks forever, and your
overnight collection job produces nothing.

---

## Status codes worth handling differently

| Code | Meaning | Response |
|---|---|---|
| 200 | fine | carry on |
| 401 / 403 | bad or missing credentials | stop — retrying will not help |
| 404 | no such resource | stop, or skip this record |
| 429 | rate limited | wait, then retry |
| 5xx | their problem | retry with backoff |

`raise_for_status()` collapses all of these into one exception. That is the right
default; the moment you need to treat 429 differently, catch it explicitly.

---

## Pagination

Nobody returns a million records in one response. Three common styles:

```python
# offset / limit
params = {"offset": 0, "limit": 100}
```

```python
# page number
params = {"page": 1, "per_page": 100}
```

```python
# cursor — the only one that is safe on changing data
params = {"cursor": None}
```

Offset pagination on a table that is being written to will skip and duplicate
records. If a cursor is offered, use it.

---

## The loop

```python
rows, cursor = [], None
while True:
    r = session.get(url, params={"cursor": cursor, "limit": 100}, timeout=10)
    r.raise_for_status()
    body = r.json()
    rows.extend(body["data"])
    cursor = body.get("next_cursor")
    if not cursor:
        break
```

Two things to add before you run it for real: a cap on the number of pages, and
writing to disk as you go. A loop that holds 400,000 records in a list and then
crashes has collected nothing.

---

## Retry with backoff

```python
from requests.adapters import HTTPAdapter, Retry

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
)))
```

`backoff_factor=1` waits 0 s, 2 s, 4 s, 8 s, 16 s: urllib3 retries the first
failure at once, then doubles. When a 429 or 503 carries `Retry-After`, that
wait replaces the backoff. Exponential backoff is what separates a client the
provider tolerates from one they block.

Note what is **not** in `status_forcelist`: 401 and 404. Retrying those is just
noise.

---

## Respect the rate limit

```python
limit = int(r.headers.get("X-RateLimit-Remaining", 1))
reset = int(r.headers.get("X-RateLimit-Reset", 0))
if limit == 0:
    time.sleep(max(reset - time.time(), 0) + 1)
```

Most APIs tell you how much budget is left in the response headers. Reading them
is cheaper than being banned.

A `Session` also reuses the TCP connection, which is a free speedup across
thousands of calls.

---

## Write raw, parse later

```python
with open("raw/orders.jsonl", "a") as f:
    for row in body["data"]:
        f.write(json.dumps(row) + "\n")
```

Append the untouched JSON to disk first; parse into a dataframe in a separate
step.

Then a parsing bug costs a re-run of the parser, not a re-run of six hours of
API calls — and you keep the evidence of what the API actually returned on the
day you called it.

---

## Make it resumable

```python
done = {json.loads(l)["id"] for l in open("raw/orders.jsonl")}
```

Any collection job that runs longer than a few minutes will be interrupted.
Design for it: record what you have, skip it on restart, and the job becomes
restartable instead of restart-from-zero.

---

## Keys are secrets

```python
API_KEY = os.environ["API_KEY"]        # crash if absent
```

Not in the source, not in the notebook, not in the commit. A key pushed to a
public repository is scraped within minutes — this is automated, and it happens
to students every year.

---

## Checklist

- `timeout` on every call
- backoff on 429 and 5xx, no retry on 4xx
- cursor pagination where offered, with a page cap
- append raw JSONL, parse in a separate step
- resumable by design
- key from the environment

---

## Check yourself

1. Which status codes belong in `status_forcelist`, and which ones must never be
   retried at all?

   **Answer.** Retry 429 and the 5xx codes with exponential backoff. Never retry
   401/403 (bad credentials — retrying will not help) or 404 (no such resource);
   both are just noise on the provider's server.

2. Run this. You should get exactly the output shown.

   ```python
   import json, os

   os.makedirs("raw", exist_ok=True)
   with open("raw/orders.jsonl", "w") as f:
       for i in (1, 2, 3):
           f.write(json.dumps({"id": i, "amount": i * 10}) + "\n")

   done = {json.loads(l)["id"] for l in open("raw/orders.jsonl")}
   print(sorted(done))                                    # -> [1, 2, 3]
   print([i for i in (1, 2, 3, 4, 5) if i not in done])    # -> [4, 5]
   ```

   **Answer.** That is the whole of "resumable": the ids already on disk are
   skipped, so an interrupted job restarts from record 4 rather than from zero.

3. Why append the untouched JSON to disk before parsing anything into a
   dataframe?

   **Answer.** A parsing bug then costs a re-run of the parser, not a re-run of
   six hours of API calls — and you keep the evidence of what the API actually
   returned on the day you called it.
