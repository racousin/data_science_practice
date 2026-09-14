# APIs

An API is a contract for getting data over HTTP. The contract covers the shape of
the response; it does not cover the network, the rate limit, or the day the
provider changes a field name.

<!-- notes: 30 minutes. Pagination and retries are the parts students skip and
then rediscover at 3 am with a half-written file. Do the retry code live. -->

---

## What a call is made of

An **endpoint** is a URL that names a resource, such as `/v1/orders`. A call
sends a request to it and gets a response back.

| Request | Response |
|---|---|
| **method** — what to do with the resource | **status code** — what happened |
| **URL** — the endpoint, plus query parameters | **headers** — metadata, rate-limit budget |
| **headers** — credentials, content type | **body** — the data, usually JSON |
| **body** — data sent with POST or PUT | **error** — a message saying what failed |

---

## HTTP methods

| Method | Does | In data collection |
|---|---|---|
| GET | reads a resource | almost every call you make |
| POST | creates a resource, or runs a query | searches too large for a URL |
| PUT | replaces or updates a resource | rarely |
| DELETE | removes a resource | never, when you only collect |

```python
r = requests.post("https://api.example.com/v1/search", timeout=10,
                  json={"city": "Lyon", "since": "2025-01-01"})
```

`json=` serialises the body and sets `Content-Type: application/json`; `data=`
would send a form instead. A GET carries its question in `params=`.

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

## Authentication

| Method | How it works | Typical use |
|---|---|---|
| API key | a fixed secret sent with every request | public data APIs, simple services |
| OAuth 2.0 | exchange credentials for a short-lived access token | delegated or fine-grained access |
| JWT | a signed token carrying claims such as identity and expiry | stateless modern web APIs |

Send a key in a header — `Authorization: Bearer ...` or `X-API-Key: ...` —
rather than as `?api_key=` in the URL: URLs end up in server logs, proxy logs
and browser history.

A JWT is signed, not encrypted. Anyone holding one can read its payload, and
use it until it expires.

---

## Getting a token first

```python
r = session.post(TOKEN_URL, timeout=10, data={
    "grant_type": "client_credentials",
    "client_id": os.environ["CLIENT_ID"],
    "client_secret": os.environ["CLIENT_SECRET"],
})
r.raise_for_status()
token = r.json()["access_token"]
session.headers["Authorization"] = f"Bearer {token}"
```

This is the OAuth 2.0 client-credentials flow: one call to an authentication
endpoint, then the token on every data call. The response also states
`expires_in`; a collection job that outlives the token gets a 401 and has to
request a new one.

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
    r = session.get(url, timeout=10,
                    params={"cursor": cursor, "limit": 100})
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

## From payload to dataframe

Check the shape of the body before converting it. Two shapes are common:

```python
# a list of records: [{"id": "P1", "volume": 12.5}, ...]
df = pd.json_normalize(body["data"])
```

```python
# a mapping keyed by id: {"P1": 12.5, "P2": 3.0}
df = pd.DataFrame.from_dict(body["data"], orient="index",
                            columns=["volume"])
```

In the second shape the id becomes the index, not a column. Name it —
`df.index.name = "id"` — before you join on it.
