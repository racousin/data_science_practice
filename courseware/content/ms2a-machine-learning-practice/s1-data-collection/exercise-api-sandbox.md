# Lab 1.2 — A Client That Survives the API

An API is a contract with conditions attached: it identifies you, returns data
a page at a time, limits how fast you may ask, and fails in ways the client has
to absorb. A client that works once against a quiet server is not yet a
collection pipeline. It needs a timeout on every call, retries with backoff that
respect `Retry-After`, cursor pagination that stays correct while data arrives,
and raw writes it can resume after a crash.

This lab runs against a practice API that misbehaves on a schedule. You build
that client one piece at a time, watch each piece earn its place, then ask the
API what your client actually did.

**Time:** 40 minutes. **Deliverable:** `data/raw/orders_since_2026-08-01.jsonl`,
and the numbers in the checklist at the end.

<!-- notes: The retry code, run by everyone at once. The failures are on a
schedule, so the room meets the same ones in the same places. Everything is kept
per X-Student name: /v1/me/stats is what to look at when a student says their
client handles 429s (early_retries_after_429 must be 0), and GET /v1/admin/stats
with the teacher's key shows everyone's. The numbers assume each block is run
once under a fresh name; a re-run adds to that name's history, which is what the
endpoint is for, not a bug. The walkthrough itself is in the notebook
(tools/notebooks/mlp_s1_lab1_2_api_sandbox.py); the API key below rotates after
the session, so refresh this page from LAB.local.md (make info in
sql_api_sandbox) before each one. -->

---

## The concept — status codes and pagination

**Status codes say whose problem it is.** 401: who you are. 400: what you sent.
404: what you asked for. None of them is fixed by sending the same request
again. 429 and 5xx are, after a wait.

**Offset or cursor.** An offset counts from the newest record, and on a busy
feed the newest keeps changing: each page starts a few records early and
repeats the end of the previous one. A cursor says *after this record*, which
no new arrival can move. Where records are also deleted, offset skips rows — and
a skipped row, unlike a duplicate, is never seen.

---

## The concept — timeouts, retries, rate limits

**A call without a timeout can wait forever.** A hung connection in production
does not answer, and the overnight job produces nothing.

**Retry what is transient, with backoff.** A `requests.Session` with an
`HTTPAdapter(max_retries=Retry(...))` retries 429 and 5xx, waits the
`Retry-After` the server asks for, and counts a timeout against the same
budget. A client that retries before `Retry-After` is the one providers ban.

**The server does the work whether or not you wait.** A request your client gave
up on was still served.

---

## The concept — resumable collection

Write each page raw, as it arrives, and resume from the last page's
`next_cursor` when the file already exists. A crash then costs one page, not
the run, and a second run of a finished collection makes no request at all.

Check the result against a second route to the same source: the API's orders
since 1 August and the shop database's `WHERE created_at >= '2026-08-01'`
(Lab 1.1) must agree. When they do not, one of the two pipelines is wrong.

---

## Access — today's sandbox

| variable | value |
|---|---|
| `SANDBOX_URL` | `https://sandbox.ml-arena.com` |
| `API_KEY` | `__API_KEY__` |
| `STUDENT` | your name: `a-z`, `0-9` and `_`, starting with a letter |

- Every `/v1` call sends two headers: `Authorization: Bearer <API_KEY>` and
  `X-Student: <STUDENT>`. The name keeps your rate limit, your failures and
  your statistics apart from the class's — pick one and keep it.
- The key is the class's and **rotates after the session**.
- Put `SANDBOX_URL` and `API_KEY` in Colab's *Secrets* panel (or a gitignored
  `.env`), never in a cell. The notebook reads them from there.

---

## Open the notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s1-lab1-2-api-sandbox.ipynb)

Six parts, run once each, in order, under one name:

| part | what you do | min |
|---|---|---|
| A | read the documentation page before calling anything | 3 |
| B | provoke 401, 400, 404 and a 200 | 4 |
| C | paginate a busy feed by offset, then by cursor | 7 |
| D | fetch daily FX rates without, then with, timeout and retries | 10 |
| E | collect every order since 1 August through the rate limit | 8 |
| F | prove from the server's statistics that the collection resumes | 5 |

---

## Did you validate this lab?

- [ ] `API_KEY` is read from Colab Secrets or `os.environ[...]`, and is in none of your code, notebooks or git history
- [ ] Part B: 401, 400, 200, 404 — and you can say why none of them is worth a retry
- [ ] Part C: offset gives 500 ids of which 473 are distinct; the cursor gives 500 distinct
- [ ] Part D: the naive loop dies on 2026-06-04 with a 503; the session collects 23 July rates and skips 8 weekend days; your stats show 3 × 503 and 8 × 404
- [ ] Part E: 126 pages and 62,879 distinct orders — the count the shop database gives since 2026-08-01
- [ ] Part F: the second run makes 0 requests, and `early_retries_after_429` is 0
