# Web Scraping

Scraping is what you do when there is no API. It has no contract, no versioning,
and real legal boundaries — so it is the last option, and it comes with
obligations.

<!-- notes: 25 minutes. Lead with the legal/ethical part, not as a disclaimer but
because it constrains the design. The selectors-break point matters: scrapers rot.
-->

---

## Before anything else

```text
https://example.com/robots.txt
```

`robots.txt` states which paths the site asks automated clients to leave alone.
It is a request, not a technical barrier — which is exactly why ignoring it is a
choice you own.

Read the Terms of Service too. "Automated access is prohibited" is a common
clause and it is binding on you.

---

## The rules that are not optional

- **Personal data is regulated.** Under the GDPR, scraping names, emails or
  profiles is processing personal data. You need a legal basis. For a course
  project, do not.
- **Content is copyrighted.** Being able to read a page does not grant a licence
  to redistribute it.
- **Load is a real cost.** Hammering a small site's server is a denial of service
  even when unintentional.
- **Identify yourself.** A `User-Agent` that says who you are and why.

---

## A polite client

```python
headers = {"User-Agent": "ENSAE course project (raphael.cousin@example.com)"}
r = requests.get(url, headers=headers, timeout=10)
time.sleep(2)
```

One request every second or two, a real contact address, and a timeout. This is
the whole of "polite" and it costs you nothing but wall-clock time.

---

## Parsing

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(r.text, "html.parser")
title = soup.select_one("h1.product-title").text.strip()
prices = [e.text for e in soup.select("span.price")]
```

`select` / `select_one` take CSS selectors, which are the same ones you read off
the browser's inspector. `lxml` is a faster parser if volume grows.

---

## Selectors break

```python
el = soup.select_one("h1.product-title")
if el is None:
    raise ValueError(f"title selector failed on {url}")
```

A site redesign changes `span.price` to `span.price-tag` and your scraper starts
producing `None` for every row. Without the check it writes 50,000 empty rows and
you find out in Session 2.

Fail on the missing selector. A crashed scraper is a scraper you fix; a silent
one is a dataset you have to throw away.

---

## Cleaning what you extract

```python
def to_float(text):
    return float(text.strip().replace("€", "").replace(",", "."))
```

HTML is presentation. `"1 234,50 €"` is a string containing a non-breaking space,
a comma decimal separator and a currency symbol.

Write the cleaner as a named function and unit-test it on the ugly cases you have
actually seen. That is one of the tests from Session 1 of the 12h module doing
real work.

---

## Rendered pages

Many sites build their content in the browser with JavaScript. `requests` gets
you the empty shell.

Options, in order of preference:

1. **Find the underlying API.** Open the network tab — the page is usually
   fetching JSON from an endpoint you can call directly. This is the good outcome.
2. **Playwright / Selenium.** A real browser, 50× slower and far heavier.
3. **Reconsider.** If it needs a headless browser and a login, ask whether the
   data is worth it.

---

## Structure of a scraper that survives

```text
scrape/
  fetch.py     <- URL -> raw HTML on disk, rate limited, resumable
  parse.py     <- raw HTML -> records
  clean.py     <- records -> typed dataframe
```

Save the HTML. Storage is free; re-fetching is not, and it is the part that
annoys the site owner.

With the raw pages on disk, a parser bug is a thirty-second re-run.

---

## When to stop

Scraping is justified when:

- there is no API and no download
- the terms permit it
- the data is not personal
- the volume is modest and the rate is polite

If any of those fails, the answer is a different data source — not a cleverer
scraper.

---

## Checklist

- `robots.txt` and the ToS read, before code
- no personal data
- identifying `User-Agent`, contact address, 1–2s delay
- raw HTML written to disk, fetch and parse separated
- every selector asserted, crash on failure
- cleaners are named functions with tests
