# Web Scraping

Scraping is what you do when there is no API. It has no contract, no versioning,
and real legal boundaries — so it is the last option, and it comes with
obligations.

<!-- notes: 25 minutes. Lead with the legal/ethical part, not as a disclaimer but
because it constrains the design. The selectors-break point matters: scrapers rot.
-->

---

## What a scraper is made of

| Part | Role | Python |
|---|---|---|
| HTTP client | fetches the page | `requests` |
| HTML parser | turns the markup into a tree | `BeautifulSoup`, `lxml` |
| Selectors | point at the elements you want | CSS selectors or XPath |

Typical uses: monitoring prices, collecting research data that is published
only as web pages, aggregating content from several sites.

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

## Reading robots.txt in code

```text
User-agent: *
Disallow: /checkout/
Crawl-delay: 5
```
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


## Scraping a table

```python
rows = [[td.text.strip() for td in tr.select("td")]
        for tr in soup.select("table tbody tr")]
df = pd.DataFrame(rows, columns=["product_id", "rating", "n_reviews"])
```

Every cell comes out as a string, so the types are yours to set, and the column
order is whatever the page shows today — assert the header row before
trusting the positions.

For a well-formed `<table>`, `pd.read_html(io.StringIO(r.text))` returns one
dataframe per table on the page.

---

## Selectors break

```python
el = soup.select_one("h1.product-title")
if el is None:
    raise ValueError(f"title selector failed on {url}")
```

A site redesign changes `span.price` to `span.price-tag` and your scraper starts
producing `None` for every row. Without the check it writes 50,000 empty rows and
you find out weeks later, when a model trains on them.

Fail on the missing selector. A crashed scraper is a scraper you fix; a silent
one is a dataset you have to throw away.

---


## A page shows only the present

```python
record = {
    "url": url,
    "price": to_float(price_el.text),
    "scraped_at": pd.Timestamp.now(tz="UTC").isoformat(),
}
```

A product page displays today's price and nothing else. Scraping it every hour
builds a price history that exists nowhere else — but only if every record
carries the time it was fetched. Without `scraped_at`, two runs are
indistinguishable.

Schedule the run with cron or a workflow scheduler rather than a
`while True: sleep(3600)` loop: a crashed loop stops collecting silently.

---

## Rendered pages

Many sites build their content in the browser with JavaScript. `requests` gets
you the empty shell.

Options, in order of preference:

1. **Find the underlying API.** Open the network tab — the page is usually
   fetching JSON from an endpoint you can call directly. This is the good outcome.
2. **Playwright / Selenium.** A real browser, 50× slower and far heavier.

---

## Driving a browser

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

driver = webdriver.Chrome()
driver.get(url)
WebDriverWait(driver, 10).until(EC.presence_of_element_located(
    (By.CSS_SELECTOR, "table tbody tr")))
page = driver.page_source          # the DOM after JavaScript ran
driver.quit()
```

Wait for the element you need, not for a fixed `time.sleep(5)`: a fixed sleep
is too long on a fast day and too short on a slow one. From `page_source` on,
parsing is the same BeautifulSoup code as before.
