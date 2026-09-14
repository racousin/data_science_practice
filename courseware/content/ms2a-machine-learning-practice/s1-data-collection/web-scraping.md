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

```python
from urllib.robotparser import RobotFileParser

rp = RobotFileParser("https://example.com/robots.txt")
rp.read()
rp.can_fetch("course-bot", "https://example.com/products/42")  # True
rp.crawl_delay("course-bot")                                    # 5
```

Check every URL before fetching it, and use the site's `Crawl-delay` when it
is longer than your own.

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
headers = {
    "User-Agent": "ENSAE course project (raphael.cousin@example.com)",
}
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

## XPath, the other selector language

```python
from lxml import html

tree = html.fromstring(r.text)
prices = tree.xpath("//span[@class='price']/text()")
```

CSS selectors cover most pages. XPath also selects by text content and walks
*up* the tree — "the row that contains the cell reading *Stock*" — in a single
expression, and it is the selector language of `lxml` and Selenium alike.

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

## Cleaning what you extract

```python
def to_float(text):
    # "\u00a0" is the non-breaking space, the one you cannot see
    cleaned = (text.replace("\u00a0", "")
                   .replace(" ", "")
                   .replace("€", "")
                   .replace(",", "."))
    return float(cleaned)      # to_float("1\u00a0234,50 €") -> 1234.5
```

HTML is presentation. `"1 234,50 €"` is a string containing a non-breaking space,
a comma decimal separator and a currency symbol. The naive cleaner — strip, drop
the symbol, swap the comma — raises `ValueError` on exactly that string, because
the space is still sitting in the middle of the number. That is why the test
comes first.

Write the cleaner as a named function and unit-test it on the ugly cases you have
actually seen. This is where a unit test does real work.

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
3. **Reconsider.** If it needs a headless browser and a login, ask whether the
   data is worth it.

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

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   def to_float(text):
       cleaned = (text.replace("\u00a0", "")
                      .replace(" ", "")
                      .replace("\u20ac", "")
                      .replace(",", "."))
       return float(cleaned)

   print(to_float("1\u00a0234,50\u00a0\u20ac"))    # -> 1234.5
   print(to_float("1 234,50 \u20ac"))              # -> 1234.5
   ```

   **Answer.** Both forms parse. `\u00a0` is the non-breaking space the browser
   shows as an ordinary one — the reason the naive cleaner raises `ValueError`
   on a string that looks perfectly clean on screen.

2. `soup.select_one("h1.product-title")` returns `None` after a site redesign.
   What should your code do, and what happens if it does not?

   **Answer.** Raise, naming the URL. A scraper that does not check writes 50,000
   empty rows and you discover it weeks later. A crashed scraper is a scraper
   you fix; a silent one is a dataset you throw away.

3. The page you want is built in the browser by JavaScript, so `requests` gets
   you an empty shell. What is the first option to try, and why is it the good
   outcome?

   **Answer.** Open the network tab and find the underlying API — the page is
   usually fetching JSON from an endpoint you can call directly. That gives you a
   contract instead of a selector, and it is 50× faster than driving a headless
   browser.
