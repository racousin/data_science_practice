# Multi-Source Store Sales — Session 1

A retail chain runs five stores. Four of them recorded how many units of each
item they sold. The fifth, **Neighborhood_Market**, did not. Predict
`quantity_sold` for its 409 items.

No single source holds everything you need. The item data is in the stores'
own files, the unit cost comes from an API, the customer ratings are on a web
page, and the store descriptions are in the course database. The work of this
challenge is to collect the four sources, join them, and only then fit a model.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s1-store-sales.ipynb)

**The starter notebook.** One section per source, a baseline model, and the
cell that writes and submits `submission.csv`. The code that reads each source
is yours to write.

## The four sources

### 1. Files — the stores' exports

Download them from the **Data** tab of this page, or with the SDK:
`client.download_dataset(190, ".")`.

| file | format | items |
|---|---|---|
| `CityMart_data.csv` | CSV | 415 |
| `Greenfield_Grocers_data.csv` | CSV separated by `\|` | 401 |
| `SuperSaver_Outlet_data.xlsx` | Excel workbook, two sheets | 379 |
| `HighStreet_Bazaar_data.json` | JSON records | 396 |
| `Neighborhood_Market_data.csv` | CSV, **without** `quantity_sold` | 409 |

Each export was written by a different system, so no two files have the same
layout. Open each one and look at it before you read it with pandas: the
separator, the header row, the column names and the sheet structure all
differ. A few cells are missing.

Once read, the stores share these columns: `item_code`, `store_name`,
`mass`, `dimension_length`, `dimension_width`, `dimension_height`,
`days_since_last_purchase`, `package_volume` and `stock_age`, plus
`quantity_sold` for the four training stores. Some files also carry
`last_modified`. Each `item_code` belongs to exactly one store.

### 2. API — `unit_cost`

The documentation is at
[raphaelcousin.com/module4/api-doc](https://www.raphaelcousin.com/module4/api-doc),
under the exercise endpoints. Access takes two calls: the authentication
endpoint returns a password, and the prices endpoint takes that password in
its path and returns `unit_cost` per `item_code`. Request the password at run
time rather than copying it into your code.

### 3. Scraping — `customer_score` and `total_reviews`

The page
[raphaelcousin.com/module4/scrapable-data](https://www.raphaelcousin.com/module4/scrapable-data)
holds two tables; this challenge uses the second one, *Exercise Data*, whose
Customer Score and Total Reviews columns give `customer_score` and
`total_reviews` per item code. The page is rendered by JavaScript:
`requests.get` returns the HTML before the tables exist, so load it in a
browser driver such as Selenium, as in the Web Scraping lesson, and parse the
rendered source.

### 4. Database — the stores

The course PostgreSQL database has a schema `retail` with two tables for this
challenge:

| table | one row per | columns |
|---|---|---|
| `retail.stores` | store | `store_name`, `format`, `city`, `opened_year`, `weekly_footfall` |
| `retail.data_dictionary` | column of the final dataset | `column_name`, `description`, `source` |

Join `retail.stores` on `store_name`. Use `retail.data_dictionary` to check
what each of your columns means and which source it should come from.

The connection values are on the Session 1 Lab page. Store them as Colab
secrets or environment variables, never in the notebook:

```python
import os
import pandas as pd
from sqlalchemy import create_engine

# In Colab: pip install "psycopg[binary]", then
# from google.colab import userdata
# os.environ["DATABASE_URL"] = userdata.get("DATABASE_URL")
engine = create_engine(os.environ["DATABASE_URL"])
stores = pd.read_sql("SELECT * FROM retail.stores", engine)
```

## What you submit

`submission.csv`, one row per item of `Neighborhood_Market_data.csv`, in any
order:

```csv
item_code,quantity_sold
P0002,200.0
P0004,187.5
```

`quantity_sold` is a real number. Every `item_code` of
`Neighborhood_Market_data.csv` must appear exactly once. A missing column, a
missing item, an unknown item, a duplicate, a non-numeric value or a `NaN` is
rejected with a message naming the problem.

## Scoring

**−MAE**: the mean absolute error, negated so that higher is better. A score
of −12.5 means your predictions are off by 12.5 units per item on average; 0
would be perfect. RMSE is shown alongside.

**The bar is a score of −20 or higher**, that is an MAE of at most 20 units.

## Your first push

Submit something before you model anything. A constant prediction proves the
whole pipeline: the files are read, the submission has the right format, and
your key works. It will not clear the bar.

```python
import pandas as pd

test = pd.read_csv("Neighborhood_Market_data.csv")
train = pd.read_csv("CityMart_data.csv")   # one store is enough for now

submission = pd.DataFrame({
    "item_code": test["item_code"],
    "quantity_sold": train["quantity_sold"].mean(),
})
submission.to_csv("submission.csv", index=False)
```

Submit it in one of two ways:

- **On this page:** *Submit Prediction*, give the submission a name, upload
  `submission.csv`, then deploy it.
- **With the SDK** (`pip install mlarena-sdk`; the key is on your Profile page,
  under API Keys, and starts with `mlk_user_`):

```python
import os
import mlarena

client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])
client.submit(challenge_id=190, files=["submission.csv"])
print(client.leaderboard(190).head())
```

Once it is on the leaderboard, replace the constant with a model, add one
source at a time, and submit again.

## Bonus — see your score in the database

The course database scores predictions too. Write yours into the shared
`playground` schema with the writer account (`WRITER_URL` and `STUDENT` are
on the Session 1 Lab page):

```python
writer = create_engine(os.environ["WRITER_URL"])
student = os.environ["STUDENT"]   # a-z, 0-9 and _, starting with a letter

submission[["item_code", "quantity_sold"]].to_sql(
    f"{student}_predictions", writer, schema="playground",
    if_exists="replace", index=False)
```

About 10 to 20 seconds later, your row appears in `retail.leaderboard`:

```python
pd.read_sql("SELECT * FROM retail.leaderboard ORDER BY mae", engine)
```

`n_scored` is the number of items matched, and `message` explains any problem,
such as a missing column or a non-numeric value. Write again with
`if_exists="replace"` to update your row; drop the table to remove it.

This board scores **only half of the items** (those with an even number), so
its MAE differs from your score here, which uses all 409. Only a submission to
this challenge counts.

Submissions are scored deterministically: the same file always gets the same
score. The data was generated by a public notebook, so the answers can be
reproduced rather than predicted; that is not what is assessed here.
