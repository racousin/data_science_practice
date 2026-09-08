# Bike Sharing Demand — Session 2

Predict how many bikes are rented in a given hour, from the calendar and the
weather. 17,379 hourly observations from a bike-share system, twelve columns,
one continuous target.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s2-bike-demand.ipynb)

**The worked notebook.** It runs end to end in Colab — downloads this dataset,
explores it, fits the model, writes `submission.csv` and submits it. Open it,
paste your API key in the second cell, and run all.

## The data

| file | rows | contents |
|---|---|---|
| `X_train.csv` | 13,903 | `id` + the 12 features |
| `y_train.csv` | 13,903 | `id,prediction` — the hourly rental count |
| `X_test.csv` | 3,476 | `id` + the same 12 features |

A random 80/20 split of a public bike-sharing dataset. There are no missing
values.

| column | meaning |
|---|---|
| `season` | `spring`, `summer`, `fall`, `winter` |
| `year` | 0 for the first year, 1 for the second |
| `month`, `hour`, `weekday` | 1–12, 0–23, 0–6 |
| `holiday`, `workingday` | `True` / `False` |
| `weather` | `clear`, `misty`, `rain`, `heavy_rain` |
| `temp`, `feel_temp` | °C |
| `humidity` | 0–1 |
| `windspeed` | normalised |

## What you submit

`submission.csv` — one row per test id, in any order:

```csv
id,prediction
te_75c7a09e8d17,143.2
te_a10c9b861188,88.0
```

`prediction` is a real number. Every id in `X_test.csv` must appear exactly
once; a missing id, an unknown id, a duplicate, a non-numeric value or a `NaN`
is rejected with a message naming the line.

Negative predictions are **not** rejected. A plain linear model returns them on
the low-traffic hours; the leaderboard shows how many you sent.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
print(client.leaderboard(<id>).head())
```

## Scoring

**−MAE** — the mean absolute error, negated so that higher is better. Read it
straight off: **−103.74 means your predictions are wrong by 103.74 bikes an
hour on average**, and 0 would be perfect. RMSE and R² are shown alongside.

Reference points on this split:

| model | −MAE |
|---|---|
| predict the training mean | −140.08 |
| `get_dummies` + `LinearRegression` — the notebook's | **−103.74** |
| the same, with `hour` one-hot encoded | −74.13 |

The third row is the second model with one change: `hour` stops being a number
— which asserts that 23:00 is twenty-three times 1:00, and that midnight is a
fall of 23 — and becomes twenty-four unordered categories instead.

Submissions are scored deterministically; the same file always gets the same
number. The source data is public, so the held-out counts can be looked up
rather than predicted — that is not modelling and not what is being assessed.
