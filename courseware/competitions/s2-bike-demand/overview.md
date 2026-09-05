# Bike Sharing Demand — Session 2

Predict how many bikes are rented in a given hour, from the calendar and the
weather. 17,379 hourly observations from a bike-share system, twelve columns,
one continuous target.

This is the **regression** challenge of Session 2 and the first one of the
course. It asks for exactly what the session taught: read a table, look at it
properly, fit a linear model, and turn its predictions into a file.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s2-bike-demand.ipynb)

**The worked notebook.** It runs end to end in Colab — downloads this dataset,
explores it, fits the model, writes `submission.csv` and submits it — and scores
the baseline quoted below. Open it, paste your API key in the second cell, and
run all.

## The data

| file | rows | contents |
|---|---|---|
| `X_train.csv` | 13,903 | `id` + the 12 features |
| `y_train.csv` | 13,903 | `id,prediction` — the hourly rental count |
| `X_test.csv` | 3,476 | `id` + the same 12 features |

A random 80/20 split of openml `Bike_Sharing_Demand` (version 2) at
`random_state=42`. Ids are freshly assigned and the source ordering is
discarded. There are no missing values.

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
te_00000,143.2
te_00001,88.0
```

`prediction` is a real number. Every id in `X_test.csv` must appear exactly
once. A missing id, an unknown id, a duplicate, a non-numeric value or a `NaN`
is rejected with a message naming the line, and scores 0 — a scorer that
silently imputed your missing rows would be lying to you.

Negative predictions are **not** rejected. A plain linear model returns them on
the low-traffic hours, and the score is what tells you it happened; the
leaderboard shows how many you sent.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
print(client.leaderboard(<id>).head())
```

## Scoring

**R², on the held-out hours.** Ranking is on R² rather than RMSE because the
leaderboard sorts descending and needs a higher-is-better number. The two are
monotone in each other, so nothing is lost, and R² has a reading RMSE does not:

> **R² = 0 is exactly the model that predicts the training mean for every hour.**

So the sign of your score answers "did I beat the average" with no further
arithmetic. RMSE and MAE are on the leaderboard as well, in bikes per hour.

The worked notebook's model — `pd.get_dummies` then `LinearRegression`, no
tuning — scores **R² = 0.3993** (RMSE 137.92). That is the bar. Here is the
ladder above it on this exact split:

| model | R² | RMSE |
|---|---|---|
| predict the training mean | 0.000 | 178.03 |
| linear regression | 0.399 | 137.92 |
| the same, clipped at 0 | 0.405 | 137.27 |
| linear regression, `hour` one-hot encoded | 0.698 | 97.75 |
| gradient boosting | 0.949 | 40.03 |

Read the third and fourth rows together, because they are the lesson. Both
models are `LinearRegression`. The only difference is that the second stops
treating `hour` as a *number* — which asserts that 23:00 is twenty-three times
1:00, and that the step from 23:00 to 00:00 is a fall of 23 — and treats it as
twenty-four unordered categories instead. That one encoding decision is worth
0.30 of R², far more than any change of model family would give you at this
stage. The data preparation lesson said the choice was a modelling assumption;
this is the size of it.

Submissions are scored once and deterministically — the same file always gets
the same number.
