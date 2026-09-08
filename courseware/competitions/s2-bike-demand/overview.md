# Bike Sharing Demand — Session 2

Predict how many bikes are rented in a given hour, from the calendar and the
weather. 17,379 hourly observations from a bike-share system, twelve columns,
one continuous target.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s2-bike-demand.ipynb)

**The worked notebook.** It runs end to end in Colab — downloads this dataset,
explores it, fills the holes, fits the model, writes `submission.csv` and
submits it. Open it, paste your API key in the second cell, and run all.

## The data

| file | rows | contents |
|---|---|---|
| `X_train.csv` | 13,903 | `id` + the 12 features, **in calendar order** |
| `y_train.csv` | 13,903 | `id,prediction` — the hourly rental count |
| `X_test.csv` | 3,476 | `id` + the same 12 features, shuffled |

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

## The split is by time, not at random

The first 80% of the hours are your training set; the **last 20%** are held
back. The test window is the second year from mid-August to 31 December — a
period the model has never seen, in a system that grew substantially between the
two years.

This is not a detail. Until 2026-09-08 the split was random, which put 20:00 in
the test set while 19:00 and 21:00 stayed in training: the same weather reading,
a count within a few bikes. A model could look up most of what it was being
asked to predict, and every score on this page was optimistic by roughly **35
MAE**. Interpolating between known hours is not forecasting, and forecasting is
the problem a bike-share operator actually has.

Two consequences you will meet:

- Your own validation split should be by time too — the **last** rows of
  `X_train.csv`, not random ones. That is why the training file ships in
  calendar order.
- The test window contains no summer hours and no `heavy_rain`. Encode with
  `pd.get_dummies` and `reindex(columns=..., fill_value=0)`, or your two
  matrices will have different widths.

## Some cells are missing

Deliberately. `df.isna().sum()` on the training file:

| column | missing | pattern |
|---|---|---|
| `windspeed` | 695 | scattered — dropped readings |
| `temp`, `feel_temp` | 371 | the **same** rows, in runs of ~60 consecutive hours — a thermometer down |
| `weather` | 278 | scattered, and the one non-numeric column |

`sklearn` will not fit through a `NaN`. Decide what to put there before you
model, and fill with a statistic learned on the **training** rows only — using
the test set's own median is leakage, and it flatters your validation score
without moving your leaderboard score.

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
straight off: **−138.88 means your predictions are wrong by 138.88 bikes an
hour on average**, and 0 would be perfect. RMSE is shown alongside.

Reference points on this split:

| model | −MAE |
|---|---|
| predict the training mean | −174.98 |
| impute + `get_dummies` + `LinearRegression` — the notebook's | **−138.88** |
| the same, with `hour` one-hot encoded | −100.28 |

Predicting the training mean is unusually bad here, and that is the split
talking: the second half of the second year is busier than the average of
everything before it, so a constant learned from the past under-shoots the
future by design.

The third row is the second model with one change: `hour` stops being a number
— which asserts that 23:00 is twenty-three times 1:00, and that midnight is a
fall of 23 — and becomes twenty-four unordered categories instead. One line, and
a larger gain than any model swap in Session 3 will buy you.

Submissions are scored deterministically; the same file always gets the same
number. The source data is public, so the held-out counts can be looked up
rather than predicted — that is not modelling and not what is being assessed.
