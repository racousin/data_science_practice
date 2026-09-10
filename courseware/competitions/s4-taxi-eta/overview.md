# Taxi Arrival Promise — Session 4

Promise an arrival time the ride beats 9 times out of 10. For each of 10,461
New York green-taxi trips, predict a duration in minutes; a minute late costs
nine times as much as a minute of padding. Tabular, twelve numeric columns, one
continuous target, and a loss you write yourself.

This is **Lab 1** of Session 4. The lab has you write a PyTorch training loop,
from the model to the saved checkpoint; this challenge scores the promises it
makes, on six days of trips it has never seen.

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s4-taxi-eta.ipynb)

**The starter notebook.** It downloads this dataset, splits it, standardises
it and scores it. The model, the loss, the optimizer and the training loop are
yours, one step at a time, and each step ends on a check cell that tells you
whether it works. Paste your API key into `API_KEY` and run it step by step. A
solution notebook is linked here after the session.

## The data

Public trip records from the NYC Taxi & Limousine Commission (TLC),
[published on its website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):
green taxis, January 2024. Kept: trips picked up in January, 1 to 120 minutes
long, 0.1 to 50 miles, starting and ending in Manhattan, Queens, Brooklyn or
the Bronx — 52,305 trips.

| file | rows | contents |
|---|---|---|
| `X.csv` | 41,844 | `id` + the 12 features, **in pickup order** |
| `y.csv` | 41,844 | `id,prediction` — the trip's duration in minutes |
| `X_submission.csv` | 10,461 | `id` + the same 12 features, shuffled |

| column | meaning |
|---|---|
| `trip_distance` | miles |
| `pickup_hour` | time of day in hours: 13.5 is 13:30 |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `is_weekend` | 1 on Saturday and Sunday, else 0 |
| `pickup_manhattan`, `_queens`, `_brooklyn`, `_bronx` | 1 for the borough the trip starts in |
| `dropoff_manhattan`, `_queens`, `_brooklyn`, `_bronx` | 1 for the borough it ends in |

`trip_distance` is the metered distance, known only once the ride is over. A
real app would use the length of the planned route; this data stands in for it
with the metered one.

## The split is by time

`X.csv` holds the first 25 days (1 January to 25 January, 20:56);
`X_submission.csv` holds the rest of the month. The leaderboard therefore
scores a forecast: trips from days the model has never seen.

Validate the same way — hold out the **last** 20% of `X.csv`, which is why it
ships in pickup order. A random split puts trips from the same hour on both
sides and tells you less about the days ahead. Expect the two scores to
differ: the lab's network scores about −0.98 on its validation days and −0.94
on the leaderboard's. The days are not alike — the 90th percentile of the
duration is 24.3 minutes over the first 80% of `X.csv`, 25.8 over its last 20%
and 25.3 over the leaderboard's days — so read a validation score as a
comparison between your own variants, not as a forecast of your rank.

## The objective

For one trip that takes y minutes when you promised ŷ:

```text
loss = max(0.9 × (y − ŷ), −0.1 × (y − ŷ))
```

- Late by 2 minutes (y = 12, ŷ = 10): 0.9 × 2 = **1.8**.
- Early by 3 minutes (y = 7, ŷ = 10): 0.1 × 3 = **0.3**.

A late minute costs 0.9 and a minute of padding 0.1, so the promise that
minimises the average loss is the **90th percentile** of the trip's duration,
not its mean. This is the pinball (or quantile) loss at τ = 0.9; at τ = 0.5 it
is half the absolute error, whose best constant is the median. In PyTorch, with
`pred` and `y` of the same shape:

```python
d = y - pred
loss = torch.maximum(0.9 * d, -0.1 * d).mean()
```

## What you submit

`submission.csv` — one row per id in `X_submission.csv`, in any order:

```csv
id,prediction
te_75c7a09e8d17,18.4
te_a10c9b861188,9.25
```

`prediction` is your promised duration in minutes, a finite real number. The
header needs an `id` and a `prediction` column; other columns are ignored. A
missing column, a header with no rows, a missing, unknown, duplicate or empty
id, a non-numeric value, a `NaN` or an infinity is rejected with a message
naming the line. Negative predictions are not rejected — a duration cannot be
negative, but a network can predict one — and the leaderboard shows how many
you sent.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
```

## Two submissions a day

Each ML-Arena account may submit **2 times per rolling 24 hours, across all
challenges**. A file with the wrong header or the wrong set of ids is refused at
upload and costs nothing; a file the scorer rejects (a `NaN`, a non-number)
still uses a slot. Compare your variants — another learning rate, a wider
network, MSE against pinball — on your validation split with the notebook's
scorer, and submit only the one you keep.

## Scoring

**−Pinball**: the mean pinball loss over the 10,461 trips, negated so that
higher is better. 0 would be perfect; −1.04 means the promises cost 1.04 per
trip on average, counting 0.9 per minute late and 0.1 per minute early. Three
more columns:

- **Promise kept (%)** — trips that arrived at or before the promise. About 90
  is calibrated; 100 means every promise was padded, which the loss charges for.
- **Avg promise (min)** — your mean prediction.
- **Negative preds** — predictions below zero.

Reference points on this split:

| submission | −Pinball | Promise kept |
|---|---|---|
| always 24.7 min — the 90th percentile of `y.csv` | −2.233 | 89.4% |
| `LinearRegression` on MSE | −2.087 | 55.7% |
| the lab's 12-64-64-1 MLP, trained on MSE | ≈ −1.74 | ≈ 54% |
| **`QuantileRegressor(quantile=0.9)` — the benchmark** | **−1.0429** | 88.4% |
| the lab's 12-64-64-1 MLP, trained on pinball | ≈ −0.94 | ≈ 89% |

The two MLP rows are the same network, the same optimizer and the same 30
epochs, over five seeds; only the loss changes. Trained on MSE, the network
aims at the average trip and keeps about half its promises. Trained on pinball,
it beats the best straight line. The constant keeps 89% of its promises and
still scores worst: it promises 24.7 minutes for a five-minute hop.

## Passing

The module counts this challenge validated when your best score reaches the
benchmark: **−Pinball ≥ −1.042916**, which the leaderboard shows as −1.0429. It
shows four decimals so that a score just either side of the bar reads
differently.

The bar measures your promises, not how you made them. An MSE-trained network
multiplied by one safety factor fitted on the training rows also clears it
(about −0.93 to −0.96), and so does scikit-learn's
`HistGradientBoostingRegressor(loss="quantile")` (about −0.93). What shows that
you wrote the loop is the notebook: its check cells — your pinball on a hand
example, one batch overfitted, the validation loss at every epoch, the best
checkpoint restored.

Submissions are scored once and deterministically: the same file always gets
the same number. The source data is public, so the held-out durations can be
looked up rather than predicted — that is not modelling and not what is being
assessed.
