# California Housing — Session 4

Predict the median house value of a California census district from eight
numeric features. 16,512 training districts, 4,128 to predict, one continuous
target.

This is the **regression** challenge of Session 4, and the one the session
works out in full. It exists to answer a question Sessions 2 and 3 left open:

> **You have a linear model and an honest way to measure it. What do you gain
> by replacing it with a neural network — and what do you have to get right to
> collect that gain?**

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s4-california-housing.ipynb)

The worked notebook builds the whole thing from the bottom: a tensor, a matrix
multiply done by hand, `requires_grad` and one `.backward()`, then the same
model as `nn.Linear`, then `nn.Sequential` with a hidden layer, then a
`DataLoader` and a real training loop with a validation split. Every line is
one you should be able to explain by the end. Open it, paste your API key, run
all.

Two shorter notebooks come before it and are worth doing first — they are
linked from the session's *Warm-up* lessons and need no account:

- **Optimization & Training** — gradient descent by hand, then the same descent
  performed by `torch.optim`.
- **CPU vs GPU** — what the hardware is actually doing, measured.

## The data

| file | rows | contents |
|---|---|---|
| `X.csv` | 16,512 | `id` + the 8 features |
| `y.csv` | 16,512 | `id,prediction` — the median house value |
| `X_submission.csv` | 4,128 | `id` + the same 8 features |

`X.csv` and `y.csv` are the labelled data — fit on them, and carve your own
validation split out of them. `X_submission.csv` is what the leaderboard
scores; its labels are held back, so it cannot serve as a validation set.

A random 80/20 split of scikit-learn's `fetch_california_housing` at
`random_state=42` — the 1990 US census, one row per block group. No missing
values, every column numeric.

| column | meaning |
|---|---|
| `median_income` | median household income, tens of thousands of dollars |
| `house_age` | median age of the houses, years |
| `avg_rooms` | rooms per household |
| `avg_bedrooms` | bedrooms per household |
| `population` | people in the block group |
| `avg_occupancy` | household members per household |
| `latitude`, `longitude` | where the block group is |

**The target is the median house value in units of $100,000** — so `2.5` means
$250,000 — and it is **capped at 5.0**. 965 of the 20,640 districts sit exactly
on that cap, which shows up as a spike at the right edge of the histogram. It
is a property of the data, not a bug, and no model can predict past it.

The two coordinates are the reason this dataset is here. `latitude` on its own
tells you almost nothing about price and `longitude` on its own tells you
almost nothing, but the pair tells you whether a district is in the Bay Area.
No straight line in the eight features can express that; a hidden layer can.

## What you submit

`submission.csv` — one row per id in `X_submission.csv`, in any order:

```csv
id,prediction
te_00000,1.943
te_00001,3.108
```

`prediction` is a real number in units of $100,000. Every id in `X_submission.csv`
must appear exactly once; a missing id, an unknown id, a duplicate, a
non-numeric value or a `NaN` is rejected with a message naming the line, and
scores 0.

A `NaN` here almost always means the training loss diverged rather than that
the CSV is malformed. Check the learning rate before checking the file.

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.submit(challenge_id=<id>, files=["submission.csv"])
```

## Scoring

**R²**, on the 4,128 held-out districts. Ranking is on R² rather than RMSE
because the leaderboard sorts descending and needs a higher-is-better number;
R² = 0 is exactly the model that predicts the training mean. RMSE and MAE are
shown too, in units of $100,000.

A fourth column, **Impossible preds**, counts predictions outside `[0.15,
5.0]` — the range the target cannot leave. It is there because a linear output
layer will happily predict a negative house price, and noticing that is worth
more than the score it costs.

## The point of this challenge

Every row measured on this exact split. Read the last column.

Every MLP row uses the notebook's own protocol: 60 epochs of Adam at `lr=1e-3`,
batch size 256, with the best-validation checkpoint restored at the end.

| model | test R² |
|---|---|
| predict the training mean | 0.000 |
| **LinearRegression** — the benchmark, and Session 2's ceiling | **0.576** |
| the same predictions, clipped to `[0.15, 5.0]` | 0.604 |
| MLP 8-64-64-1, **inputs not standardised**, 60 epochs | 0.541 |
| MLP 8-64-64-1, standardised, 20 epochs | 0.735 |
| MLP 8-64-64-1, standardised, 60 epochs | 0.775 |
| **the same, clipped to `[0.15, 5.0]`** — what the notebook submits | **0.776** |

Three things are true here, and all three are checkable:

1. **The network wins, and wins big.** 0.576 to 0.776 is the largest jump in
   this module — larger than anything tuning bought in Session 3. When the
   truth really is non-linear, capacity buys generalisation rather than
   memorisation, and this is what that looks like.
2. **The same network, without `StandardScaler`, scores 0.541 — worse than the
   straight line it was supposed to beat.** Same architecture, same optimiser,
   same 60 epochs, same best-checkpoint restore. `population` runs to 35,000
   and `avg_bedrooms` sits near 1,
   so the first layer's gradients are dominated by one column and the others
   barely move. Nothing warns you: the loss goes down, the run completes, the
   number is bad. This is the single most common way a first PyTorch model
   quietly fails.
3. **0.028 of R² is sitting on the floor.** The linear benchmark predicts a
   *negative* house value for 15 districts and $1.15M for another, in a dataset
   capped at $500k. Clipping the predictions to the target's known range —
   one line, no retraining — takes it from 0.576 to 0.604. Look at your
   predictions, not only at your loss.

The 20-epoch row is the fourth thing, and it is the one that costs people time:
0.735 is not a worse model, it is the same model stopped early. Watch the
validation loss and stop when it stops falling, which is the protocol Session 3
gave you.

The run is stable, so a low score is a bug rather than bad luck: across seeds
0, 1 and 2 the submitted score is 0.776, 0.776 and 0.779.

Submissions are scored once and deterministically — the same file always gets
the same number. The leaderboard is still a test set: choose on your validation
split, and use the board to confirm.
