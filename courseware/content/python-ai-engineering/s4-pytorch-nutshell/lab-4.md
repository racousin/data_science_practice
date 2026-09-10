# Lab 1 — Taxi Arrival Promise

Write the training loop yourself, on a problem where the usual loss is the
wrong one. A ride-hailing app shows an arrival time before the ride starts. You
train the lab's network to promise a time the ride beats 9 times out of 10,
step by step in a Colab notebook, and submit its promises to ML-Arena.

**Time:** 85 minutes. **Deliverable:** a leaderboard entry on challenge 189 at
or above the pass bar, **−Pinball ≥ −1.0429**.

<!-- notes: 85 minutes. 5 for setup: Colab, the ML-Arena key, and a free W&B
account (about five minutes; a student without one switches W&B to offline and
reads the notebook's own plot). 5 to present the objective and the ladder: the
network trained on MSE keeps about half its promises, which is why the loss is
theirs to write. 60 for Steps 1 to 5c; walk the room at Step 4 (a loss that
does not fall on one batch is a bug in the code, not in the settings) and at
Step 5c. 15 to submit and tick the checklist. Each account may deploy 2 times
per rolling 24 hours across all challenges (daily_agent_deploy_limit), so
Session 3 submissions from the last 24 hours count against it: variants are
compared on the notebook's validation score, and only the best-validation
checkpoint is submitted. The solution notebook, aie-s4-taxi-eta-solution.ipynb,
is linked after the session. -->

---

## The challenge

**Challenge:** <https://ml-arena.com/viewchallenge/189>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s4-taxi-eta.ipynb)

- 52,305 New York green-taxi trips, January 2024. `X.csv` and `y.csv`: the
  first 25 days, 41,844 trips, in pickup order. `X_submission.csv`: the rest of
  the month, 10,461 trips, shuffled.
- 12 features: trip distance (miles), pickup hour, day of week, a weekend flag,
  and the pickup and drop-off borough (four 0/1 columns each).
- Target: the trip's duration in minutes. You submit `id,prediction`: your
  promised duration.

---

## Setup, the first five minutes

1. Open the notebook with the Colab badge and paste your ML-Arena key
   (`mlk_user_...`, the one from Sessions 2 and 3) into `API_KEY`.
2. In the notebook's W&B cell, choose one line to uncomment. With a free
   account (wandb.ai, GitHub sign-in works): `wandb.login()`, then paste the
   key from wandb.ai/authorize.
3. No account: the `WANDB_MODE = "offline"` line. The run stays on disk, and
   the notebook's own plot draws the curves.
4. Run the given cells down to Step 1: download, split, standardisation,
   tensors.

---

## The objective

**Promise an arrival time the ride beats 9 times out of 10.**

$$
L_{0.9}(y, \hat{y}) = \max(0.9\,(y - \hat{y}),\ -0.1\,(y - \hat{y}))
$$

- $y$ is the trip's duration and $\hat{y}$ the promise. A minute late costs
  0.9 and a minute early 0.1, so the best promise is the 90th percentile of the
  duration, not its mean
  ([lesson 6](/courses/python-ai-engineering/s4-pytorch-nutshell/course/losses)).
- The leaderboard ranks on **−Pinball**, the mean loss negated: higher is
  better, 0 is perfect, four decimals. **Promise kept (%)**, the share of trips
  that arrive at or before the promise, should be near 90.
- A real app would base the promise on the length of the planned route; this
  data stands in with the metered `trip_distance`, known only once the ride is
  over.

---

## The ladder

| Submission | −Pinball | Promise kept |
|---|---|---|
| Always 24.7 min, the 90th percentile of `y.csv` | −2.233 | 89.4% |
| `LinearRegression` on MSE | −2.087 | 55.7% |
| The lab's MLP, trained on MSE | ≈ −1.74 | ≈ 54% |
| **`QuantileRegressor(quantile=0.9)`: the pass bar** | **−1.0429** | 88.4% |
| The lab's MLP, trained on pinball (the solution) | −0.941 | 87.7% |

The two MLP rows are the same network and the same 30 epochs; only the loss
changes. The bar checks the promises, not how they were made: an MSE network
multiplied by one factor fitted on the training rows also clears it (about
−0.93 to −0.96). What shows that you wrote the loop is the notebook's check
cells.

---

## If you finish early

Compare each variant on the validation part with `evaluate` or
`leaderboard_score`, not by submitting: every submission uses one of your two
daily slots.

- Train the same network on `nn.MSELoss()` and compare its **Promise kept**
  with the pinball network's.
- Train with `tau=0.5`. Which of the two numbers moves, and why?
- Try a wider or deeper network, or another learning rate. Log each run to W&B
  with its `config`, compare the validation curves, and keep the best.
