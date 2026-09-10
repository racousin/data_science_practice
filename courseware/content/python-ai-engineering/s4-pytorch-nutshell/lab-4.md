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

## What is given, what you write

Given: the download, the split (the last 20% of `X.csv`, in time order, is the
validation part), standardisation, the tensors `X_tr_t`, `y_tr_t`, `X_val_t`,
`y_val_t`, the scorer `leaderboard_score`, and the plot, save, predict and
submit cells. Each step you write ends on a check cell that reads these names.

| Step | You write | The check reads | Lesson | The check prints |
|---|---|---|---|---|
| 1 | 12 → 64 → 64 → 1, ReLU between | `make_model()`, `model` | [5](/courses/python-ai-engineering/s4-pytorch-nutshell/course/modules-and-optimizers) | 5057 parameters, output `(5, 1)` |
| 2 | the loss, a scalar tensor | `pinball(pred, target, tau=TAU)` | [6](/courses/python-ai-engineering/s4-pytorch-nutshell/course/losses) | 1.05 on the hand example |
| 3 | Adam at `1e-3`; batches of 256, shuffled | `opt`, `loader` | [4](/courses/python-ai-engineering/s4-pytorch-nutshell/course/optimizers), [7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end) | 131 batches per epoch |
| 4 | 500 steps on 64 rows: a fresh model, Adam at `1e-2` | `probe`, `probe_opt`, `first`, `last` | [7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end) | `first -> last`, e.g. 11.80 -> 0.564 |
| 5a | one training epoch | `train_one_epoch(model, loader, opt)` → float | [7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end), [9](/courses/python-ai-engineering/s4-pytorch-nutshell/course/weights-and-biases) | two epochs of a fresh model, e.g. 6.015 -> 1.340 |
| 5b | one validation pass | `evaluate(model, X, y)` → `(loss, kept_pct)` | [7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end) | your two numbers beside the scorer's |
| 5c | 30 epochs, keeping the best | `history`, `best_val`, `best_epoch`, `best_state` | [7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end), [8](/courses/python-ai-engineering/s4-pytorch-nutshell/course/save-and-load), [9](/courses/python-ai-engineering/s4-pytorch-nutshell/course/weights-and-biases) | `best_epoch`; `best_val` below 1.10 |

---

## Step 5, in words

- **5a** — `model.train()`, then for each batch: `opt.zero_grad()`, the
  pinball loss, `backward()`, `opt.step()`, and
  `total += loss.item() * len(xb)`. Return `total / len(loader.dataset)`, the
  epoch's mean training pinball, as a float.
- **5b** — `model.eval()`, predict under `torch.no_grad()`, and return the
  pinball loss as a float and `kept_pct`, the percentage of rows with
  `y <= pred`.
- **5c** — for each of `EPOCHS = 30` epochs: 5a on `loader`, 5b on `X_val_t`
  and `y_val_t`, one dict appended to `history` and logged to W&B. When the
  validation loss is the lowest so far, record `best_val`, `best_epoch` and
  `copy.deepcopy(model.state_dict())`. After the loop, load that copy back into
  `model`.

Expect a best validation pinball near 0.98 (0.983 at epoch 29 in the
solution run), and seconds, not minutes, per run on a CPU.

---

## Submit

```python
client.submit(challenge_id=CHALLENGE_ID, files=["submission.csv"])
client.leaderboard(CHALLENGE_ID)
```

- The given cells save the model to `taxi.pt` with the scaler's mean and std,
  reload it, standardise `X_submission` with that mean and std, predict, and
  check the file before writing it: one row per id of `X_submission.csv`, one
  finite prediction in minutes each.
- The scorer rejects a missing, unknown, duplicate or empty id, a non-number, a
  `NaN` or an infinity, and names the line. Negative predictions are scored.
- Each account may submit **2 times per rolling 24 hours, across all
  challenges**, and a file the scorer rejects still uses one. Validate first:
  choose on the notebook's validation score, then submit once.

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

---

## Did you validate this session?

- [ ] Step 1's check prints 5,057 parameters and an output of shape `(5, 1)`
- [ ] Step 2's check passes: my `pinball` gives 1.05 on the hand example
- [ ] Step 3's check passes: `loader` has 131 batches per epoch
- [ ] Step 4's check passes: one batch ends below a tenth of its first loss
- [ ] The checks of Steps 5a and 5b pass
- [ ] Step 5c's check prints my `best_epoch` and passes: `best_val` < 1.10
- [ ] A chart — W&B or the notebook's plot — shows both losses per epoch
- [ ] My submission is on the leaderboard of AIE S4 — Taxi Arrival Promise (#189)
- [ ] My score beats the baseline: **−Pinball ≥ −1.0429**
