# Tracking with Weights & Biases

A training run is an experiment: settings in, curves out. Printed numbers vanish
with the notebook, and comparing two learning rates needs both curves on one
chart. Weights & Biases (W&B) records each run's settings and the numbers it
logs, and draws them.

<!-- notes: 7 minutes. The slides follow the loop's order: init before
training, a float collected inside the training step, one log per epoch, finish
at the end. The blocks run in order in one Python session; the loss, the batch
and val_loss are stand-ins for what the lab computes. Run them live with the
environment variable WANDB_MODE=disabled, so no account is needed on the
projector. Students create a free wandb.ai account at the start of the lab, not
now; without one, WANDB_MODE=disabled and the matplotlib plot give the same two
curves. Both figures are the lab's network on the lab's data (Adam, batch 256,
seed 0, one run per learning rate), computed by
tools/figures/s4_pytorch_nutshell.py; with one seed per rate, rerun a close pair
with another seed before trusting the difference. Checked with wandb 0.30.0,
torch 2.14.0 and matplotlib 3.11.1. MS2A Machine Learning Practice uses
TensorBoard for the same job: the habit transfers, the calls differ. -->

---

## `init`: a run and its settings

```python
import wandb
run = wandb.init(project="aie-s4-taxi", config={"lr": 1e-3})
model = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=run.config.lr)
```

- A **run** is one training: its settings and the numbers it logs. A
  **project**, here `"aie-s4-taxi"`, holds the runs you compare.
- `config` records the hyperparameters, so each curve stays attached to the
  settings that produced it. Reading `lr` back from `run.config` keeps the
  record and the run in agreement: `1e-3` is typed once.
- The call names no mode: an environment variable chooses online, offline or
  disabled (see *Online, offline, disabled*).

---

## One number per epoch

For training, log the epoch's mean loss per row, built from the batch losses the
training step already computes, with no second pass over the data:

$$
\ell_{\mathrm{train}} = \frac{1}{n} \sum_{\mathcal{B}} |\mathcal{B}| \, \ell_{\mathcal{B}}
$$

The sum runs over the epoch's batches: each batch's mean loss, from the training
step of [lesson 7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end),
weighted by its $|\mathcal{B}|$ rows; $n$ rows in all. Stand-ins for lesson 7's
loss and one batch:

```python
loss_fn = nn.L1Loss()                 # any loss from lesson 6
xb, yb = torch.randn(256, 12), torch.randn(256, 1)   # one batch
total = 0.0                           # reset before each epoch
```

---

## Log numbers, not tensors

```python
loss = loss_fn(model(xb), yb)         # the batch's loss: a tensor
loss.backward()
opt.step()
total += loss.item() * len(xb)        # a float: no graph kept
```

- Lesson 7's training step, after `opt.zero_grad()`, with the loss kept in a
  variable and one line added.
- `.item()` returns a Python float. `total += loss * len(xb)` would make
  `total` a tensor carrying every batch's autograd history: memory grows at
  each step.
- Multiplying by `len(xb)` gives the short last batch its true weight.
- The weights move during the epoch, so this mean mixes early and late weights.
  In the first epoch it can sit well above a validation loss measured at the
  epoch's end.

---

## `log` each epoch, `finish` once

```python
train_loss = total / len(xb)          # rows in this one-batch epoch
val_loss = 1.02                       # stand-in: lesson 7's val step
run.log({"train_loss": train_loss, "val_loss": val_loss}, step=1)
run.finish()                          # once, after the last epoch
```

- In the lab, `train_loss = total / len(loader.dataset)`: every training row
  once. `val_loss` comes from lesson 7's validation step, already a float.
- `log` records a dictionary of numbers at a step; with `step=epoch`, the
  x-axis counts epochs.
- `init`, `log`, `finish`: the three calls. After `finish`, the next
  `wandb.init` starts a new run.

---

## Online, offline, disabled

The environment variable `WANDB_MODE`, a setting of the running process that
wandb reads, chooses the mode. The code stays the same on every machine.

- **online**, the default: charts at wandb.ai while the run trains. It needs an
  account: `wandb.login()` asks for an API key, created at wandb.ai/authorize,
  and stores it on the machine.
- **offline**: the run is written under `./wandb` and becomes charts only after
  `wandb sync`, which needs an account too.
- **disabled**: every call does nothing, and no account is needed. Automated
  tests run this way.

Set it before the first `wandb.init` of the session, for instance
`os.environ["WANDB_MODE"] = "disabled"` in the first cell: wandb 0.30 ignores a
later change until the runtime restarts.

---

## Both losses on one chart

W&B draws one chart per logged key, so `train_loss` and `val_loss` start on
separate charts. To overlay them, open the `train_loss` chart's settings (the
gear icon that appears on hover) and add `val_loss` to its y-axis.

Without W&B, keep the same numbers in a list: each epoch, beside `run.log`,
call `history.append((train_loss, val_loss))`. Then:

```python
import matplotlib.pyplot as plt
history = [(train_loss, val_loss)]    # one pair per epoch
plt.plot(history, label=["train_loss", "val_loss"])
plt.legend()
```

`plt.plot` draws one line per column: each loss against the epoch index,
counted from 0.

---

## Reading one run

![Training and validation pinball loss per epoch, one run of the lab's network at lr = 1e-3](assets/s4-pytorch-nutshell/weights-and-biases/train-val-curves.png)

Validation reaches its lowest, 0.981, at epoch 28 and ends at 0.984: the curve
has flattened. From epoch 10 on it stays 0.08 to 0.11 above training, and the
gap does not grow: no overfitting. Part of the gap is the data: the validation
trips (21 to 25 January) run longer, 90th percentile 25.8 minutes against 24.3.

---

## Comparing runs

![Validation loss per epoch for four learning rates, one run each](assets/s4-pytorch-nutshell/weights-and-biases/lr-comparison.png)

Four runs in one project: W&B draws one line per run. `1e-4` is still falling
at epoch 30; `1e-1` is noisy and ends at 1.225. `1e-2` beats `1e-3` on 27 of
30 epochs but jumps: best 0.957 at epoch 29, last 1.003, hence keeping the best
epoch ([lesson 8](/courses/python-ai-engineering/s4-pytorch-nutshell/course/save-and-load)).
Choose on validation, never on the leaderboard: it is the test set.

---

## Check yourself

1. What goes in `config`, and what goes in `log`?

   **Answer.** `config`: the settings, once per run (learning rate, width,
   batch size). `log`: the measurements, every epoch.

2. Batches of 256, 256 and 8 rows have mean losses 1.0, 1.0 and 3.0. Run
   this: which printed number is the epoch's training loss?

   ```python
   losses = [1.0, 1.0, 3.0]              # each batch's mean loss
   sizes = [256, 256, 8]                 # rows in each batch
   total = sum(b * n for b, n in zip(losses, sizes))
   print(total / sum(sizes), sum(losses) / len(losses))
   ```

   **Answer.** `1.0307692307692307 1.6666666666666667`. The first: each row
   counts once. The second gives the 8-row batch the weight of 256 rows.

3. At epoch 30, training loss 0.90 and validation loss 0.98; the gap has
   stayed near 0.1 since epoch 10. Is the network overfitting?

   **Answer.** No. Overfitting is a gap that grows: validation turning up
   while training keeps falling. A steady gap is normal on unseen rows.
