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
