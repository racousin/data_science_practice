# Batches, Epochs and the Training Loop

Every piece is on the table: tensors, a model, a loss, gradients, an optimizer.
The training loop runs them in order, on the data cut into small random batches,
and measures the model on validation rows after every pass. This lesson covers
the batches, the loop, and how to tell whether it works.

<!-- notes: 13 minutes, the longest lesson: it assembles lessons 3 to 6. Batch
and epoch are new words for this group (Session 2 used "epoch" without defining
it). The first figure comes from the website's unpublished draft of this
material; tools/figures/s4_pytorch_nutshell.py draws the other two. Dataset
and DataLoader are kept to a minimum: no Dataset class is taught, and the
hand-written epoch comes first, so DataLoader reads as a shortcut for four
known lines. Run the blocks live, in order, in one
session: each continues from the previous one, on random stand-ins with the
lab's shapes whose target is noise. Slow down on the picture and on the
overfit slide. The lab asks the students to write the loop themselves. -->

---

## Mini-batch gradient descent

Session 2 computed the loss and its gradient on all $n$ rows at every step.
Mini-batch gradient descent takes each step on a random subset $\mathcal{B}$ of
$B$ rows, a **batch**, and descends on its mean loss:

$$
\ell_{\mathcal{B}}(\theta) = \frac{1}{B} \sum_{i \in \mathcal{B}} L(y_i, f_\theta(x_i))
$$

![Batch, stochastic and mini-batch gradient descent on the same loss](assets/s4-pytorch-nutshell/training-loop-end-to-end/batch-sgd-minibatch.png)

- All $n$ rows: the exact gradient, but a full pass over the data per step. One
  row: cheap steps on an erratic path. $B$ from 32 to 512: the usual compromise.
- The batch mean estimates the full mean: right on average, noisier as $B$
  shrinks.

---

## Batch size and epoch

![The lab's 33,475 training rows cut into batches of 256: one update per batch, 131 per epoch, the last batch holding the 195 rows left over](assets/s4-pytorch-nutshell/training-loop-end-to-end/batch-and-epoch.png)

- **Batch size** $B$: the rows behind one update of $\theta$, one
  **iteration**.
- **Epoch**: one pass over all $n$ training rows, $\lceil n / B \rceil$
  iterations: 131 for the lab's 33,475 rows at $B = 256$, as in the figure.

The code of this lesson runs on random stand-ins with the lab's shapes: 33,475
training and 8,369 validation rows of 12 raw features between 0 and 50, and
noise as the target.

```python
torch.manual_seed(0)
X_tr, y_tr = torch.rand(33475, 12) * 50, torch.randn(33475, 1)
X_val, y_val = torch.rand(8369, 12) * 50, torch.randn(8369, 1)
B = 256
```

---

## Standardise the inputs

One learning rate serves every weight, and a weight's gradient grows with the
size of its input. A year, about 2,000, gets gradients about 2,000 times those
of a feature of size 1: gradient descent fitting a line to it is stable only
for $\eta < 2.5 \times 10^{-7}$; standardised, for any $\eta < 1$.

```python
mean = X_tr.mean(dim=0)                 # the training part only
std = X_tr.std(dim=0)
X_tr = (X_tr - mean) / std
X_val = (X_val - mean) / std            # the same mean and std
```

- Session 2's `StandardScaler` does the same. Fitting it on every row is
  Session 3's Leak 1.
- Keep `mean` and `std` with the weights: new rows need the same
  transformation ([lesson 8](/courses/python-ai-engineering/s4-pytorch-nutshell/course/save-and-load)).

---

## One epoch, by hand

No `Dataset` class and no `DataLoader` are needed: an epoch is a random order
of the rows, cut into slices of $B$.

```python
perm = torch.randperm(len(X_tr))        # a new random order each epoch
for i in range(0, len(X_tr), B):        # i = 0, 256, 512, ...
    xb = X_tr[perm[i:i + B]]            # the next B rows
    yb = y_tr[perm[i:i + B]]            # and their targets
```

`torch.randperm(n)` returns 0 to $n - 1$ in random order; `perm[i:i + B]` is
the next $B$ of those row numbers, and indexing both tensors with them keeps
each row with its target. The loop yields 131 batches: 130 of 256 rows, then
the 195 left over. A new order each epoch means new batches: the "stochastic"
in `torch.optim.SGD`.

---

## `DataLoader` does the same

```python
from torch.utils.data import TensorDataset, DataLoader
ds = TensorDataset(X_tr, y_tr)          # pairs row i of X_tr and y_tr
loader = DataLoader(ds, batch_size=B, shuffle=True)
len(loader)                             # 131, as by hand
```

- `for xb, yb in loader:` yields the epoch's batches, `xb` of shape (256, 12)
  and `yb` of shape (256, 1), the last of 195 rows, and reshuffles every
  epoch. This is the form most PyTorch code uses.
- `TensorDataset` keeps the tensors it was given: standardise before building
  it. Shuffle the training set only.
- A custom `Dataset` class is for data that does not fit in memory, such as
  images or text: *MS2A Machine Learning Practice*, Session 4.

---

## The loop, in one picture

![One epoch: train on every batch, validate, keep the best, log; then restore the best](assets/s4-pytorch-nutshell/training-loop-end-to-end/training-loop.png)

The dashed box runs once per epoch, the blue box once per batch. The objects it
calls, with a smaller network than the lab's:

```python
model = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 1))
loss_fn = nn.MSELoss()                  # the lab uses its pinball loss
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

## The training step

```python
for xb, yb in loader:                   # one epoch: every batch once
    opt.zero_grad()                     # lesson 4
    loss_fn(model(xb), yb).backward()   # lessons 5, 6 and 3
    opt.step()                          # lesson 4
```

- Call `model.train()` before it. Some layers act differently in training and
  in evaluation, such as dropout (Session 3), which switches off random neurons
  during training only. Forgetting the switch raises no error.
- On a GPU, move each batch first: `xb, yb = xb.to(device), yb.to(device)`
  ([lesson 1](/courses/python-ai-engineering/s4-pytorch-nutshell/course/why-tensors)).
- One run of this loop is one epoch, 131 updates, in under 0.1 s on an Apple M4
  CPU. The picture's outer loop repeats it once per epoch.

---

## The validation step

```python
model.eval()
with torch.no_grad():                   # no graph: nothing to update
    val_loss = loss_fn(model(X_val), y_val).item()
```

- Once per epoch, after the training step, on rows the model never trains on:
  Session 3's validation split, in time order when the future is what gets
  predicted. All 8,369 rows go through at once, with no loader.
- `.item()` gives a Python float, to print or log. Here it is about 1.0, the
  variance of the noise: nothing in the features predicts the target.
- Log the epoch, the training loss and `val_loss`. If training falls while
  validation rises, the network overfits (Session 3): keep the epoch with the
  lowest `val_loss` (lesson 8).

---

## First, check it can learn: overfit one batch

Before a long run, train a fresh model with a fresh optimizer on one fixed
batch:

```python
model = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
```

```python
for step in range(500):                 # the same 64 rows, 500 times
    opt.zero_grad()
    loss_fn(model(X_tr[:64]), y_tr[:64]).backward()
    opt.step()
```

The loss on those 64 rows falls from about 1 to below $10^{-4}$, in under
0.1 s on an Apple M4 CPU. A working network memorises 64 rows, even of noise;
the result is no model (its validation loss is now about 2), only proof that
the loop drives a loss down. If the loss does not fall, the bug is in the
model, the loss or the three lines: look there before tuning anything.

---

## Reading the symptoms

| Symptom | Likely cause | First thing to try |
|---|---|---|
| the loss explodes or turns `nan` | `lr` too high, or unscaled inputs | divide `lr` by 10; standardise |
| the loss does not move | `opt.step()` missing, or `lr` far too small | overfit one batch |
| the loss falls, then climbs and swings | `opt.zero_grad()` missing | the three lines, in order (lesson 4) |
| every prediction is the same | target $(B,)$ against output $(B, 1)$ | equal shapes (lesson 2) |
| training falls, validation rises | overfitting | keep the best epoch (lesson 8) |

None of these raises an error; `nn.MSELoss` only warns about the shapes, and a
hand-written loss does not. The logged losses are the only witness.

---

## Check yourself

1. A training set has 41,844 rows. With a batch size of 512, how many updates
   make one epoch?

   **Answer.** $\lceil 41844 / 512 \rceil = 82$: 81 full batches and a last one
   of 372 rows.

2. Run this. What does it print, and which batch is the short one?

   ```python
   from torch.utils.data import TensorDataset, DataLoader
   ds = TensorDataset(torch.zeros(25, 3), torch.zeros(25, 1))
   dl = DataLoader(ds, batch_size=10, shuffle=True)
   print(len(dl), [len(xb) for xb, yb in dl])
   ```

   **Answer.** `3 [10, 10, 5]`. Shuffling reorders the rows, not the cut: the
   last batch is always the short one.

3. Why does the validation step run under `model.eval()` and `torch.no_grad()`,
   and never call `opt.step()`?

   **Answer.** It measures the model and must not change it: no update, no
   graph to record, and every layer in its evaluation behaviour.
