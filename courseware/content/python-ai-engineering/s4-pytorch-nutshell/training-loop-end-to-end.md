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

Session 2-3 computed the loss and its gradient on all $n$ rows at every step.
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


---

## One epoch, by hand


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
    opt.zero_grad()
    loss_fn(model(xb), yb).backward()
    opt.step()
```

- Call `model.train()` before it. Some layers act differently in training and
  in evaluation, such as dropout , which switches off random neurons
  during training only. Forgetting the switch raises no error.
- On a GPU, move each batch first: `xb, yb = xb.to(device), yb.to(device)`

---

## The validation step

```python
model.eval()
with torch.no_grad():                   # no graph: nothing to update
    val_loss = loss_fn(model(X_val), y_val).item()
```

- Once per epoch, after the training step, on rows the model never trains on:
  validation split, in time order when the future is what gets
  predicted.
- `.item()` gives a Python float, to print or log.
- Log the epoch, the training loss and `val_loss`. If training falls while
  validation rises, the network overfits : keep the epoch with the
  lowest `val_loss` (lesson 8).


![early-stopping.png](assets/s4-pytorch-nutshell/training-loop-end-to-end/early-stopping.png)

---

## The `state_dict`: every parameter, by name

```python
sd = model.state_dict()          # an ordered dict: name -> tensor
list(sd)[:3]                     # ['0.weight', '0.bias', '2.weight']
sd["0.weight"].shape             # torch.Size([64, 12])
```

- A key is the layer's position in the `nn.Sequential`, then the parameter's
  name: `0.weight`, `0.bias`, `2.weight`, `2.bias`, `4.weight`, `4.bias`. The
  ReLUs, at positions 1 and 3, hold no parameters and have no key.
- In a subclassed module (lesson 5) the key is the attribute's name:
  `fc1.weight`, `fc1.bias`, …
- Each value is a tensor; together they hold the 5,057 numbers.

---

## Save the numbers, rebuild the code

```python
torch.save(model.state_dict(), "weights.pt")
```

Later
```python
model2 = model.copy()                # same code, new random weights
model2.load_state_dict(torch.load("weights.pt"))
torch.equal(model2[0].weight, model[0].weight)      # True
```

You can continue training from this point..

---
