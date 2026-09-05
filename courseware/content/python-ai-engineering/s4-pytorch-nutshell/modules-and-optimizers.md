# Modules & Optimizers

The two abstractions that turn autograd into a usable library: `nn.Module` holds
the parameters, `torch.optim` updates them.

<!-- notes: 30 minutes. Get them to count parameters by hand once — it makes
shapes concrete in a way that reading `nn.Linear` does not. -->

---

## nn.Linear

One layer is an affine map:

$$
y = xW^T + b
$$

```python
import torch
import torch.nn as nn

layer = nn.Linear(in_features=10, out_features=5)

layer.weight.shape    # torch.Size([5, 10])
layer.bias.shape      # torch.Size([5])
```

```python
x = torch.randn(32, 10)     # batch of 32
layer(x).shape              # torch.Size([32, 5])
```

The batch dimension passes through untouched. Only the **last** dimension has to
match `in_features`.

---

## Parameter counting

`nn.Linear(in, out)` has `in * out + out` parameters.

```python
sum(p.numel() for p in layer.parameters())      # 10*5 + 5 = 55
```

Worth doing by hand once. It is how you notice that the flatten before your
first linear layer is producing 100k features when you expected 500.

---

## Activations

Without a non-linearity between them, stacked linear layers collapse into a
single linear layer — a two-layer network with no activation is exactly as
expressive as one layer.

```python
nn.ReLU()        # max(0, x)  — the default
nn.Sigmoid()     # (0, 1)     — binary output
nn.Tanh()        # (-1, 1)
nn.GELU()        # smooth ReLU; standard in transformers
```

Start with ReLU. Change only when you have a measured reason.

---

## Sequential

For a straight-line stack:

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

Note there is **no activation after the final layer** — the loss function
applies it. Adding a softmax here is the most common beginner error in PyTorch.

---

## Subclassing nn.Module

For anything with structure — skip connections, branches, anything conditional:

```python
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()                     # required
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

```python
model = MLP(784, 128, 10)
out = model(x)            # calls forward() — never call model.forward(x)
```

`model(x)` runs the hooks PyTorch relies on. `model.forward(x)` skips them.

---

## What Module gives you

```python
model.parameters()        # every parameter, for the optimizer
model.to(device)          # move the whole model
model.state_dict()        # weights, for saving
model.train()             # training mode (dropout on, batchnorm updating)
model.eval()              # evaluation mode
```

Assigning `nn.Linear` to `self.fc1` registers it automatically. A layer stored
in a plain Python list is **not** registered — its parameters will not be
trained. Use `nn.ModuleList` for that.

---

## train() vs eval()

```python
model.train()      # before the training loop
model.eval()       # before validation / test
```

This changes behaviour for dropout (active vs identity) and batch norm (updating
running statistics vs using them).

Forgetting `model.eval()` gives you a validation score that is worse and noisier
than the truth. It is silent.

---

## Loss functions

```python
nn.MSELoss()             # regression
nn.L1Loss()              # regression, robust to outliers
nn.BCEWithLogitsLoss()   # binary classification — takes raw logits
nn.CrossEntropyLoss()    # multi-class — takes raw logits + int64 labels
```

---

## CrossEntropyLoss, carefully

```python
criterion = nn.CrossEntropyLoss()

logits = model(x)                 # (batch, n_classes) — raw, unnormalised
labels = torch.tensor([3, 1, 4])  # (batch,) int64 class INDICES
loss = criterion(logits, labels)
```

Two rules that account for most of the confusion:

- It applies log-softmax **internally**. Do not add a softmax to your model.
- Labels are integer indices, not one-hot vectors, and must be `int64`.

Doing the softmax yourself makes training silently worse — it does not error.

---

## Optimizers

You have gradients. An optimizer decides what to do with them.

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

`model.parameters()` is what connects the optimizer to the model. Pass the wrong
model's parameters and training does nothing, silently.

---

## SGD vs Adam

| | SGD (+ momentum) | Adam |
|---|---|---|
| Per-parameter learning rate | no | yes |
| Tuning needed | more | less |
| Typical `lr` | 0.01–0.1 | 1e-4–1e-3 |
| Final quality | often slightly better | converges faster |

**Use Adam at `lr=1e-3` unless you have a reason not to.** It is the right
default for everything in this course.

---

## The three lines

```python
optimizer.zero_grad()    # clear
loss.backward()          # compute
optimizer.step()         # apply
```

Always in this order, always together. `step()` before `backward()` applies last
iteration's gradients; skipping `zero_grad()` accumulates them.

---

## Learning rate is the hyperparameter

If you tune one number, tune this one.

| Symptom | Diagnosis |
|---|---|
| Loss is `nan` after a few steps | far too high |
| Loss oscillates, does not settle | too high |
| Loss falls smoothly | about right |
| Loss barely moves | too low |

Try `1e-2, 1e-3, 1e-4` and look at the curves before touching anything else in
the architecture.

---

## Scheduling

Reduce the learning rate as training progresses — large steps early, fine
adjustment later:

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()          # once per EPOCH, not per batch
```

```python
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

`ReduceLROnPlateau` takes the validation metric: `scheduler.step(val_loss)`.

---

## Saving and loading

```python
torch.save(model.state_dict(), "model.pt")

model = MLP(784, 128, 10)                          # same architecture
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

Save the `state_dict`, not the model object — pickling the object couples the
file to your source layout, and it breaks the moment you rename a module.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import torch.nn as nn
   layer = nn.Linear(in_features=10, out_features=5)
   print(sum(p.numel() for p in layer.parameters()))   # -> 55
   ```

2. The `nn.Sequential` stack in this lesson ends with `nn.Linear(64, 10)` and no
   activation. Why is adding an `nn.Softmax()` there wrong?

   **Answer.** `nn.CrossEntropyLoss()` applies log-softmax internally and expects
   raw logits. A softmax in front of it does not raise — it makes training
   silently worse. The same holds for `nn.BCEWithLogitsLoss()`.

3. Your loss is `nan` after a few steps. What does the diagnosis table say, and
   what does this lesson tell you to change before touching the architecture?

   **Answer.** The learning rate is far too high. Learning rate is *the*
   hyperparameter: try `1e-2, 1e-3, 1e-4` and look at the curves before changing
   anything else. The course default is Adam at `lr=1e-3`.

4. You stored some layers in a plain Python list and they never train. Why?

   **Answer.** Only a layer assigned to an attribute — `self.fc1 = nn.Linear(...)`
   — is registered, so one inside a plain list never appears in
   `model.parameters()` and the optimizer never sees it. Use `nn.ModuleList`.
