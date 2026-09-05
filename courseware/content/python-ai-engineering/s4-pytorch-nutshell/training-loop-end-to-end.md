# Training Loop End to End

Everything from the last three lessons, assembled into the loop you will write
for the rest of the year.

<!-- notes: 30 minutes. Type the whole thing live. Then break it deliberately —
remove zero_grad, remove model.eval() — and let them see the symptoms. -->

---

## Data: Dataset and DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

val_ds = TensorDataset(X_val, y_val)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)
```

- `shuffle=True` for training — otherwise the model learns the ordering
- `shuffle=False` for validation — you want a comparable number each epoch
- Larger validation batches: no gradients, so no memory pressure

---

## A custom Dataset

Three methods, and it works everywhere in the ecosystem:

```python
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.X = torch.tensor(df.drop(columns="y").values, dtype=torch.float32)
        self.y = torch.tensor(df["y"].values, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
```

Note the dtypes: `float32` features, `int64` labels. Getting this wrong produces
an error at the loss, far from the cause.

---

## The complete script

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

---

## The training epoch

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, running = 0, 0, 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)

    return running / total, correct / total
```

`loss.item() * xb.size(0)` weights by batch size, so the final average is correct
even when the last batch is short.

---

## The validation epoch

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, running = 0, 0, 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)

    return running / total, correct / total
```

Three differences from training: `model.eval()`, `@torch.no_grad()`, and no
optimizer. All three matter.

---

## Putting it together

```python
for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, device)
    va_loss, va_acc = evaluate(model, val_dl, criterion, device)

    print(f"epoch {epoch:3d}  "
          f"train {tr_loss:.4f}/{tr_acc:.3f}  "
          f"val {va_loss:.4f}/{va_acc:.3f}")
```

Print both. A training curve alone cannot tell you about overfitting, which is
the thing you are watching for.

---

## Early stopping

```python
best_val, patience, wait = float("inf"), 10, 0

for epoch in range(1, epochs + 1):
    ...
    if va_loss < best_val:
        best_val, wait = va_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"early stop at epoch {epoch}")
            break

model.load_state_dict(torch.load("best.pt"))
```

Save on improvement, restore at the end. Without the restore you keep the
*last* model, which is the overfitted one — the whole point was the earlier one.

---

## Overfit one batch first

Before a full training run, prove the plumbing works:

```python
xb, yb = next(iter(train_dl))
xb, yb = xb.to(device), yb.to(device)

for _ in range(200):
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()

print(loss.item())      # should be ~0
```

If a model cannot memorise 64 examples, it will not learn 60,000. This catches
wrong shapes, a broken graph, a missing `zero_grad`, the wrong loss — in twenty
seconds instead of twenty minutes.

<!-- notes: If they take one habit from Session 4, this is the one. -->

---

## The failure table

| Symptom | Cause |
|---|---|
| Loss is `nan` | learning rate too high; `log(0)`; unnormalised inputs |
| Loss flat from step 0 | `lr` far too low; optimizer got the wrong parameters |
| Train falls, val rises | overfitting — regularise, stop earlier, get more data |
| Train and val both flat | underfitting — more capacity, better features |
| Val worse than train by a lot, noisily | `model.eval()` missing |
| Loss decreases then explodes | `zero_grad()` missing |
| RuntimeError: device mismatch | a tensor was not `.to(device)` |
| Memory grows each epoch | storing tensors that still carry a graph |

---

## The checklist

- [ ] `torch.manual_seed(...)` set
- [ ] Model and every batch `.to(device)`
- [ ] `model.train()` / `model.eval()` in the right places
- [ ] `zero_grad` → `backward` → `step`, in that order
- [ ] Validation inside `torch.no_grad()`
- [ ] No softmax before `CrossEntropyLoss`
- [ ] Labels are `int64`, features are `float32`
- [ ] Overfit one batch before the real run
- [ ] Best checkpoint saved and restored

---

## Where this goes next

This loop is the skeleton of everything in *MS2A - Machine Learning Practice*: the CNN in Session
5, the transformer fine-tune in Session 8, the policy-gradient update in Session
10. The data and the model change; `zero_grad / backward / step` does not.

---

## Check yourself

1. Run this — it is the "overfit one batch" check from this lesson, on 32 random
   examples. A couple of seconds on a laptop CPU.

   ```python
   import torch, torch.nn as nn, torch.optim as optim
   torch.manual_seed(0)
   model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
   opt, crit = optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss()
   xb, yb = torch.randn(32, 784), torch.randint(0, 10, (32,))
   for _ in range(200):
       opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
   print(loss.item() < 0.01)     # -> True   (a real run lands near 3e-4)
   ```

2. `evaluate()` differs from `train_epoch()` in exactly three ways. Name them.

   **Answer.** `model.eval()` instead of `model.train()`, the `@torch.no_grad()`
   decorator, and no optimizer at all — no `zero_grad`, no `backward`, no `step`.

3. Your validation score is much worse than training and noisy from epoch to
   epoch. Which row of the failure table is this, and what is the fix?

   **Answer.** `model.eval()` is missing. In training mode dropout is still active
   and batch norm is still updating its running statistics, so each validation
   pass measures a slightly different model.

4. Early stopping already saves the best checkpoint. Why must the loop also load
   it back afterwards?

   **Answer.** Without the restore you keep the *last* model, which is the
   overfitted one — the earlier checkpoint was the whole point. Hence
   `model.load_state_dict(torch.load("best.pt"))` after the loop ends.
