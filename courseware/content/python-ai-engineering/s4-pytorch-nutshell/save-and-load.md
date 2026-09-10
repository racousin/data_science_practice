# Saving and Loading

A trained network is its code plus a set of numbers, the parameters $\theta$.
Saving the numbers turns a training run into a file that can be reloaded,
compared and used to predict. It is also how the best epoch is kept rather than
the last one.

<!-- notes: 7 minutes. Three ideas. (1) Save the state_dict, the numbers; the
architecture is rebuilt by running the same code. (2) Keep the best epoch with
copy.deepcopy, because state_dict() returns the model's live tensors. (3) A
checkpoint holds every piece of the prediction formula: the architecture's
settings, theta, and the scaler's mu and s; the reload slide uses every key.
Early stopping was named in Session 3: recall it in one sentence. The blocks use
stand-in data (random rows, a list of three validation losses) so that each one
runs on its own; say so, the lab supplies the real values. -->

---

## A trained model is code plus numbers

```python
def make_model(hidden=64):                 # lesson 5's network
    return nn.Sequential(nn.Linear(12, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))
```

- The **architecture** is code: each call to `make_model` builds the same
  layers, with new random weights.
- The **parameters** $\theta$ are numbers, 5,057 of them here (lesson 5), and
  they are all that training changes.
- Saving a model means saving the numbers. Loading it means running the same
  code, then putting the numbers back.

---

## The `state_dict`: every parameter, by name

```python
model = make_model()
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
model2 = make_model()                  # same code, new random weights
model2.load_state_dict(torch.load("weights.pt"))
torch.equal(model2[0].weight, model[0].weight)      # True
```

- `load_state_dict` copies each tensor into the parameter of the same name and
  reports `<All keys matched successfully>`. `model[0]` is the first layer;
  `torch.equal` is `True` when two tensors have the same shape and values.
- The file holds no architecture. Loaded into `make_model(32)`, it fails:
  `size mismatch for 0.weight: copying a param with shape torch.Size([64, 12])
  from checkpoint, the shape in current model is torch.Size([32, 12])`.
- `torch.save(model)` would store the object itself, which ties the file to the
  code that defined its class. Since PyTorch 2.6, `torch.load` refuses such a
  file by default: `Weights only load failed`.

---

## Keep the best epoch, not the last

![Training loss keeps falling while validation loss is lowest at epoch 266 and then rises](assets/s4-pytorch-nutshell/save-and-load/early-stopping.png)

Session 3 named early stopping: stop when the validation error turns up, and
return the model of the epoch with the lowest validation loss, not the last.
This network, trained on only 200 trips for 1,500 epochs, is best at epoch 266,
at a validation pinball loss of 1.25 minutes; by the last epoch it has risen to
1.88.

---

## The best epoch, in code

```python
import copy
best_val = float("inf")                  # before the epoch loop
```

```python
for val_loss in [1.35, 1.21, 1.24]:      # stand-in: three epochs
    if val_loss < best_val:              # validation improved
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
```

```python
model.load_state_dict(best_state)        # after the loop
```

The list stands in for lesson 7's validation step, one value per epoch; any
first value beats `float("inf")`. `best_val` ends at 1.21, and the last line
puts back the parameters of that epoch. Why `copy.deepcopy`: next slide.

---

## Why a deep copy

```python
alias = model.state_dict()                   # the model's own tensors
snap = copy.deepcopy(model.state_dict())     # new tensors, same values
with torch.no_grad():
    model[0].bias += 1.0                     # in place, as opt.step()
```

```python
torch.equal(alias["0.bias"], model[0].bias)      # True: it moved too
torch.equal(snap["0.bias"], model[0].bias)       # False: it did not
```

- The tensors of a `state_dict` share memory with the parameters, and
  `opt.step()` updates the parameters in place (lesson 4). Kept without a copy,
  the "best" changes with every step and ends as the last.
- Writing a file at once, `torch.save(model.state_dict(), "best.pt")`, also
  works: the file holds the values of that moment.

---

## A checkpoint holds what the prediction needs

$$
\hat{y} = f_\theta\left(\frac{x - \mu}{s}\right)
$$

A row is standardised with the training part's mean $\mu$ and standard
deviation $s$ (lesson 7), then goes through the network. A file meant for later
use carries every piece:

```python
X_raw = 50 * torch.rand(1000, 12)        # stand-in: raw training rows
ckpt = {"state_dict": model.state_dict(), "hidden": 64,
        "mean": X_raw.mean(dim=0), "std": X_raw.std(dim=0)}
torch.save(ckpt, "checkpoint.pt")
```

- `hidden` rebuilds $f$, `state_dict` is $\theta$, `mean` and `std` are $\mu$
  and $s$: float32 tensors of 12 values, the dtype the model expects.
- `torch.load` reads tensors, numbers, strings, lists and dicts by default, and
  refuses arbitrary Python objects. A NumPy array fails with `Weights only load
  failed`: from Session 2's `StandardScaler`, store
  `torch.tensor(scaler.mean_, dtype=torch.float32)`, and `scaler.scale_` alike.

---

## Reloading a checkpoint

```python
ckpt = torch.load("checkpoint.pt", map_location="cpu")
model = make_model(ckpt["hidden"])          # rebuild the architecture
model.load_state_dict(ckpt["state_dict"])   # then fill it with theta
```

- `map_location="cpu"` loads every tensor into CPU memory. Tensors are saved
  with their device: without it, a file written on a GPU fails on a machine
  without one, `Attempting to deserialize object on a CUDA device but
  torch.cuda.is_available() is False`. Move the model afterwards with
  `model.to(device)` (lesson 5).
- `model.load_state_dict(ckpt)` fails: the checkpoint's keys are `state_dict`,
  `hidden`, `mean` and `std`, not parameter names
  (`Missing key(s) in state_dict: "0.weight", …`).

---

## Predicting on new rows

```python
X_new_raw = 50 * torch.rand(5, 12)       # stand-in: 5 new raw rows
X_new = (X_new_raw - ckpt["mean"]) / ckpt["std"]   # the saved mu, s
model.eval()                             # evaluation mode (lesson 7)
```

```python
with torch.no_grad():                    # no graph (lesson 3)
    y_hat = model(X_new).squeeze(1).cpu().numpy()
y_hat.shape                              # (5,)
```

- The same four steps every time: standardise with the saved $\mu$ and $s$,
  evaluation mode, no graph, back to NumPy.
- `.squeeze(1)` turns the $(n, 1)$ output into $(n,)$. `.numpy()` reads CPU
  memory only (lesson 2), hence `.cpu()` first whenever the model ran on a GPU.
- `y_hat` is a NumPy array with one prediction per row, ready for pandas and a
  CSV file.

---

## Check yourself

1. Run this. What does it print?

   ```python
   m = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 1))
   print(list(m.state_dict()))
   ```

   **Answer.** `['0.weight', '0.bias', '2.weight', '2.bias']`. The ReLU, at
   position 1, has no parameters and no key.

2. Why `copy.deepcopy(model.state_dict())`, and not `model.state_dict()`, when
   keeping the best epoch?

   **Answer.** The tensors of `model.state_dict()` share memory with the
   parameters, which every `opt.step()` changes in place: the kept "best" would
   follow training and end equal to the last epoch.

3. A checkpoint written on a Colab GPU holds `hidden`, `state_dict`, `mean` and
   `std`. On a laptop without a GPU, which argument does `torch.load` need, and
   what must new rows go through before `model(...)`?

   **Answer.** `map_location="cpu"`. The rows are standardised with the saved
   statistics, `(X - ckpt["mean"]) / ckpt["std"]`, as the training rows were.
