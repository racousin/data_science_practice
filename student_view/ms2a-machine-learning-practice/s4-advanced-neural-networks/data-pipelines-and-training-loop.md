# Data Pipelines and the Training Loop

The model is usually not the slow part and almost never the broken part. The
pipeline that feeds it, and the loop that drives it, are both.

<!-- notes: 30 minutes. The 12h module built a working loop; this one hardens it.
The gradient-accumulation and resume slides are the two they have not seen. -->

---

## Dataset: the contract

```python
class TabularDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_parquet(path)
        self.X = torch.tensor(df.drop(columns="y").values, dtype=torch.float32)
        self.y = torch.tensor(df["y"].values, dtype=torch.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
```

`__getitem__` does **per-sample** work: read one file, decode one image, apply
one transform. Anything per-*dataset* — reading a Parquet, opening a connection,
computing normalisation statistics — belongs in `__init__`. Use `IterableDataset`
only when the data cannot be indexed at all.

Each worker process receives a *copy* of this object, so hold the data as NumPy
arrays, tensors or file paths: a Python list of ten million dicts becomes ten
copies, because refcounting defeats copy-on-write.

---

## DataLoader

```python
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                      num_workers=8, pin_memory=True,
                      persistent_workers=True, drop_last=True)
```

| Argument | What it buys | Default |
|---|---|---|
| `num_workers` | overlaps loading with compute | 0 — everything serial |
| `pin_memory` | page-locked host buffer, faster to GPU | `False` |
| `persistent_workers` | no process restart per epoch | `False` |
| `prefetch_factor` | batches queued per worker | 2 |
| `drop_last` | uniform batch size | `False` |

`num_workers=0` runs loading in the training process, so the GPU sits idle for
every millisecond of decoding. Start at `min(8, os.cpu_count())`; more workers
than cores is slower, not faster. A connection or file handle opened in
`__init__` cannot be pickled to a worker — open it lazily, once per worker.

`drop_last=True` whenever the model contains BatchNorm: a final batch of size 1
raises in `train()` mode, at the end of epoch 1, after you left for lunch.

---

## collate_fn

The default collate stacks samples, which requires identical shapes.
Variable-length sequences need an explicit one:

```python
def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    return pad_sequence(xs, batch_first=True), torch.stack(ys), lengths
```

Return the lengths. A model that does not know where the padding starts averages
over it, so the score degrades with your batch composition — which changes every
epoch.

---

## Transforms and augmentation

Augmentation goes in `__getitem__`, so the workers parallelise it, and **only on
the training split**:

```python
train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize(MEAN, STD)])
```

Two rules carried over from Session 2. Augment train only — augmenting validation
makes the score noisy and optimistic. And compute `MEAN` and `STD` on train only:
from the full dataset it is leakage, exactly as fitting a scaler on everything
is.

---

## Seeding

```python
def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

g = torch.Generator(); g.manual_seed(seed)
dl = DataLoader(ds, shuffle=True, generator=g, worker_init_fn=seed_worker)
```

The `generator` seeds the shuffle; `worker_init_fn` seeds each worker's NumPy and
Python RNG, which are otherwise duplicated — every worker applying the same
"random" crop.

The seed is configuration, not a constant: pass it in, log it, and report a mean
over three seeds. A single-seed comparison of two architectures is not evidence.

---

## The training step, precisely

```python
model.train()
for xb, yb in train_dl:
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    loss = criterion(model(xb), yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

`set_to_none=True` (the default since 2.0) leaves gradients as `None` rather than
zeros — faster, and it makes an unused branch detectable: a parameter whose
`grad is None` after `backward` is one your forward pass never reached.
`non_blocking=True` pairs with `pin_memory=True` and does nothing without it.

---

## The validation step

```python
@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    return sum(criterion(model(xb.to(device)), yb.to(device)).item() * len(yb)
               for xb, yb in loader) / len(loader.dataset)
```

`torch.inference_mode()` is `no_grad` plus the removal of version counting —
slightly faster, and correct for anything you will not backpropagate through. It
is a separate mechanism from `eval()`, which switches dropout and BatchNorm.
Forgetting either is silent; forgetting both is the classic "validation is worse
than training and I do not know why".

---

## Gradient accumulation

Simulate a batch you cannot fit:

```python
for i, (xb, yb) in enumerate(train_dl):
    loss = criterion(model(xb), yb) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Divide the loss by `accum_steps`. Without it the accumulated gradient is
`accum_steps` times too large, multiplying your effective learning rate by the
same factor — and the symptom is divergence you will blame on the schedule. It is
still not equal to a real large batch: BatchNorm computes per micro-batch.

---

## What a resumable checkpoint contains

```python
torch.save({
    "epoch": epoch, "global_step": step,
    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
    "best_val": best_val, "seed": seed, "git_sha": sha,
}, "ckpt.pt")
```

The model weights alone are not a checkpoint. Resuming without the optimizer
state discards Adam's moments and gives a visible loss spike; resuming without
the scheduler restarts the learning rate at its maximum.

---

## Resuming

```python
ck = torch.load("ckpt.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ck["model"])          # strict=True — leave it alone
optimizer.load_state_dict(ck["optimizer"])
scheduler.load_state_dict(ck["scheduler"])
start_epoch = ck["epoch"] + 1
```

`strict=False` exists to load a partial state dict deliberately — a backbone
without its head. Using it to silence a key mismatch loads whatever happens to
match and trains the rest from scratch, silently. If the keys do not match, find
out why. `weights_only=True` refuses to unpickle arbitrary objects; use it for
anything you did not produce yourself.

> Every default in this lesson — `num_workers=0`, `drop_last=False`, an unseeded
> shuffle, a `strict=False` that makes an error go away — is a decision someone
> made for you. Make them yourself, log them, and own them.
