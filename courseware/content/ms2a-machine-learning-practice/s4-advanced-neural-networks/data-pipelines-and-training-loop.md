# Data Pipelines and the Training Loop

The model is usually not the slow part and almost never the broken part. The
pipeline that feeds it, and the loop that drives it, are both. This lesson is the
machinery between a directory of files and a gradient step: how a batch gets
built, which process builds it, which random numbers it uses, and in what order
the seven lines of a training step have to run.


<!-- notes: 30 minutes. The duplicated-RNG demo is the one to run live: it takes
ten seconds and it overturns advice they will find in every blog post. Correct
the old folklore explicitly — PyTorch has seeded numpy and random per worker
since 1.9, and the bug that remains is a Generator object held on the Dataset.
Gradient accumulation and resume are the two things the 12h module never showed
them. The augmentation figure is the bridge to Lab 4: spend a minute on why
RandomAffine is meaningless after a pixel permutation. -->

---

## The pipeline, end to end

![How a DataLoader turns a Dataset and a Sampler into batches on the device](assets/nn/dataloader-pipeline.png)

The main process holds the `Dataset` and the `Sampler`. The sampler produces
indices; each worker process is handed a whole batch's worth of them, calls
`__getitem__` once per index, passes the resulting list to `collate_fn`, and puts
one finished batch on a queue.

---

## Dataset: the contract

A map-style dataset is two methods. That is the whole interface.

```python
import numpy as np, torch
from torch.utils.data import Dataset

class DigitsDataset(Dataset):
    def __init__(self, path, transform=None):
        blob = np.load(path)                        # per-dataset work: once
        self.X = torch.from_numpy(blob["X"])        # uint8, (N, 28, 28)
        self.y = torch.from_numpy(blob["y"]).long()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):                       # per-sample work: N times per epoch
        x = self.X[i].float().div(255.0)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]
```

`__getitem__` does **per-sample** work: read one file, decode one image, apply one
transform. Anything per-*dataset* — reading a Parquet, opening a connection,
computing normalisation statistics — belongs in `__init__`. Reach for
`IterableDataset` only when the data genuinely cannot be indexed: a socket, a
stream, a file you can only read forwards.


---

## DataLoader: the arguments that are decisions

```python
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True,
                      num_workers=8, persistent_workers=True, prefetch_factor=4,
                      pin_memory=True, generator=g, worker_init_fn=seed_worker)

val_dl = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)
```

| Argument | What it buys | Default |
|---|---|---|
| `num_workers` | loading overlaps compute, in separate processes | `0` — loading blocks the loop |
| `pin_memory` | page-locked host buffer, so the copy can be asynchronous | `False` |
| `persistent_workers` | workers survive the epoch boundary | `False` |
| `prefetch_factor` | batches queued ahead, per worker | `None`, which means 2 once workers exist |
| `drop_last` | every batch the same size | `False` |
| `generator` | the shuffle order is reproducible | `None` — a fresh seed per run |
| `worker_init_fn` | your own per-worker setup | `None` |

---


---

## What a resumable checkpoint contains

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "scheduler": sched.state_dict(),
    "scaler": scaler.state_dict(),
    "epoch": epoch, "global_step": global_step, "best_val": best_val,
    "config": vars(args), "seed": seed, "wandb_run_id": run.id,
    "rng": {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()},
}, "ckpt.pt")

```

![The eight entries of a resumable checkpoint and what breaks when each one is missing](assets/nn/resumable-checkpoint-contents.png)


---

## Resuming, measured

```python
ck = torch.load("ckpt.pt", map_location="cpu")      # weights_only=True by default since 2.6
model.load_state_dict(ck["model"])                  # strict=True — leave it alone
opt.load_state_dict(ck["optimizer"])
sched.load_state_dict(ck["scheduler"])
torch.set_rng_state(ck["rng"]["torch"])
start_epoch = ck["epoch"] + 1
model.to(device)
```
