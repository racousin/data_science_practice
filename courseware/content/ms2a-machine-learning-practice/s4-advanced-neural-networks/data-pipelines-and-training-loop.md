# Data Pipelines and the Training Loop

The model is usually not the slow part and almost never the broken part. The
pipeline that feeds it, and the loop that drives it, are both. This lesson is the
machinery between a directory of files and a gradient step: how a batch gets
built, which process builds it, which random numbers it uses, and in what order
the seven lines of a training step have to run.

Half of what follows exists because a default is wrong for you rather than wrong
in general — `num_workers=0`, `drop_last=False`, an unseeded shuffle. The other
half exists because the failure is silent: a duplicated random stream, a padding
token the loss averages over, a resume that quietly throws Adam's moments away.
None of them raise. All of them cost you score.

The worker-seeding numbers below are printed by the snippet next to them, and the
resume measurement is a real run — an MLP on 20 000 MNIST digits, checkpointed
and restarted two different ways.

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
one finished batch on a queue. A pinning thread copies it into page-locked host
memory, and `.to(device, non_blocking=True)` starts the transfer without waiting
for it.

Everything left of the orange panel is CPU work that can overlap with the
previous step's compute. That overlap is the entire point of the apparatus: the
orange box should be the only stage that ever waits. Whether it is, on your
machine, is a measurement rather than an opinion —
[Performance and Profiling](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/performance-and-profiling)
shows the one-line test.

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

Each worker process receives a **copy** of this object, so hold the data as NumPy
arrays, tensors or file paths. A Python list of ten million dicts becomes
`num_workers` copies even under `fork`, because CPython's refcounting writes to
every object header it touches and defeats copy-on-write. Arrays and tensors have
one header for millions of values, so they survive it.

Anything that cannot be pickled — a database connection, an open HDF5 handle —
must not exist when the workers start. Open it lazily, once per worker:

```python
    def __getitem__(self, i):
        if self._h5 is None:                        # None in __init__, opened per worker
            self._h5 = h5py.File(self.path, "r")
        ...
```

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

`num_workers=0` runs loading in the training process, so the device sits idle for
every millisecond of decoding. Start at `min(8, os.cpu_count())`; more workers
than cores is slower, not faster, because they compete with the threads doing the
compute.

`persistent_workers=True` matters more than it looks. Without it, every epoch
boundary tears down eight processes and starts eight more, each of which re-copies
the dataset object — a fixed cost per epoch that is invisible in the loss curve
and obvious in the wall clock.

`drop_last=True` whenever the model contains BatchNorm: a final batch of size 1
raises in `train()` mode, at the end of epoch 1, after you left for lunch. It also
makes the step count identical across runs, which is what a per-step scheduler and
a step-indexed log both assume.

On the **validation** loader, `drop_last` stays `False` and `shuffle` stays
`False`. Dropping there silently changes the denominator of your metric, and
shuffling there changes nothing except your ability to compare two epochs' worst
mistakes. The batch can be much larger than the training one — no activations are
kept for a backward pass that never happens.

---

## Every worker is a copy — including its randomness

When a worker starts, PyTorch seeds three global generators for it: `torch`'s,
Python's `random`, and NumPy's `np.random`. Each gets a distinct per-worker seed
derived from a base seed drawn from the main process's torch RNG, so passing
`generator=g` makes the shuffle *and* every worker's global randomness
reproducible. Most of the `worker_init_fn` advice online predates this and
re-seeds NumPy by hand; it has been redundant since PyTorch 1.9.

What PyTorch cannot seed is a generator object **you** made. `np.random.default_rng(0)`
in `__init__` is created in the parent, before any worker exists. `fork` copies the
parent's memory and `spawn` pickles the dataset, state included — either way every
worker holds the same generator at the same position:

```python
# save as worker_rng.py and run it as a file: python worker_rng.py
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, get_worker_info

class Noisy(Dataset):
    def __init__(self):
        self.rng = np.random.default_rng(0)         # created once, in the parent process
    def __len__(self): return 8
    def __getitem__(self, i):
        w = get_worker_info().id
        return torch.tensor([w, self.rng.random(), np.random.rand(), torch.rand(1).item()])

if __name__ == "__main__":
    g = torch.Generator(); g.manual_seed(0)
    rows = torch.cat([b for b in DataLoader(Noisy(), batch_size=1, num_workers=4,
                                            generator=g)])
    print(torch.stack([rows[rows[:, 0] == w][0, 1:] for w in range(4)]).numpy().round(4))
```

```text
[[0.637  0.8372 0.7821]
 [0.637  0.6703 0.6938]
 [0.637  0.2392 0.654 ]
 [0.637  0.9734 0.1343]]
```

![First random value drawn by each of four workers, with the default set-up and with a worker_init_fn that re-creates the generator](assets/nn/worker-seeding-duplicated-rng.png)

One column per generator, one row per worker. The global NumPy and torch streams
are distinct across workers, exactly as documented. The middle column — the
`Generator` built in `__init__` — is **0.6370 in all four workers**, every epoch,
for the whole run. Re-create it inside the worker instead:

```python
def seed_worker(worker_id: int) -> None:
    info = get_worker_info()                        # this runs inside the worker
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)

DataLoader(ds, ..., num_workers=4, generator=g, worker_init_fn=seed_worker)
```

`torch.initial_seed()` called inside a worker returns that worker's own seed, so
the four generators are distinct across workers and identical across runs: the
middle column becomes 0.8308, 0.6470, 0.7702, 0.1197 and stays those four numbers
every time. If you would rather not write a `worker_init_fn`, do not hold a
generator at all — build it in `__getitem__` from `get_worker_info()`, or use the
global `np.random` and `torch` calls, which are already handled for you.

The symptom when you get this wrong is not a crash. It is `num_workers` copies of
the same augmentation applied to different images, an effective dataset smaller
than you believe, and a validation score that moves when you change
`num_workers`. **If a score changes when you change the number of processes, this
is why.**

In the main process, one function covers the rest:

```python
import random, numpy as np, torch

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                         # seeds every device, CUDA included
```

---

## collate_fn: what becomes a batch

The default collate turns a list of B samples into a batch: it stacks tensors,
recurses into tuples, lists, dicts and namedtuples, and promotes plain numbers. It
requires every sample to have the same shape — which is exactly the assumption
variable-length data breaks.

```python
from functools import partial
from torch.nn.utils.rnn import pad_sequence

def collate(batch, pad_id=0):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    return padded, torch.stack(ys), lengths

DataLoader(ds, batch_size=4, collate_fn=partial(collate, pad_id=0))
```

![Four variable-length samples padded into one batch tensor, with the lengths returned alongside](assets/nn/collate-variable-length-padding.png)

Return the lengths, or a mask. The padding is real data to the model: a
mean-pooling layer that does not know where the sequence ends averages over the
grey cells, so the representation of the length-2 sample depends on the longest
sample that happened to land in its batch — which changes every epoch, because the
shuffle changes. The loss needs the same information (`ignore_index=pad_id` on
`CrossEntropyLoss`), and so does the metric.

`collate_fn` runs **in the worker**, which makes it the right place for work that
only makes sense per batch: sorting by length, building the mask, tokenising to a
common width. Use `functools.partial` rather than a lambda — under `spawn` the
function has to be picklable.

---

## Augmentation, and what survives a permutation

Augmentation goes in `__getitem__`, so the workers parallelise it, and on the
**training split only**:

```python
from torchvision.transforms import v2

train_tf = v2.Compose([v2.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize([mean], [std])])
eval_tf  = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize([mean], [std])])
```

Two rules carried over from Session 2. Augment train only — augmenting validation
makes the score noisy and optimistic. And compute `mean` and `std` on train only:
taking them from the full dataset is leakage, exactly as fitting a scaler on
everything is.

![MNIST digits under affine augmentation, random erasing and pixel noise, and the same digits after a fixed pixel permutation](assets/nn/augmentation-mnist-and-permutation.png)

The top row is the originals and the next three are ordinary MNIST augmentation;
the last row is the same eight digits after a fixed permutation of the 784 pixel
positions. Read the last two rows together, because the pair is the whole
argument.

`RandomAffine` and `RandomErasing` are **spatial**: they assume pixel $(r, c)$ is
next to pixel $(r, c{+}1)$. After a fixed permutation that assumption is gone, and
rotating the permuted image produces a picture no un-permuted digit could ever map
to — you have augmented your training set with samples from outside the test
distribution. Additive noise and pixel dropout are different: they act on each
pixel independently, so they **commute** with the permutation. Applying them
before or after gives the same distribution, which is what makes them the only
augmentation in the figure that still means anything once the pixels are shuffled.

```mlarena:challenge id=8
```

That is the data you meet in Lab 4. `train()` receives `X_train` as a
`(60000, 28, 28)` uint8 array with the pixel positions permuted and mild noise
added, and `y_train` as `(60000, 1)` int64 with the label meanings permuted too —
real arrays, already in memory. There is no file to decode, so there is no
`Dataset` and no `DataLoader` worth building: convert once, normalise with
statistics computed from `X_train` itself rather than constants copied from an
MNIST tutorial, and index the tensor.
[Performance and Profiling](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/performance-and-profiling)
has the measurement that says how much that is worth on three cores.

---

## One step, in order

```python
model.train()
for xb, yb in train_dl:
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    opt.zero_grad(set_to_none=True)
    loss = criterion(model(xb), yb)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # log gn
    opt.step()
    sched.step()                                    # per-batch schedules only
    global_step += 1
```

![The order of one training step, and the four things that happen once per epoch](assets/nn/training-step-order.png)

The order is not a matter of taste:

- **`.grad` accumulates.** That is not a quirk to work around — it is what makes
  gradient accumulation and multi-loss backward passes possible — but it means
  `zero_grad` has to happen somewhere. Put it at the top, not after `step()`: the
  loop then starts from a known state, and a `continue` that skips a bad batch
  cannot leave a stale gradient behind.
- **`set_to_none=True`** has been the default since 2.0. Gradients become `None`
  rather than zero, which is cheaper and diagnostic: a parameter whose `grad is
  None` after `backward` is one your forward pass never reached.
- **Clipping sits between `backward` and `step`.** Before `backward` there is
  nothing to clip; after `step` it is too late. `clip_grad_norm_` returns the total
  norm *before* clipping, so log it —
  [Monitoring and Debugging](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging)
  explains what its shape tells you.
- **`sched.step()` after `opt.step()`**, once per batch for `OneCycleLR` and
  warmup, once per epoch for everything else. Which is which, and what stepping it
  at the wrong frequency looks like, is
  [Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules).
- **`non_blocking=True`** does something only when the source tensor is pinned and
  the destination is CUDA. On its own it is decoration.

---

## The validation pass

```python
@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n
```

`model.eval()` and `torch.inference_mode()` are two independent mechanisms and
you need both.

| | `model.eval()` | `torch.inference_mode()` |
|---|---|---|
| Dropout | switched off | untouched |
| BatchNorm | uses its running statistics | untouched |
| Autograd graph | still built | not built |
| Forgetting it | a noisy, pessimistic score — and BatchNorm updates its running statistics from validation data, which is leakage that survives into the next epoch | wasted memory on the largest batch you run, and an out-of-memory error that looks like a model problem |

`inference_mode` is `no_grad` plus the removal of version counting, so it is
slightly faster and correct for anything you will never backpropagate through.
And note the weighting: `loss.item() * len(yb)` divided by `n`, not the mean of
the per-batch means, which is wrong the moment the last batch is short. Put
`model.train()` back at the top of the next epoch — the figure's per-epoch row
ends with it for a reason.

---

## Gradient accumulation

Simulate a batch that does not fit in memory by summing the gradients of several
that do.

```python
accum_steps = 4
opt.zero_grad(set_to_none=True)
for i, (xb, yb) in enumerate(train_dl):
    loss = criterion(model(xb.to(device)), yb.to(device)) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0 or (i + 1) == len(train_dl):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()                                # once per optimizer step
        opt.zero_grad(set_to_none=True)
```

![One batch of 256 versus four accumulated micro-batches of 64 reaching the same optimizer step](assets/nn/gradient-accumulation-timeline.png)

Four things go wrong here, in descending order of how often:

1. **The missing `/ accum_steps`.** Your loss is a mean over the micro-batch, so
   summing four of them gives a gradient exactly four times the one a real batch of
   256 would produce. That multiplies your effective learning rate by four, and the
   symptom is a divergence you will spend an afternoon blaming on the schedule.
2. **Clipping every micro-batch.** A partial gradient has a smaller norm than the
   finished one, so a per-micro-batch clip either does nothing or clips something
   that is not the quantity you meant to bound. Clip once, on the step.
3. **`sched.step()` inside the micro-batch loop.** The schedule then runs four
   times too fast. `OneCycleLR` needs
   `total_steps = epochs * len(train_dl) // accum_steps`, not `epochs * len(train_dl)`.
4. **The tail.** If `len(train_dl)` is not a multiple of `accum_steps`, the last
   few micro-batches accumulate a gradient that is never stepped and is then
   discarded by the next epoch's `zero_grad`. The `or (i + 1) == len(train_dl)`
   clause spends it, at a slightly smaller effective batch.

It is still not identical to a real large batch. BatchNorm computes its statistics
per micro-batch, so four passes of 64 normalise like four batches of 64, not like
one of 256; `LayerNorm` and `GroupNorm` have no such problem. And it buys memory
with time — four times the steps for a quarter of the peak activation memory. The
memory arithmetic that tells you whether that trade is available is in
[Performance and Profiling](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/performance-and-profiling).

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
}, "ckpt.pt.tmp")
os.replace("ckpt.pt.tmp", "ckpt.pt")                # atomic: never a half-written best model
```

![The eight entries of a resumable checkpoint and what breaks when each one is missing](assets/nn/resumable-checkpoint-contents.png)

The model weights alone are not a checkpoint. Read the right-hand column as a
checklist: each line is a failure that produces a *plausible* curve rather than an
error.

Two practical notes on that dict. Write to a temporary path and `os.replace` it —
a process killed during `torch.save` otherwise leaves a truncated file where your
best model used to be. And note that the code above keeps the `rng` block to
torch's own states, which are byte tensors, while the figure's row also lists
NumPy's and Python's: since torch 2.6 `torch.load` defaults to
`weights_only=True` and **refuses** to unpickle a NumPy RNG state or a pickled
`argparse.Namespace`. Keep those two streams by allowlisting them with
`torch.serialization.add_safe_globals(...)` at load time, or write them to a JSON
sidecar. Save on improvement, not every epoch.

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

![Training loss after resuming from the same checkpoint with the full state and with the weights only, against an uninterrupted reference run](assets/nn/resume-full-state-vs-weights-only.png)

An MLP on 20 000 MNIST digits, AdamW at 3e-3 with a per-step cosine over six
epochs, checkpointed at the end of epoch 3 and resumed twice. The blue curve —
model, optimizer, scheduler and torch RNG state all restored — is not merely close
to the uninterrupted reference. It is bit-identical to it: `max |diff| = 0.00e+00`
over all 1 872 steps, which is why the dashed line is invisible underneath it.

The orange curve restored the weights and nothing else. Over the 100 steps
following the resume its mean loss is **0.0850** against **0.0361** for the other
two, and it never fully catches up — three epochs later the final 100-step mean is
**0.0205** against **0.0088**. Two causes, both visible in the shape. A fresh
`AdamW` starts with zero first and second moments, so with bias correction at
$t = 1$ its first steps are effectively sign-like at full learning rate, and it
takes a few hundred steps to rebuild estimates it already had. A fresh
`CosineAnnealingLR` restarts at its maximum, so the run spends its last three
epochs at a learning rate the schedule had already decided it should be past.
Neither raises. The loss curve is the only place either one shows.

Three more rules for the load side:

- **`strict=False` is not an error suppressor.** It exists to load a partial state
  dict deliberately — a pretrained backbone without its head. Using it to silence a
  key mismatch loads whatever happens to match and trains the rest from scratch,
  silently. `load_state_dict` returns `missing_keys` and `unexpected_keys`; print
  them when you pass `strict=False`, and find out why otherwise.
- **`map_location="cpu"`, then `.to(device)`.** Loading straight onto `cuda:0`
  fails on a machine that has none, and ties the file to the GPU it was saved from.
- **Never `weights_only=False` on a checkpoint you did not produce.** That path is
  `pickle.load`, and `pickle.load` is arbitrary code execution.

---

## Reproducibility you can actually claim

| Claim | What it takes | Realistic |
|---|---|---|
| Same machine, same code, run twice | `seed_everything`, `generator=`, `worker_init_fn=`, a fixed `num_workers` | yes |
| A resume identical to the uninterrupted run | the above, plus the RNG state in the checkpoint | yes — the figure above is the proof |
| Same numbers on a different GPU, cuDNN or torch version | nothing you can do | no |

```python
torch.use_deterministic_algorithms(True)   # raises on any op with no deterministic kernel
torch.backends.cudnn.benchmark = False     # stop the convolution algorithm auto-tuner
# and CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment, for CUDA
```

Both cost real speed, and the first one turns "my result is not reproducible" into
a stack trace naming the operation responsible, which is the useful failure mode.

`num_workers` is part of your seed. Changing it changes which worker draws which
sample, and therefore the order in which the per-worker RNG streams are consumed:
the shuffle is unchanged, the augmentation is not. Record it next to the seed.

The seed itself is configuration, not a constant. Pass it in, put it in `config`,
store it in the checkpoint, and report a mean over three seeds. A single-seed
comparison of two architectures is one draw from each of two distributions whose
spread you did not measure — it is not evidence.
[Monitoring and Debugging](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging)
covers what W&B records for you without being asked: the command line, the git
commit, a patch of your uncommitted changes and the environment.

> Every default in this lesson — `num_workers=0`, `drop_last=False`, an unseeded
> shuffle, a `strict=False` that makes an error go away — is a decision somebody
> made for you, in a context that was not yours. Make them yourself, log them, and
> put them in the checkpoint.

---

## Check yourself

1. Which work belongs in `__init__` and which in `__getitem__`? A colleague's
   `__init__` builds a Python list of ten million dicts and the job runs out of
   memory at `num_workers=8` but not at `num_workers=0`. Explain both facts.

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   torch.manual_seed(0)
   x, y = torch.randn(256, 16), torch.randn(256, 1)
   net, crit = nn.Linear(16, 1), nn.MSELoss()

   net.zero_grad(); crit(net(x), y).backward()
   big = net.weight.grad.clone()                    # one batch of 256

   net.zero_grad()
   for i in range(4):                               # four micro-batches of 64
       sl = slice(i * 64, (i + 1) * 64)
       (crit(net(x[sl]), y[sl]) / 4).backward()

   print(torch.allclose(big, net.weight.grad, atol=1e-6))   # -> True
   ```

   Now delete the `/ 4` and print `(net.weight.grad / big).mean()` instead. Which
   number comes out, and what has it just done to your effective learning rate?

3. Your validation accuracy moves by half a point when you change `num_workers`
   from 4 to 8, with the seed unchanged. What is happening, and which line of
   `Dataset.__init__` would you look at first?

4. A resumed run's loss jumps from 0.07 to 0.10 and takes six hundred steps to
   come back. Name the two pieces of state the checkpoint is missing, and say
   which of them you would blame if the loss had instead stayed high for the whole
   remaining run.

5. You accumulate over four micro-batches and pass
   `total_steps=epochs * len(train_dl)` to `OneCycleLR`. What does the learning
   rate do over the run, and what should `total_steps` have been?

6. Your `collate_fn` pads to the longest sequence in the batch and returns only
   `(padded, y)`. The score is fine when the loader is sorted by length and worse
   with `shuffle=True`. Why, and what does the fix return?
