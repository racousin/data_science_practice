# Monitoring and Debugging

A training run you are not logging is a forty-minute experiment that produces one
number. Instrument it with Weights & Biases and the same run tells you what to
change, keeps every run you ever made next to every other, and writes down the
config you will otherwise have forgotten by Friday. The first half of this lesson is
the tool; the second half is the debugging discipline that no tool replaces.

Every figure marked *from a real run* below comes from small MNIST runs logged to
W&B while writing this lesson; the panels are redrawn with matplotlib so that they
read at print size.

<!-- notes: 35 minutes. Have a W&B project open with the three learning-rate runs
before the lesson starts and switch the x-axis to epoch live. The overfit-one-batch
slide is the one they must leave with; the sweep section is a demo, not an exercise
— Lab 4 does it. Students without an account use wandb.login(anonymous="allow")
and claim the runs later. -->

---

## What to log, and how often

| Quantity | Frequency | Why |
|---|---|---|
| `train/loss` | every ~50 steps | shape of the descent |
| `val/loss` and the real metric | every epoch | the only thing you optimise for |
| `lr` | with every train log | catches a mis-stepped scheduler |
| `grad_norm` | every ~50 steps | vanishing, exploding, clipping |
| weight and gradient histograms | every 200–1000 steps | dead layers, drift |
| samples per second | every epoch | regressions in the pipeline |
| a confusion matrix, worst mistakes, per-class table | once per epoch, or at the end | what the number hides |

Scalars are cheap: a `run.log` call costs microseconds and the data is buffered
and uploaded in the background. Histograms are not — a 10 M-parameter tensor
summarised every step dominates the run time — which is why `wandb.watch` takes a
`log_freq`. The keys use `section/name`: W&B groups `train/*` and `val/*` into
separate sections of the workspace.

---

## One run, instrumented

```python
import wandb, torch

run = wandb.init(
    project="ms2a-s4",                          # one project per problem
    name=f"mlp-w{width}-lr{lr:g}-s{seed}",      # say what you varied
    config=dict(lr=lr, width=width, batch_size=128, epochs=20, seed=seed),
    tags=["mlp", "baseline"],
)
cfg = run.config                                # read hyper-parameters back from here
run.define_metric("val/acc", summary="max")     # the run summary keeps the best, not the last
run.watch(model, log="all", log_freq=200)       # gradient + parameter histograms every 200 steps

step = 0
for epoch in range(cfg.epochs):
    for xb, yb in train_dl:
        loss = criterion(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # returns the norm before clipping
        opt.step(); sched.step(); step += 1
        if step % 50 == 0:
            run.log({"train/loss": loss.item(), "grad_norm": grad_norm.item(),
                     "lr": opt.param_groups[0]["lr"], "epoch": epoch}, step=step)
    val_loss, val_acc = evaluate(model, val_dl)
    run.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch}, step=step)
run.finish()
```

`config` is the contract: everything you might vary goes in it, and nothing else
should be a hard-coded constant in the file. `name` is for humans — you will have
forty runs by Friday and `exp3_final_v2` identifies none of them; W&B also gives every
run an auto-generated name and a unique id. `step` must never go backwards: W&B
warns and drops a `log` call whose step is smaller than the last one it saw. Logging
the epoch-end validation at the *same* `step` as the last training log merges the
two rows, and since `epoch` is itself a logged key you can pick it as the x-axis of
any panel.

---

## The plot that tells you nothing

![Training loss alone over sixty epochs, from a real run](assets/nn/train-loss-alone.png)

Sixty epochs on 3 000 MNIST digits, no regularisation. The training loss falls
smoothly by four orders of magnitude and is still falling at the end. Is this the
best model you have ever trained, or the worst? The plot cannot say, because the
quantity that answers it is not on it.

---

## Plot both, always

![Training and validation loss on the same panel, same run](assets/nn/train-and-val-loss.png)

Same run, with `val/loss` on the same panel. Validation bottoms out at epoch 5 at
0.248 and drifts up to 0.38 while training keeps descending: the best checkpoint is
at epoch 5, and 55 epochs of compute bought a worse model. W&B draws one panel per
key by default; open the `train/loss` panel's settings, add `val/loss` as a second Y
and the layout is saved for every future run of the project.

| Shape | Diagnosis | Action |
|---|---|---|
| val flat or rising, train falling | overfitting | stop earlier, regularise, more data |
| both flat and high | underfitting or LR too low | capacity, LR |
| loss oscillates, no trend | LR too high | lower it, clip |
| smooth then a vertical spike | one bad batch, or `nan` | inspect that batch |
| staircase drops | your scheduler | expected |
| val below train | dropout or augmentation is on in train | expected, not a bug |

---

## Three shapes you will meet

![Loss flat at chance, loss exploding to NaN, and train falling while val rises — three real runs](assets/nn/three-failure-shapes.png)

All three are real. **A** is a network that never learns: the loss sits at
$\ln 10 = 2.303$, the cross-entropy of guessing uniformly over ten classes. The bug
is two lines apart — `opt = AdamW(MLP().parameters())` was built on a *fresh*
model instance, so the network being trained is not the one the optimizer updates.
Whenever the loss is pinned at $\ln(\text{classes})$, suspect the plumbing, not the
architecture. **B** is raw 0–255 pixels fed to plain SGD with `lr=5`: the loss is
$10^{24}$ at step 5 and `nan` at step 6. **C** is the overfitting run above.

Knowing these three shapes by sight is most of what "reading a loss curve" means.

---

## Overfit one batch

Before any real run, prove the plumbing works. Take 32 samples and fit them.

```python
xb, yb = next(iter(train_dl))
for _ in range(200):
    loss = criterion(model(xb), yb)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(loss.item())      # must be ~0
```

![Loss on a single batch of 32 over 200 steps, with dropout off and with dropout left on](assets/nn/overfit-one-batch.png)

Turn augmentation, dropout and weight decay off first — each puts a floor under the
achievable loss. In the figure the same network reaches $9 \times 10^{-6}$ with
dropout off and stalls, noisily, around $6 \times 10^{-3}$ with `Dropout(0.5)` left
on: still "fitting", but you can no longer tell a healthy model from a broken one.
If a network cannot memorise 32 examples it will not learn 60 000. Twenty seconds of
runtime, and the first thing to run after any architecture change — including the
`train()` you write for the challenge in Lab 4.

---

## Reading the failure

| What happens | What it means |
|---|---|
| loss does not move at all | optimizer has the wrong parameters, or `lr=0` |
| loss moves then stops high | graph broken — a `.detach()`, a NumPy round-trip |
| loss falls to a floor near chance | inputs and labels are misaligned in `__getitem__` |
| loss falls slowly but steadily | fine — raise the LR and rerun |
| loss becomes `nan` | initialization, LR, or unnormalised inputs |

Dwell on the third row: a dataset returning `self.y[i+1]` for `self.X[i]` trains,
converges and produces a plausible score. Overfitting one batch is how you find it —
32 misaligned pairs can still be memorised, but the loss on them stops falling long
before $10^{-3}$ because the network is memorising noise.

---

## Histograms: what `wandb.watch` shows

`run.watch(model, log="all", log_freq=200)` registers hooks on every parameter and,
every 200 backward passes, logs a histogram of each weight tensor and of its
gradient under `parameters/<name>` and `gradients/<name>`. In the workspace each
one is a panel with step on the x-axis and the distribution as a colour column;
below, twelve epochs of one layer's gradient distribution are redrawn as a ridge.

![First-layer gradient distributions across twelve epochs for a healthy ReLU MLP and for a six-layer sigmoid MLP, from real runs](assets/nn/gradient-histograms-across-epochs.png)

Left, a healthy layer: centred on zero, standard deviation shrinking slowly from
$1.3 \times 10^{-3}$ to $3.8 \times 10^{-4}$ as the loss falls. Right, the first
layer of a six-layer sigmoid network — the histogram *looks* the same until you read
the axis: $10^{-7}$, four orders of magnitude smaller, and not changing. That
network sat at chance — 10 to 11% accuracy — for all twelve epochs; the histogram
said why before the accuracy did. A distribution that collapses to a spike at exactly zero is a dead
layer (ReLU units that never fire again); one that widens without bound is heading
for `nan`. The first calls for a lower learning rate, then a `LeakyReLU` or
`GELU`, which keep a gradient on the negative side; the second for a lower
learning rate and `clip_grad_norm_`, as in the loop above.

---

## Beyond scalars: images and tables

Accuracy is one number; the mistakes are where the information is. Log them once per
epoch, or once at the end.

```python
fig = plot_confusion_matrix(cm)                       # any matplotlib figure
run.log({"val/confusion": wandb.Image(fig),
         "val/worst": [wandb.Image(img, caption=f"true {t} → pred {p} ({conf:.2f})")
                       for img, t, p, conf in worst_mistakes]}, step=step)

table = wandb.Table(columns=["class", "precision", "recall", "support"])
for c in range(10):
    table.add_data(str(c), prec[c], rec[c], int(support[c]))
run.log({"val/per_class": table}, step=step)
```

![A confusion matrix and the eight most confident mistakes of an MLP on the MNIST test set, from a real run](assets/nn/confusion-matrix-and-worst-mistakes.png)

The matrix says the 97.3% model's errors are not uniform: 8 → 3 and 9 → 3 account
for 36 of the 273 mistakes, and class 3 attracts errors from everywhere. The
gallery says something the matrix cannot: several of the most confident errors
($p > 0.9999$) are label noise or unreadable digits, so the ceiling on this test set
is below 100%. A `wandb.Table` is sortable and filterable in the UI, and a list of
`wandb.Image` under one key renders as a gallery with a step slider.

---

## Comparing runs

![Three learning rates trained from the same seed: training loss and validation accuracy per epoch, from real runs](assets/nn/three-learning-rates.png)

Three runs, same seed, `lr` ∈ {1e-4, 1e-3, 1e-2}. Because all three are in one
project, W&B overlays them on every panel by default and the runs table lists their
`config` columns next to their summaries. `define_metric("val/acc", summary="max")`
is what makes the table honest: the `val/acc` column shows each run's *best*
accuracy — 0.958, 0.973, 0.955 — not whatever the last epoch happened to be.

Three habits pay for themselves:

- **Filter and sort.** The runs table filters on any config or summary key
  (`config.width = 256`, `summary.val/acc > 0.97`, tags) and sorts on any column.
- **Group.** `wandb.init(group="lr-sweep")` — or *Group by* `config.lr` in the
  table — collapses runs with the same value into one line with a min–max band. Three
  seeds per setting, grouped, is the difference between a result and a lucky draw.
- **Parallel coordinates.** One axis per hyper-parameter and one for the metric, one
  line per run, coloured by the metric. Twenty runs read at a glance.

![The parallel-coordinates panel: one axis per hyper-parameter, one line per run, coloured by accuracy (W&B documentation)](assets/nn/wandb-parallel-coordinates.png)

In the panel above the high-accuracy runs are the light lines: they all pass through
low `learning_rate` values, and `dropout` barely matters. That is a conclusion no
single loss curve could have given you.

---

## Sweeps

When the grid is small enough to enumerate and large enough to be tedious, hand it to
W&B. A sweep is a search configuration plus one or more *agents* that pull the next
setting, run your training function, and report back.

```python
sweep_cfg = {
    "method": "bayes",                                  # or "grid", "random"
    "metric": {"name": "val/acc", "goal": "maximize"},
    "parameters": {
        "lr":      {"distribution": "log_uniform_values", "min": 1e-4, "max": 3e-2},
        "width":   {"values": [128, 256, 512]},
        "dropout": {"values": [0.0, 0.2, 0.5]},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
}

def train():
    run = wandb.init()          # the agent fills run.config with the next setting
    cfg = run.config
    ...                         # build the model from cfg, train, run.log({"val/acc": ...})

sweep_id = wandb.sweep(sweep_cfg, project="ms2a-s4")
wandb.agent(sweep_id, function=train, count=20)
```

The same config as YAML with `wandb sweep sweep.yaml` and `wandb agent <id>` lets
you start agents on several machines at once. `grid` for a handful of discrete
values, `random` for a first look at a wide space, `bayes` when each run is expensive
and you can afford to wait for the model of the space to become useful — roughly ten
runs in. `log_uniform_values` matters: a learning rate is searched on a log scale,
never a linear one.

![A sweep's workspace: the run list, validation curves, parameter importance, the parallel-coordinates panel and the System section (W&B documentation)](assets/nn/wandb-sweep-workspace.png)

The sweep workspace assembles the panels for you: `val_acc` against `lr`, a
*parameter importance* panel that fits a random forest from config to metric and
ranks the inputs, and the parallel-coordinates view. The **System** section at the
bottom — GPU power, GPU memory — is recorded for every run without a line of code;
[Performance and Profiling](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/performance-and-profiling)
makes it the first thing you look at.

---

## NaN hunting

```python
loss = criterion(logits, yb)
if not torch.isfinite(loss):
    run.alert(title="non-finite loss",
              text=f"{run.name}: loss={loss.item()} at step {step}",
              level=wandb.AlertLevel.ERROR)
    raise RuntimeError(f"non-finite loss at step {step}")
```

Crash on the step it happens, not three epochs later when every weight is `nan`.
`run.alert` sends the message to your email or Slack once *Scriptable run alerts* is
switched on in your W&B user settings; the same panel offers a no-code alert for
"run finished" and "run crashed". Then work backwards — inputs, logits, loss,
gradients — and the first non-finite one names the layer:

```python
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print("non-finite gradient in", name)
```

The usual causes in order: learning rate too high; an unnormalised input column with
values in the millions (shape **B** above); `log(0)` or a division by a zero variance
in a custom loss; `sqrt` at exactly zero, which is finite but whose gradient is not.

`torch.autograd.set_detect_anomaly(True)` makes the backward pass raise on the first
op that produces a `nan`, with a traceback to the forward line that created it. It
is several times slower — use it to locate, then remove it.

---

## Checkpoints and data as Artifacts

A checkpoint you cannot connect to its run is a file called `best.pt` in a
directory called `old`. An Artifact is a versioned, content-addressed folder tied to
the run that produced it and every run that consumes it.

```python
if val_acc > best_acc:
    best_acc = val_acc
    torch.save(model.state_dict(), "best.pt")
    art = wandb.Artifact(f"mlp-{run.id}", type="model",
                         metadata={"val_acc": val_acc, "epoch": epoch})
    art.add_file("best.pt")
    run.log_artifact(art, aliases=["best"])

# in another script
art = run.use_artifact("mlp-abc123:best", type="model")
state = torch.load(f"{art.download()}/best.pt", weights_only=True)
```

![The lineage graph of a dataset artifact: the runs that produced it and the model versions that consumed it (W&B documentation)](assets/nn/wandb-artifact-lineage.png)

`type="dataset"` works the same way — `art.add_dir("data/")` uploads only the files
whose content changed — and the lineage view then answers "which data trained the
model that is in production" in one click. Free accounts get 100 GB of artifact
storage; a checkpoint per *improving* epoch, not per epoch, stays well inside it.

---

## Reproducibility, honestly

Seeding gets you close, not identical: cuDNN picks convolution algorithms by
benchmark, and several GPU reductions use atomics whose ordering varies.
`torch.use_deterministic_algorithms(True)` removes most of it, at a real cost in
speed, and raises on any op with no deterministic implementation.

W&B records for every run the command line, the hostname, the Python version, the
git commit hash and a `diff.patch` of uncommitted changes, and a `requirements.txt`
of the environment. You add the rest to `config`: the seed, a hash of the dataset
file, and the flags that decide whether the run is deterministic.

```python
run.config.update({"seed": seed, "deterministic": True,
                   "data_sha256": sha256_of("data/train.npz")})
```

A result you cannot regenerate is not a result. A result you can regenerate to
within a documented tolerance is one — say which you have.

---

## Without an account, without a network

```python
wandb.login(anonymous="allow")      # no account: a temporary one is created
run = wandb.init(project="ms2a-s4")
```

`anonymous="allow"` gives you a run page that works for seven days and a banner to
claim it into a real account later; the link is the only credential, so do not post
it. A free account takes a minute and is the sane default.

```python
run = wandb.init(project="ms2a-s4", mode="offline")   # or WANDB_MODE=offline
...
# later, on a connected machine:
#   wandb sync wandb/offline-run-2026...
```

`mode="offline"` writes everything to a local `wandb/` directory and `wandb sync`
uploads it later — how every figure in this lesson was produced. It is also the
right setting for a machine with no route to the internet. The challenge-8 agent in
Lab 4 is one: it has no network and a 60-second budget, so W&B belongs in the
training script you develop and measure with, never in `agent.py`. The second half
of that discipline —
[Performance and Profiling](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/performance-and-profiling)
— is next.

---

## Check yourself

1. The loss falls quickly to a floor near chance and stays there, and the run
   *does* pass the overfit-one-batch test. What is the first suspect, and what in
   `__getitem__` would you check?

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   torch.manual_seed(0)
   x, y = torch.randn(32, 16), torch.randint(0, 4, (32,))
   net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
   opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
   for _ in range(200):
       loss = nn.functional.cross_entropy(net(x), y)
       opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
   print(loss.item() < 0.01)      # -> True
   ```

   Now change `opt = torch.optim.AdamW(net.parameters(), ...)` to build the
   optimizer on `nn.Linear(16, 4).parameters()` instead. What do you see, and which
   panel of the three-shapes figure is it?

3. Your runs table shows `val/acc = 0.91` for a run whose curve clearly peaked at
   0.95 six epochs before the end. Which one line is missing from its script?

4. Why is a training-loss curve on its own uninformative, and what does "val flat,
   train falling" tell you to do?

5. A student runs `wandb.watch(model, log="all", log_freq=1)` on a 20 M-parameter
   model and the epoch takes three times longer. Why, and what should the call be?

6. You are searching a learning rate between 1e-5 and 1e-1 with a Bayesian sweep.
   Which `distribution` do you give it, and what goes wrong with `uniform`?
