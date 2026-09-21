# Optimization and Schedules

The optimizer decides what to do with a gradient. The schedule decides how large
that step is at each point in training. Together they are worth more than any
architecture change you will make this session, and one of the numbers they
expose — the learning rate — is worth more than everything else on the page.

This lesson is about choosing those numbers from evidence instead of habit.
[Making Training Work](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/making-training-work)
owns what happens inside the network — activations, normalization, clipping;
[Data Pipelines and the Training Loop](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/data-pipelines-and-training-loop)
owns the loader and the shape of the loop. Here the loop is given, and we decide
what the update rule does with each batch it produces.

Every measurement quoted below was run while writing this lesson, on three CPU
threads — the allocation the challenge-8 agent gets. The last section trains to a
wall clock rather than to an epoch count, because that is the constraint Lab 4
hands you: 60 seconds to train, 60 seconds to predict.

<!-- notes: 30 minutes. The 12h module gave them "use Adam at 1e-3"; this lesson
is why that is a default and not an answer. Run the LR range test live — forty
seconds, and it is the one thing from this session they will reuse in every
project afterwards. The early-stopping figure is the surprise: restoring the
best-*loss* checkpoint costs 1.2 accuracy points, so it makes the "monitor what
you are graded on" point better than any slide. If time runs short, cut the
warm-restarts panel and the batch-size table; never cut the range test or the
60-second budget. -->

---

## Momentum, and why a first-order step is not enough

![Plain SGD stalling in the first basin while SGD with momentum carries through it](assets/nn/optimize_with_momentum.gif)

Plain SGD follows the current gradient and nothing else, so it stops wherever the
gradient is zero — including the first shallow dip it happens to meet. Both markers
above start together on the upper-left slope; the red one settles in the shallow
basin and stays there, while the green one arrives carrying speed, rolls over the
bump, and ends in the deeper minimum on the right.

Momentum accumulates a velocity instead of consuming each gradient in isolation:

$$
v_t = \beta v_{t-1} + g_t \qquad \theta_{t+1} = \theta_t - \eta v_t
$$

With $\beta = 0.9$ the velocity is a decaying sum over roughly the last ten
gradients. Components that keep pointing the same way compound toward an effective
step of $\eta / (1 - \beta)$ — ten times the nominal rate — while components that
flip sign every step cancel. `momentum=0.9` is the value; it is almost never worth
tuning, and `nesterov=True` costs nothing and helps slightly.

---

## Three optimizers on one ill-conditioned bowl

![Eighty steps of SGD, SGD with momentum and Adam on a quadratic whose curvature differs by a factor of a hundred between the two directions](assets/nn/optimizer-trajectories-sgd-momentum-adam.png)

The surface is $f(\theta) = \tfrac{1}{2}(\theta_1^2 + 100\,\theta_2^2)$: a hundred
times more curved in one direction than the other, which is what a real loss
surface looks like locally. Eighty steps of the actual `torch.optim` classes from
the same starting point.

- **SGD**, `lr=0.019` — the largest rate the steep direction tolerates before it
  diverges. Almost the whole step budget goes into zig-zagging across the ravine
  rather than along it. After 80 steps, $f = 0.836$: still most of the way out.
- **SGD + momentum 0.9**, `lr=0.01`. It rings *harder* at first, because the
  effective rate in the steep direction is $0.01/(1-0.9) = 0.1$, five times what
  plain SGD could survive. But the ringing decays while the consistent
  shallow-direction component compounds, and it arrives: $f = 0.00528$, a
  hundred and fifty times lower than SGD.
- **Adam**, `lr=0.15`. It rescales each coordinate by its own gradient history, so
  the 100:1 conditioning is divided out and the path is nearly a straight line.
  $f = 0.0132$.

Read the three learning rates again: 0.019, 0.01, 0.15. They are not comparable
across optimizers, and this is the single most common source of a diverging run.
An SGD rate is a distance in parameter space. An Adam rate is closer to "how far to
move in units of this parameter's own recent gradient scale", which is why `1e-3`
transfers across wildly different models — and why an SGD rate of 0.1 handed to
Adam explodes on the first step.

---

## Adam, precisely

Momentum on the gradient, and a second momentum on its square used to normalize
the step per parameter:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \qquad
\theta_{t+1} = \theta_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The hats
are bias corrections: both moments start at zero, so without them the first steps
would be biased toward zero by exactly the factor $1 - \beta^t$ — and since
$\beta_2^t$ needs about a thousand steps to decay, that correction matters far
longer for $v$ than for $m$.

A parameter whose gradients are consistently small gets a proportionally larger
step. That is why Adam works out of the box where SGD needs a per-layer rate, and
it is also the mechanism that breaks weight decay in the next section.

The cost is memory: two extra tensors the size of your parameters — 8 bytes per
`float32` parameter against momentum's 4, on top of the 4 for the weight and 4 for
its gradient. On a 10 M-parameter model that is 160 MB of optimizer state before
a single activation is stored.

---

## Weight decay, coupled and decoupled

```python
torch.optim.Adam(model.parameters(),  lr=1e-3, weight_decay=1e-2)   # coupled — not what you meant
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)   # decoupled
```

`Adam(weight_decay=λ)` implements classical L2 by adding $\lambda\theta$ to the
gradient — and then divides the whole gradient, decay included, by
$\sqrt{\hat{v}}$. The decay a parameter actually receives is therefore scaled by
$1/\sqrt{\hat{v}}$: it depends on how large that parameter's gradients happen to
be, which has nothing to do with how much you wanted to regularize it.
`AdamW` applies $\theta \leftarrow (1 - \eta\lambda)\,\theta$ directly to the
weight, outside the normalized update.

![Two weights with identical decay but different gradient scales, under Adam and under AdamW](assets/nn/weight-decay-adam-l2-vs-adamw.png)

Both weights start at 1.0 and receive a **zero-mean** gradient — no data signal at
all, so the only thing that should move them is the decay. One sees gradients of
scale 10, the other of scale 0.1; 512 replicas averaged, `lr=0.01`,
`weight_decay=0.1`, 1000 steps. The decay you asked for is the dashed line,
$(1 - \eta\lambda)^t$, which reaches **0.368** after 1000 steps.

Left, `Adam`: the large-gradient weight ends at **0.920**, essentially
unregularized, while the small-gradient weight is driven through zero by step 600
and ends at **−0.004**. One $\lambda$, two behaviours three orders of magnitude
apart, decided by a quantity you did not choose. Right, `AdamW`: **0.379** and
**0.369** — both on the dashed line, both a function of $\lambda$ and $\eta$ alone.

**Use `AdamW`.** There is no situation in this course where `Adam` with a non-zero
`weight_decay` is the right call. Note the consequence of decoupling, too: in
`AdamW` the decay is multiplied by the learning rate, so a schedule that anneals
$\eta$ to zero also anneals the regularization away. How much decay, measured
against every other regularizer, is
[Regularization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/regularization).

---

## Do not decay the biases and the norm parameters

```python
decay    = [p for _, p in model.named_parameters() if p.ndim > 1]
no_decay = [p for _, p in model.named_parameters() if p.ndim <= 1]

opt = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": 0.01},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
)
```

Weight decay encodes a prior: "smaller weights generalize better". That is true of
a weight matrix and false of a bias or a normalization parameter. Pulling a
BatchNorm $\gamma$ toward zero scales the layer's whole output toward zero, which
is damage, not regularization — see
[Making Training Work](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/making-training-work)
for what $\gamma$ and $\beta$ do. The convenient part is that the parameters you
want to exclude are exactly the one-dimensional ones, so `p.ndim > 1` is the whole
rule, no name matching required.

Two parameter groups means `opt.param_groups` has two entries, and every scheduler
in this lesson drives all of them. When you log the rate, `param_groups[0]["lr"]`
is the decayed group — fine, as long as you know which one you are reading.

---

## The learning-rate range test

Everything else in this session moves your score by a point or two. The learning
rate moves it between "works" and "does not". Do not guess it. Raise it
geometrically over a few hundred steps, record the loss, and read the answer off
the plot.

```python
model, opt = build()            # a throwaway copy — this run destroys the weights
crit = nn.CrossEntropyLoss()
lrs, curve, avg, beta = torch.logspace(-6, 0, 300), [], 0.0, 0.98

for i, (lr, (xb, yb)) in enumerate(zip(lrs, loader)):
    for g in opt.param_groups:
        g["lr"] = lr.item()
    loss = crit(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    avg = beta * avg + (1 - beta) * loss.item()
    curve.append(avg / (1 - beta ** (i + 1)))       # bias-corrected EMA, as in Adam
    if curve[-1] > 4 * min(curve):
        break                                       # diverged; the rest tells you nothing
```

![Training loss against learning rate over 300 steps, raised geometrically from 1e-6 to 1](assets/nn/lr-range-test-mnist.png)

An MLP 784-256-10 with `AdamW` and batch 128, on MNIST. The curve has four regions
and you want the boundary between the second and the third:

- **Flat**, below about $10^{-4}$. The loss sits at 2.30, which is $\ln 10$ — the
  cross-entropy of guessing uniformly over ten classes. The steps are too small to
  have done anything at all.
- **Descending**, roughly $10^{-4}$ to $10^{-2}$. This is the usable band. Its
  steepest point is at **lr ≈ 6.2e-4**.
- **Minimum** of the smoothed loss, at **lr ≈ 2.6e-2** — the lowest point on the
  plot, and already too large to train at: the raw per-batch losses behind it are
  visibly scattering.
- **Diverging**, past about $5 \times 10^{-2}$. The run aborted itself when the
  smoothed loss passed four times its minimum, which is what a real range test
  does.

Take the steepest descent, not the minimum. The old rule of thumb — "one order of
magnitude below the minimum" — would give $2.6 \times 10^{-3}$ here, four times
higher than the steepest point; both are defensible constant rates, and the plot is
what tells you the safe band spans nearly two decades rather than leaving you to
hope. For a **schedule**, use the band's two ends: `max_lr` a factor of a few below
the minimum, and the annealing takes care of the rest.

The test costs 300 steps — forty seconds here — and the model it leaves behind is
garbage, so rebuild before training. Re-run it whenever you change the
architecture, the batch size or the normalization, because all three move the
curve. Logged to
[Weights & Biases](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging),
the variants overlay on one panel.

---

## The schedules

![Six PyTorch learning-rate schedules over a thirty-epoch budget](assets/nn/lr-schedules-grid.png)

All six curves are what the real `torch.optim.lr_scheduler` classes produce over 30
epochs of 100 batches, `max_lr = 1e-3`.

| Schedule | Shape | Stepped | Use it when |
|---|---|---|---|
| `StepLR(step_size, gamma)` | ×0.1 drops at fixed epochs | per epoch | you are reproducing a result that used it |
| `CosineAnnealingLR(T_max)` | smooth decay to ~0 | per epoch | fixed budget, nothing left to tune |
| `OneCycleLR(max_lr, total_steps)` | warm up, then anneal to ~0 | **per batch** | fixed budget, fewest steps |
| `LinearLR` → `CosineAnnealingLR` via `SequentialLR` | explicit warmup, then cosine | **per batch** | transformers, large batches |
| `CosineAnnealingWarmRestarts(T_0)` | sawtooth | per epoch | you want several checkpoints to ensemble |
| `ReduceLROnPlateau(factor, patience)` | drops when the metric stalls | per epoch, **with the metric** | the budget is open-ended |

`OneCycleLR` is the strongest default whenever you know the step count in advance.
Its warmup occupies `pct_start` of the total (0.3 by default) and it starts at
`max_lr / div_factor`, one twenty-fifth of the peak, so the warmup is built in and
you pass one number instead of composing two schedulers.

The bottom-right panel is the odd one out. Its validation loss stops improving
around epoch 12; `ReduceLROnPlateau(factor=0.1, patience=3)` waits its three epochs
and cuts the rate at epoch 16, then again at 26. It is the only scheduler here that
reads a number out of your run — `sched.step(val_loss)`, not `sched.step()` — and
consequently the only one whose curve cannot be drawn before training starts.

---

## Step it at the right frequency

```python
for xb, yb in train_dl:
    ...
    opt.step()
    sched.step()          # OneCycleLR and every other per-batch schedule: inside the batch loop
```

Getting this wrong does not raise. Take a run with 469 batches per epoch over 30
epochs: `OneCycleLR` sized for 14,070 steps but stepped once per epoch advances 30
steps into its own schedule and finishes training still inside the warmup, having
never come down at all. The reverse mistake — a per-epoch scheduler stepped per
batch — collapses the entire decay into the first epoch and trains the remaining 29
at a rate of essentially zero. Both produce a loss curve that merely looks
disappointing.

Log `opt.param_groups[0]["lr"]` alongside every training loss. One scalar, and both
mistakes become a shape you can see in
[the workspace](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging)
within the first epoch.

---

## Warmup

At step 1, Adam's $\hat{v}$ is built from a single batch of gradients, so the
per-parameter scaling it divides by is close to noise. A full-size step taken on
that noise can destroy the initialization before training has begun — and the
damage is worst exactly where the network is deepest.

```python
warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=500)
cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - 500)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[500])
```

500 to 2000 linear steps, then cosine. Mandatory for transformers, for batches in
the thousands and for post-norm residual stacks; harmless everywhere else, which is
why `OneCycleLR` includes it and why you rarely have to write the three lines
above.

---

## Batch size, learning rate and throughput

![Training throughput and one-epoch test loss against batch size, on three CPU threads](assets/nn/batch-size-throughput-and-loss.png)

One epoch of the full 60,000-image MNIST through an MLP 784-512-256-10, `AdamW`,
in-memory tensors, three CPU threads. Measured, not estimated:

| Batch | Samples / s | Steps / epoch | Epoch time | Test acc, lr 1e-3 | Test acc, lr ·√(B/128) |
|---|---|---|---|---|---|
| 16 | 524 | 3750 | 114.5 s | 0.9656 | 0.9663 |
| 32 | 3 284 | 1875 | 18.3 s | 0.9674 | 0.9690 |
| 64 | 7 227 | 938 | 8.3 s | 0.9687 | **0.9698** |
| 128 | 12 846 | 469 | 4.7 s | 0.9643 | 0.9643 |
| 256 | 21 871 | 235 | 2.7 s | 0.9598 | 0.9616 |
| 512 | 43 132 | 118 | 1.4 s | 0.9516 | 0.9599 |
| 1024 | 43 183 | 59 | 1.4 s | 0.9361 | 0.9563 |
| 2048 | 51 220 | 30 | 1.2 s | 0.9123 | 0.9433 |

Two independent things are happening at once. **Throughput** rises with the batch
because the per-step overheads — the Python loop, the optimizer update, the thread
barriers around each small matrix multiply — are paid once per *step*, not once per
sample; an epoch at batch 128 is 24 times faster than at batch 16. **Quality per
epoch** falls, because a larger batch buys fewer updates: 30 steps at batch 2048
against 938 at batch 64.

The learning rate is what reconciles them. Held at `1e-3`, batch 2048 loses 5.6
accuracy points against batch 64 — 91.2% against 96.9%. Scaled as
$\eta \propto \sqrt{B}$, the rule for Adam-family optimizers (SGD uses the linear
$\eta \propto B$), it recovers 3.1 of those points, to 94.3%. Both rules are
approximations that break above a few thousand, and neither substitutes for
re-running the range test at the batch size you will actually ship.

Look at the throughput curve between 512 and 1024: flat. Past the point where the
matrix multiplies saturate the cores, a larger batch buys no speed and still costs
you updates. Find that knee once per machine, then take the largest batch below it —
[Performance and Profiling](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/performance-and-profiling)
is about measuring it properly.

---

## Early stopping, and the restore that is the point

```python
best, wait, patience = float("inf"), 0, 10
for epoch in range(max_epochs):
    train_one_epoch(model, train_dl, opt, sched)
    val_loss, val_acc = evaluate(model, val_dl)
    if val_loss < best - 1e-4:
        best, wait = val_loss, 0
        torch.save({"model": model.state_dict(), "epoch": epoch}, "best.pt")
    else:
        wait += 1
        if wait >= patience:
            break

model.load_state_dict(torch.load("best.pt", weights_only=True)["model"])   # the point
```

Stopping without restoring keeps the *last* model — the overfitted one whose decline
was your reason for stopping. The `load_state_dict` is the feature; the `break` only
saves time.

![A forty-epoch overfitting run with the best checkpoint at epoch five and the stop at epoch fifteen](assets/nn/early-stopping-best-checkpoint-restore.png)

A real run: MLP 784-256-256-10 on 3,000 MNIST digits with 10,000 held out, `AdamW`
at `1e-3`, no regularization, `patience=10`. Validation loss bottoms out at
**0.262 at epoch 5** and never beats it again; the patience window expires at epoch
15 and training stops at **0.292**. The pale continuation shows what the remaining
24 epochs would have bought — validation loss climbing to 0.36 while the training
loss sits at 0.002.

Now read the lower panel, which is the uncomfortable part. Validation *accuracy* at
the restored epoch-5 checkpoint is **92.4%**. At epoch 15, the model you would have
kept by not restoring, it is **93.6%**. Monitoring loss while being graded on
accuracy cost 1.2 points. There is no contradiction: cross-entropy punishes a
confidently wrong prediction without limit, so a handful of hardening mistakes can
drive the mean loss up while the `argmax` — all that accuracy sees — keeps
improving on everything else.

> Early-stop on the metric you are graded on. For challenge 8 that is accuracy.

How much early stopping buys as a *regularizer*, next to weight decay, dropout
and augmentation on the same data, is measured in
[Regularization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/regularization).

Two more rules. `patience` must be **larger** than any scheduler's, or you stop
before the rate drop that would have rescued the run: `ReduceLROnPlateau(patience=3)`
underneath early stopping at `patience=10` is the right ordering, never the reverse.
And what you restore here is the model only — see
[Data Pipelines and the Training Loop](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/data-pipelines-and-training-loop)
for everything a *resumable* checkpoint additionally has to carry.

---

## Training to a clock

Challenge 8 gives you no epoch budget. It calls `train(X_train, y_train)` behind a
60-second deadline and then `predict(X_test)` behind another one, and a timeout
scores 0.0 — not a reduced score, zero. An epoch count is a guess about how fast
your machine is. A clock is not a guess.

```mlarena:challenge id=8
```

```python
import time, torch, torch.nn as nn

class Agent:
    def __init__(self, output_dim=10, seed=None):
        self.output_dim, self.seed = output_dim, seed

    def train(self, X_train, y_train):
        torch.set_num_threads(3)                    # the agent container has 3 cores
        t0, budget = time.perf_counter(), 52.0      # 60 s deadline, 8 s of margin
        X, y = self._prepare(X_train, y_train)
        self.model = self._build(X.shape[1])        # a fresh model on every call
        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3, weight_decay=0.01)
        crit = nn.CrossEntropyLoss()
        bs, step, sched = 256, 0, None

        while time.perf_counter() - t0 < budget:
            for idx in torch.randperm(len(X)).split(bs):
                loss = crit(self.model(X[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if sched is None and step == 20:            # 20 steps is enough to time one
                    spent = time.perf_counter() - t0
                    total = step + int((budget - spent) / (spent / step))
                    sched = torch.optim.lr_scheduler.OneCycleLR(
                        opt, max_lr=3e-3, total_steps=total, pct_start=0.25)
                elif sched is not None and step < sched.total_steps:
                    sched.step()
                if time.perf_counter() - t0 > budget:
                    break
```

`torch.set_num_threads(3)` matches the container and also makes your local timings
mean something. The margin is not optional: setup, the final batch and the model's
first allocation all happen inside the deadline. And `train` builds the model from
scratch every time it is called, because the environment calls it a second time on
a different dataset when your accuracy is high enough to be worth checking.

![Test accuracy against training wall-clock for three configurations under a fifty-second budget](assets/nn/budget-60s-cpu-accuracy-vs-seconds.png)

Three configurations, 50 seconds of training wall-clock each, on a permuted MNIST
built with challenge 8's recipe — a fixed random pixel permutation plus mild
per-image noise — through an MLP 784-512-256-10 on three threads. Evaluation time
is excluded from the axis.

- **batch 64, constant `1e-3` → 97.96%.** The smallest batch pays the most per-step
  overhead and gets through the fewest samples in the time available.
- **batch 256, constant `1e-3` → 98.15%.** Same optimizer, four times fewer steps
  over the same seconds, more samples seen.
- **batch 256, `OneCycleLR` to `3e-3` sized to the budget → 98.63%**, and it got
  there in **27 seconds**.

The schedule is worth about half a point over the identical optimizer at the
identical batch size, and it earns it late: at the 10-second mark all three runs are
within 1.2 points of each other, and the gap opens as the annealing phase arrives.
That is the argument for a schedule under a budget — the last third of the steps,
taken at a small rate, is what turns a good model into a settled one.

The green run stopping at 27 seconds is the sizing bug, visible: it timed itself
over its first 20 steps, which are slower than steady state because the allocator
and the thread pool are still warming up, so it over-estimated its own step time,
built a schedule for too few steps and finished with a third of the budget unspent.
It still won. Time steps 5 through 25 instead of 0 through 20, or re-size once at a
quarter of the budget, and it has 23 more seconds of annealing to spend.

For calibration, on the platform itself: the published benchmark on challenge 8 —
a pure-numpy softmax regression — scores **0.9256**, and a torch MLP of roughly this
shape, trained inside the real 60-second deadline, scores **0.9834**. The module's
pass bar is **0.926**.

---

## Defaults

- **`AdamW`**, with biases and normalization parameters in a `weight_decay=0.0`
  group.
- **A learning rate from a range test**, not from memory: the steepest-descent point
  as a constant rate, or a factor of a few below the minimum as `OneCycleLR`'s
  `max_lr`.
- **`OneCycleLR` over the real step count**, stepped per batch. `CosineAnnealingLR`
  if you would rather pass one fewer argument.
- **The largest batch below the throughput knee**, with the rate re-scaled by
  $\sqrt{B/B_0}$ from wherever you measured it — then re-measured.
- **Early stopping on the graded metric**, restoring the best checkpoint,
  with a patience larger than any scheduler's.
- `clip_grad_norm_(..., 1.0)`, and `lr` and `grad_norm` logged on every training
  log.

`SGD(momentum=0.9, nesterov=True)` under a cosine schedule still wins some vision
benchmarks on half the optimizer memory, given a long budget and the patience to
tune it. Under a 60-second clock you have neither. Change one thing at a time, and
keep the run that justified the change.

---

## Check yourself

1. In the coupled-decay figure, one weight ended at 0.920 and another at −0.004
   under the same `weight_decay=0.1`. What decided which was which, and why does
   that make `Adam(weight_decay=...)` the wrong tool rather than merely a
   differently-tuned one?

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   m = nn.Linear(4, 1)
   opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
   sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=100)
   lrs = []
   for _ in range(100):
       opt.step(); sched.step(); lrs.append(opt.param_groups[0]["lr"])
   print(f"{lrs[0]:.5f} {max(lrs):.5f} {lrs[-1]:.2e}")
   # -> 0.00004 0.00100 5.07e-07
   ```

   The whole curve is spent in 100 **steps**, not 100 epochs. Your real loop has 469
   batches per epoch and runs 30 epochs, and `sched.step()` is in the epoch loop.
   What fraction of the schedule does it complete, and what does the learning rate
   do for the entire run?

3. Your range test is flat below `1e-4`, steepest at `6e-4`, and bottoms out at
   `3e-2`. You are about to train with `OneCycleLR` for a fixed 5,000 steps. What do
   you set `max_lr` to, and why is `6e-4` the wrong answer *here* when it was the
   right answer for a constant rate?

4. You raise the batch from 256 to 1024 to use the cores better, and one-epoch
   accuracy drops by two points. Name the two independent causes, say which one the
   throughput table shows you cannot fix, and give the one-line change that
   addresses the other.

5. A run is graded on accuracy, early-stopped on validation loss, and restores the
   best-loss checkpoint. Using the figure in this lesson, say what that costs and
   explain how validation loss can rise for ten epochs while validation accuracy
   also rises.

6. Your `train()` measures its own speed over the first 20 steps and sizes a
   `OneCycleLR` from the result. The run finishes with a third of its 60 seconds
   unspent. What went wrong, in which direction, and what would happen to your score
   if the estimate had erred the other way?
