# Optimization and Schedules

The optimizer decides what to do with a gradient; the schedule decides how big
that step is at each point in training. One of these two numbers matters far more
than the other.

<!-- notes: 30 minutes. The 12h module gave them "use Adam at 1e-3" — this lesson
is why that is a default and not an answer. Run the LR range test live; it takes
forty seconds and it is the thing they will actually reuse. -->

---

## Momentum

![Gradient descent with and without momentum](assets/nn/optimize_with_momentum.gif)

Plain SGD steps along the current gradient: in a ravine, that means oscillating
between the walls and crawling down the floor.

$$
v_t = \beta v_{t-1} + g_t \qquad \theta_{t+1} = \theta_t - \eta v_t
$$

Momentum accumulates a velocity, cancelling the oscillating component and
compounding the consistent one. `momentum=0.9`, and almost never worth tuning.

---

## Adam

Momentum on the gradient, and a second momentum on its square, used to normalise
the step size per parameter:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The hats are bias corrections: $m$ and $v$ start at zero, so early estimates are
divided by $1 - \beta^t$. A parameter with consistently small gradients gets a
proportionally larger step — which is why Adam works out of the box where SGD
needs per-layer tuning, and why it costs two extra tensors per parameter.

---

## Weight decay, done correctly

```python
optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)    # coupled, wrong
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)   # decoupled
```

L2 regularization added to the loss and Adam's per-parameter normalisation
interact badly: the effective decay ends up scaled by $1/\sqrt{\hat{v}}$, so the
parameters with the largest gradients are barely regularised. AdamW decouples
them, applying the decay straight to the weight.

**Use AdamW.** There is no situation in this course where `Adam` with
`weight_decay` is the right call.

---

## Do not decay biases or norm parameters

```python
decay = [p for _, p in model.named_parameters() if p.ndim > 1]
no_decay = [p for _, p in model.named_parameters() if p.ndim <= 1]
opt = optim.AdamW([{"params": decay, "weight_decay": 0.01},
                   {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
```

Weight decay is a prior that says "smaller weights generalise better". True of a
weight matrix, false of a BatchNorm $\gamma$ or a bias, where shrinking toward
zero damages the layer — and the one-dimensional parameters are exactly those.

---

## The LR range test

Everything else in this session moves your score by a few percent; the learning
rate moves it between "works" and "does not". Do not guess it — raise it
geometrically over a few hundred steps and record the loss.

```python
for lr, (xb, yb) in zip(torch.logspace(-7, 0, 100), loader):
    for g in opt.param_groups:
        g["lr"] = lr.item()
    loss = criterion(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
    history.append((lr.item(), loss.item()))
```

Plot loss against $\log(lr)$: flat, then descending, then exploding. Take a rate
one order of magnitude below the minimum — the steepest part of the descent, not
the bottom — then reset the model.

---

## Schedules

| Schedule | Shape | Use it when |
|---|---|---|
| `StepLR` | drops ×0.1 every N epochs | you know the epoch budget |
| `CosineAnnealingLR` | smooth decay to ~0 | the default, fixed budget |
| `OneCycleLR` | warm up, then cosine down | you want the fewest epochs |
| `ReduceLROnPlateau` | drops when val stalls | the budget is open-ended |

```python
sched = optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-3, total_steps=epochs * len(train_dl))
```

`OneCycleLR` is the strongest default for a fixed budget: it reaches a given
accuracy in noticeably fewer epochs than a constant rate. `ReduceLROnPlateau` is
the exception to the rule below — it is stepped as `sched.step(val_loss)`, with
the metric.

---

## Step it at the right frequency

```python
opt.step()
sched.step()          # OneCycleLR, and any other per-batch schedule
```

`OneCycleLR` and warmup schedules step **per batch**; `StepLR`,
`CosineAnnealingLR` and `ReduceLROnPlateau` step **per epoch**. Getting it wrong
does not raise — a per-batch schedule stepped per epoch completes 1/N of its
curve and the learning rate never comes down.

Log `opt.param_groups[0]["lr"]`. One scalar, and the mistake becomes visible.

---

## Warmup

Adam's second-moment estimate is built from a handful of gradients at step 1, so
its scaling is close to noise, and a large step on that noise destroys the
initialization before training starts.

```python
warm = optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=500)
cos = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - 500)
sched = optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[500])
```

500 to 2000 linear warmup steps, then cosine. Mandatory for transformers, large
batches and post-norm residual networks.

---

## Early stopping with a restored checkpoint

```python
if val_loss < best - 1e-4:
    best, wait = val_loss, 0
    torch.save({"model": model.state_dict(), "epoch": epoch}, "best.pt")
else:
    wait += 1
    if wait >= patience:
        break
model.load_state_dict(torch.load("best.pt")["model"])
```

Stopping without restoring keeps the *last* model — the overfitted one you
stopped because of. The restore is the point, not the break.

`patience` must be **larger** than any scheduler's, or training halts before a
scheduled drop can rescue it. And watch the metric you are graded on: stopping on
loss while reporting AUC picks a different epoch.

---

## Batch size interacts with everything

| | Small batch (32–128) | Large batch (1024+) |
|---|---|---|
| Gradient noise | high — acts as a regulariser | low |
| Steps per epoch | many | few |
| GPU utilisation | poor | good |
| Needs | nothing | LR scaling + warmup |

Doubling the batch roughly halves the gradient noise, so the same rate takes a
step that is half as exploratory. The linear scaling rule: multiply the batch by
$k$ and the rate by $k$ for SGD, by roughly $\sqrt{k}$ for Adam. Both break above
a few thousand — re-run the range test rather than trusting them.

---

## Defaults

- `AdamW`, `lr=3e-4`, `weight_decay=0.01`, biases and norms excluded
- `OneCycleLR` over the full step count, stepped per batch
- `clip_grad_norm_(..., 1.0)`, early stopping restoring the best checkpoint

SGD with momentum, a cosine schedule and a long budget still wins some vision
benchmarks on half the optimizer memory — reach for it when you have the time to
tune it, not before. Change one thing at a time, and only after the range test
has told you where you are.
