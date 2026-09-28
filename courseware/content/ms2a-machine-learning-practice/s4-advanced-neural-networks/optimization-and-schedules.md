# Optimization and Schedules

The optimizer decides what to do with a gradient. The schedule decides how large
that step is at each point in training. Together they are worth more than any
architecture change you will make this session, and one of the numbers they
expose — the learning rate — is worth more than everything else on the page.

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

The cost is memory: two extra tensors the size of your parameters — 8 bytes per
`float32` parameter against momentum's 4, on top of the 4 for the weight and 4 for
its gradient. On a 10 M-parameter model that is 160 MB of optimizer state before
a single activation is stored.


---


---

## Learning Rate schedules

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
