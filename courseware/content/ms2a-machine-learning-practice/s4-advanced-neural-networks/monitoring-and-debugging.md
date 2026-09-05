# Monitoring and Debugging

A training run you are not logging is a forty-minute experiment that produces one
number. Instrument it, and the same run tells you what to change.

<!-- notes: 25 minutes. Have TensorBoard open with three real runs in it before
the lesson starts. The overfit-one-batch slide is the one they must leave with.
-->

---

## What to log, and how often

| Quantity | Frequency | Why |
|---|---|---|
| train loss | every ~50 steps | shape of the descent |
| val loss + the real metric | every epoch | the only thing you optimise for |
| learning rate | every epoch | catches a mis-stepped scheduler |
| gradient norm | every ~50 steps | vanishing, exploding, clipping |
| weight/grad histograms | every epoch | dead layers, drift |
| samples per second | every epoch | regressions in the pipeline |

A histogram of a 10 M-parameter tensor logged every step dominates the run time
and leaves gigabytes of event files. Scalars are cheap; histograms are not.

---

## Scalars

```python
writer = SummaryWriter(f"runs/{date}_{arch}_lr{lr}_s{seed}")
writer.add_scalar("loss/train", tr_loss, epoch)
writer.add_scalar("loss/val", va_loss, epoch)
writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
```

![TensorBoard scalar panel, two runs](assets/nn/Loss.png)

The `a/b` tag groups panels. Name the run directory after what you varied — you
will have forty by Friday, and `runs/exp3_final_v2` identifies none of them.

---

## The plot that tells you nothing

![Training loss alone over ten epochs](assets/nn/training_history.png)

Training loss falls, then ticks up at epoch 9. Overfitting? Noise? A learning
rate that is now too large? This plot cannot say, because the quantity that
answers it is not on it.

---

## Plot both, always

![Training and validation loss over 200 epochs](assets/nn/training_validation_loss.png)

Validation flattens near epoch 60 while training keeps descending. Everything
after that is memorisation: the best checkpoint is at epoch 60, and 140 epochs of
GPU time bought nothing.

| Shape | Diagnosis | Action |
|---|---|---|
| val flat, train falling | overfitting | stop earlier, regularise, more data |
| both flat and high | underfitting or LR too low | capacity, LR |
| loss oscillates, no trend | LR too high | lower it, clip |
| smooth then a vertical spike | one bad batch, or `nan` | inspect that batch |
| staircase drops | your scheduler | expected |
| val below train | dropout is on in train | expected, not a bug |

---

## Histograms

```python
for name, p in model.named_parameters():
    writer.add_histogram(f"weights/{name}", p, epoch)
    if p.grad is not None:
        writer.add_histogram(f"gradients/{name}", p.grad, epoch)
```

![Gradient distribution of one layer across training steps](assets/nn/histogram.png)

Each ridge is one epoch, most recent at the front. A healthy gradient
distribution is centred on zero and narrows slowly; one that collapses to a spike
at zero is a dead layer, one that widens without bound is heading for `nan`.

---

## The graph and the projector

```python
writer.add_graph(model, next(iter(train_dl))[0][:1])
writer.add_embedding(features, metadata=labels, tag="penultimate")
```

![TensorBoard computation graph](assets/nn/tensorboard_graph.png)

![TensorBoard embedding projector](assets/nn/embedding_projector.png)

You use each roughly once per project. The graph confirms that the module you
thought you wired in is in the forward pass. The projector reduces the
penultimate-layer activations to three dimensions and shows whether your classes
separate at all — the fastest way to tell "the model is weak" from "the labels
are noise".

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

Turn augmentation, dropout and weight decay off first — each puts a floor under
the achievable loss. If a network cannot memorise 32 examples it will not learn
60,000: twenty seconds of runtime, and the first thing to run after any
architecture change.

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
converges and produces a plausible score. This is how you find it.

---

## NaN hunting

```python
loss = criterion(logits, yb)
if not torch.isfinite(loss):
    raise RuntimeError(f"non-finite loss at step {step}")
```

Crash on the step it happens, not three epochs later when every weight is `nan`.
Then work backwards — inputs, logits, loss, gradients — and the first non-finite
one names the layer. The usual causes in order: learning rate too high; an
unnormalised input column with values in the millions; `log(0)` or a division by
zero variance in a custom loss; `sqrt` at exactly zero, which is finite but whose
gradient is not.

`torch.autograd.set_detect_anomaly(True)` names the offending backward op with a
traceback. It is several times slower — use it to locate, then remove it.

---

## Reproducibility, honestly

Seeding gets you close, not identical: cuDNN picks convolution algorithms by
benchmark, and several GPU reductions use atomics whose ordering varies.
`torch.use_deterministic_algorithms(True)` removes most of it, at a real cost in
speed, and raises on any op with no deterministic implementation. Record with
every run the seed, the git SHA and whether the tree was dirty, the config, the
package versions, and a hash of the dataset file.

```python
writer.add_text("config", json.dumps(cfg, indent=2))
```

A result you cannot regenerate is not a result. A result you can regenerate to
within a documented tolerance is one — say which you have.

---

## Check yourself

1. The loss falls quickly to a floor near chance and stays there. What is the
   first suspect?

   **Answer.** Inputs and labels misaligned in `__getitem__`. A dataset
   returning `self.y[i+1]` for `self.X[i]` still trains, still converges, and
   still produces a plausible score — which is why overfitting one batch is the
   test that finds it.

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

   Twenty seconds of runtime. A `False` here means stop and debug the
   plumbing — no real run is worth starting.

3. Why is a training-loss curve on its own uninformative, and what does "val
   flat, train falling" mean?

   **Answer.** Training loss alone cannot separate overfitting from noise from
   a learning rate that is now too large; the quantity that answers it is the
   validation loss, so plot both. Val flat while train falls is overfitting:
   stop earlier, regularise, or get more data.
