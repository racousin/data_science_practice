# Making Training Work

A loop that runs is not a loop that learns. Between the two sit four choices —
activation, normalization, loss, and what you do when the gradient misbehaves.
Each one is cheap to change and each one is the difference between a network
that reaches 0.97 and the identical network that sits at 0.10 forever. Three
more — initialization, depth and dropout — have lessons of their own.

One idea is under all of them. The gradient that reaches the first layer of an
$L$-layer network is a product of $L$ Jacobians, one per layer. Multiply numbers
slightly below 1 thirty times over and you get zero; slightly above 1 and you get
`inf`. Everything below is a way of holding that product near 1.

Every figure marked *measured* comes from a run made while writing this lesson —
scikit-learn's 1 797-image digits set, or a synthetic sequence task — small
enough that each one reproduces on a laptop in under a minute.

<!-- notes: 40 minutes, the core of the session. Do not re-teach layers (lesson
78) or optimizers and schedules (lesson 80). The live demo worth the time: print
the dead-unit fraction while raising the learning rate. Initialization, the
24-block plain vs residual pair and dropout each moved to a lesson of their own
(Parameter Initialization, Skip Connections, Dropout). The clipping section is
where RNN people get religion. -->

---

## The product that decides everything

$$
\frac{\partial \mathcal{L}}{\partial h_1}
= \frac{\partial \mathcal{L}}{\partial h_L}\prod_{\ell=2}^{L}\frac{\partial h_\ell}{\partial h_{\ell-1}}
$$

Each factor is set by a weight matrix and by the derivative of an activation.
Scale them wrongly and the product is $10^{-12}$ or `inf`, whatever the data says.
This is why the same five symptoms come back forever, and it is worth knowing
which section of this lesson each one belongs to.

| What you see | What it usually is | Where to look |
|---|---|---|
| loss pinned at $\ln 10 = 2.303$ | nothing reaches the early layers | *Initialization*, then *Skip Connections* |
| `nan` within the first few steps | init, learning rate, unnormalised inputs | *Initialization*, clipping |
| a whole layer outputs zeros | dead ReLU units | dead units |
| train and eval differ by points | BatchNorm statistics, dropout | normalization, then *Dropout* |
| gradient norm spikes to $10^{3}$ | exploding gradient | clipping |

Reading the curve that shows you these belongs to
[Monitoring and Debugging](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging);
this lesson is what to change once you have read it. The entries in italics are
lessons of their own:
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization),
[Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections)
and
[Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout).

---

## Activations

![Six activation functions and their derivatives](assets/nn/activation-functions-and-derivatives.png)

The top row is the derivative story in one picture. Sigmoid's derivative peaks at
0.25, so ten stacked sigmoids multiply the gradient by at most
$0.25^{10} \approx 9.5 \times 10^{-7}$ and the early layers receive nothing. tanh
reaches 1 at the origin but flattens in both tails, so any unit driven into
saturation stops learning. ReLU's derivative is exactly 1 on the positive side —
no shrinkage at all, at the price of exactly 0 on the negative side.

| | Range | Derivative | Reach for it |
|---|---|---|---|
| `ReLU` | $[0, \infty)$ | 0 or 1 | the default, everywhere |
| `LeakyReLU(0.01)` | $(-\infty, \infty)$ | 0.01 or 1 | when units are dying |
| `GELU` | $(-0.17, \infty)$ | smooth, slightly > 1 near 1.0 | transformers, modern CNNs |
| `SiLU` / swish | $(-0.28, \infty)$ | smooth | EfficientNet-style CNNs |
| `tanh` | $(-1, 1)$ | $\le 1$, vanishing in the tails | RNN internals, bounded outputs |
| `sigmoid` | $(0, 1)$ | $\le 0.25$ | binary **output** only |

**Default to ReLU. Use GELU if you are copying a transformer.** Never put a
sigmoid or a tanh in a hidden layer of a deep feed-forward stack — the numbers
above are the whole reason. The differences among ReLU, GELU and SiLU are worth a
few tenths of a point; the difference between ReLU and sigmoid is worth the run.

One more rule, and it is the most common beginner bug in the whole session:

```python
self.head = nn.Linear(256, 10)        # no activation here
loss = F.cross_entropy(self.head(h), y)
```

`cross_entropy` applies the log-softmax itself, so the last layer emits **logits**
— raw, unbounded, positive or negative. A `nn.Softmax` or a `nn.ReLU` before the
loss is not a style choice, it is a bug; the ReLU version cannot even represent a
negative logit.

---

## Dead ReLU units, measured

If a unit's pre-activation is negative for every sample in the dataset, its
gradient is exactly zero for every sample, forever. It is not learning slowly; it
is gone. A large learning rate is the usual cause: one oversized step drives the
bias hard negative and nothing can bring it back.

```python
h = model.fc1(xb)                                  # pre-activation, before the ReLU
dead = (h <= 0).all(dim=0).float().mean()
print(f"{dead:.0%} of fc1 units are dead on this batch")
```

![Dead units and test accuracy against learning rate for a two-hidden-layer MLP, measured](assets/nn/dead-relu-fraction-vs-learning-rate.png)

A 64-256-256-10 MLP on the digits set, SGD with momentum 0.9, 15 epochs, five
seeds per rate; a unit counts as dead when its pre-activation is $\le 0$ for
*every* training sample after training. Up to `lr = 0.03` nothing dies and the
network reaches 97.7%. At `lr = 0.1`, 1.9% of the second layer is dead and
accuracy is untouched at 97.4%. At `lr = 0.15` the mean jumps to 61% dead and
accuracy collapses to 59.6% — and the five seeds went 2%, 32%, 79%, 97%, 97%,
which is the real lesson: near the edge, the outcome is a coin flip. From
`lr = 0.3` up, essentially every unit of the second layer is dead, accuracy is
10.0% — chance on ten classes — and two of five seeds produced `nan` outright.

The first layer never lost a single unit at any learning rate. Death happens
where a layer is fed by *another* ReLU, because that input is already
non-negative and a negative bias can push the whole distribution below zero.

Above roughly 40% dead in a layer: lower the learning rate first, then switch to
`LeakyReLU` or `GELU`, which keep a gradient on the negative side. Choosing that
learning rate is
[Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules);
this figure is why the range test there is not optional.

---

## Normalization: BatchNorm and LayerNorm

Normalise, then let the network undo it if it wants to:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \qquad y = \gamma \hat{x} + \beta
$$

$\gamma$ and $\beta$ are learned parameters, one pair per feature. The only
question is which axis $\mu$ and $\sigma^2$ are pooled over, and that single
choice produces every practical difference between the two layers.

![Which axis BatchNorm and LayerNorm pool their statistics over, for 2-D and 3-D inputs](assets/nn/batchnorm-vs-layernorm-axes.png)

BatchNorm pools over the samples: one $(\mu, \sigma^2)$ per feature, computed
from the batch, which means a sample's output depends on the other samples it
travelled with. Because that is impossible at inference, BatchNorm keeps a
running estimate of both during training and substitutes it under `model.eval()`
— which is *why* `eval()` exists. LayerNorm pools over the features of one
sample, needs no running statistics, and behaves identically in both modes.

```python
nn.Sequential(nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU())
nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU())
```

`bias=False` in the first line: BatchNorm subtracts the mean immediately
afterwards, so that bias has no effect on anything except your parameter count.
Note also the two orders — **Linear → Norm → Activation → Dropout** is the
BatchNorm convention from the CNN literature, while transformers normalise
*first*, inside the residual block, for the reason given below. `BatchNorm1d`
wants channels in dimension 1 and `LayerNorm` normalises the last dimension, so
on a sequence one of them needs a `permute` — the layouts are in
[Essential Layers](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/essential-layers).

| | BatchNorm | LayerNorm |
|---|---|---|
| Statistics over | the batch, per feature | one sample, per sample |
| Depends on batch size | yes | no |
| `train()` ≠ `eval()` | yes | no |
| Extra state | running mean and variance | none |
| Sequence-friendly | needs `(B, C, T)` | native on `(B, T, D)` |
| Standard in | CNNs | transformers, RNNs, RL |

---

## What BatchNorm costs you, measured

![Test accuracy and train/eval discrepancy against batch size for BatchNorm, LayerNorm and no normalization, measured](assets/nn/normalization-and-batch-size.png)

The same 64-256-256-10 MLP with `BatchNorm1d`, with `LayerNorm`, and with
nothing, trained for 8 epochs with AdamW at 1e-3 across batch sizes from 2 to
128, three seeds each. From batch 16 up, the three are within half a point of
each other — normalization is not free accuracy on a small network. Below that,
BatchNorm falls apart: 88.0% at batch 2 against its own 98.3% at batch 16, and
2.0 points *worse* in `eval()` than the same weights scored with batch
statistics, because the running estimates were accumulated from pairs of
samples. LayerNorm and the unnormalised network have a train/eval gap of exactly
zero, by construction.

Normalization is insurance against depth and against a large learning rate, not a
free point of accuracy. In
[Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections),
a 24-block residual stack trains to the same 96% with or without it up to
`lr = 3e-3`; at `1e-2` the pre-norm version still trains and the one without
normalization turns to `nan`. Four ways BatchNorm bites:

- **Small batches.** Below ~16 the statistics are noise. At batch size 1 in
  `train()` mode `BatchNorm1d` raises `ValueError: Expected more than 1 value per
  channel when training`. Use `drop_last=True` in the `DataLoader` (see
  [Data Pipelines and the Training Loop](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/data-pipelines-and-training-loop))
  so the final short batch never reaches the model.
- **A gap that closes when you call `model.train()`.** The running statistics are
  wrong: too few updates, or a genuine shift between your splits. Never "fix" it
  by evaluating in `train()` mode.
- **Dropout before it.** Dropout's output variance differs between the two
  modes, so the norm accumulates statistics that are wrong at inference. Put
  dropout after the norm; a `Linear` between them does not remove the shift —
  [Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout) measured it.
- **Fine-tuning and tiny effective batches.** A frozen backbone in `eval()`
  behaves differently from the same backbone in `train()`; decide deliberately
  which one you want, rather than discovering it from a 3-point gap.

For sequences, for reinforcement learning, for batch size 1, and for anything you
plan to evaluate one sample at a time, use LayerNorm and stop thinking about it.

---

## Initialization, depth and dropout, briefly

**Initialization** sets every factor of the product at the top of this lesson
before the first step. He's rule, $\operatorname{Var}(w) = 2/n_{in}$, holds the
signal of a ReLU stack at a constant scale through any depth; PyTorch's default
is a third of LeCun's variance and decays it. The measurements, the defaults of
every layer and the parameters that start at zero on purpose are
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization).

**Residual connections** are the answer to the product at the top of this
lesson. A block that computes $x + F(x)$ has the derivative
$I + \partial F/\partial x$, so the gradient has a path to the first layer
along which nothing multiplies it. Past about fifteen weight layers a network
without that path is one you are debugging, not training. The measurements, the
block and its rules are
[Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections).

**Dropout** zeroes each unit with probability $p$ on every training step, scales
the survivors by $1/(1-p)$, and is exactly the identity under `model.eval()`. It
is a regularizer: it helps when validation diverges from training, and costs
convergence speed when it does not. How much, where, and what else to try first
are [Regularization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/regularization)
and [Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout).

---

## Losses that fit the problem

![Cross-entropy against squared error on probabilities, and the four regression losses, with annotations](assets/nn/loss-shapes-classification-and-regression.png)

The left panel is why classification uses cross-entropy and not squared error.
$-\log p$ is unbounded: a confidently wrong prediction keeps a large gradient
exactly where you need one. Squared error on probabilities saturates at 2 and its
gradient flattens where the model is most wrong, which is the worst possible
place to stop learning.

```python
F.cross_entropy(logits, y)                      # y: (B,) int64 class indices
F.cross_entropy(logits, y, weight=w)            # w: (C,) per-class weights
F.cross_entropy(logits, y, ignore_index=-100)   # padding in sequence tasks
F.binary_cross_entropy_with_logits(logit, y)    # binary, and multi-label
```

`cross_entropy` fuses log-softmax with the negative log-likelihood, and
`binary_cross_entropy_with_logits` fuses sigmoid with BCE; both are computed
through `logsumexp`, so they do not overflow the way a hand-written
`log(softmax(x))` does. Passing probabilities where logits are expected does not
raise — it just trains a worse model:

```python
logits = torch.tensor([[2.0, 0.5, 0.1]]); y = torch.tensor([0])
F.cross_entropy(logits, y)             # 0.3168   correct
F.cross_entropy(logits.softmax(1), y)  # 0.7448   softmax applied twice
```

For regression, the right panel is the whole decision:

| Loss | Grows like | Outliers | Use it when |
|---|---|---|---|
| `MSELoss` | $r^2$ | dominate the gradient | the target is clean and Gaussian-ish |
| `L1Loss` | $\lvert r \rvert$ | bounded influence | the target has heavy tails; you want the median |
| `HuberLoss(delta)` | $r^2$ then $\lvert r \rvert$ | bounded beyond `delta` | the usual answer |
| `SmoothL1Loss(beta)` | the same curve | the same | identical to Huber at `delta = beta`; the detection-head spelling |

At a residual of 4, MSE charges 16.0, L1 charges 4.0 and `HuberLoss(delta=1)`
charges 3.5; at a residual of 0.5 they charge 0.25, 0.5 and 0.125. One outlier of
magnitude 40 contributes as much squared error as 100 residuals of 4 — which is
how a single mislabelled row quietly steers an MSE model.

And normalise the *target*, not only the inputs. A target in the millions makes
the first gradients enormous whatever the loss; standardise it, train, then
invert the transform on the predictions.

---

## Label smoothing

![A one-hot target vector next to the same target with label smoothing 0.1](assets/nn/label-smoothing-target-vector.png)

A one-hot target asks for $p = 1$ on the true class, which a softmax can only
approach by driving the true logit to $+\infty$. Label smoothing replaces the
target with $(1-\epsilon)\cdot\text{onehot} + \epsilon/K$: at $\epsilon = 0.1$ and
$K = 10$, that is 0.91 on the truth and 0.01 on each of the other nine. The
optimum now requires the true logit to exceed the others by
$\log(0.91/0.01) \approx 4.5$ — a finite, reachable margin.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

One keyword. It typically buys a few tenths of a point on image classification
and it makes the model measurably less over-confident. Two things to know before
you read the logs: the loss now has a floor — the entropy of the smoothed target,
$\approx 0.50$ for $K = 10$, $\epsilon = 0.1$ — so a smoothed run's loss is only
comparable to another smoothed run, and it shrinks the spread of the logits,
which hurts if you intend to distil the model into a smaller one.

Use 0.05–0.1 for classification with many classes. Do not use it when the labels
are already noisy (you are smoothing twice) or when the metric rewards calibrated
probabilities you are about to threshold by hand.

---

## Custom losses

Any function of tensors that returns a scalar, built from differentiable torch
operations, is a loss.

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")   # (B,)
        pt = torch.exp(-ce)                                      # the true-class probability
        return ((1 - pt) ** self.gamma * ce).mean()
```

Focal loss down-weights the examples the model already gets right — the standard
answer to extreme class imbalance in detection (Session 6). Three ways to break a
custom loss silently: an `.item()` or a `.detach()` inside it, a NumPy round-trip,
or a missing reduction to a scalar. Assert once, on the first batch:

```python
assert loss.ndim == 0 and loss.requires_grad
```

---

## Measure the gradient, then clip it

```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Between `backward()` and `step()`, never anywhere else. `clip_grad_norm_`
computes the L2 norm of *all* gradients concatenated, and if it exceeds
`max_norm`, multiplies every gradient by `max_norm / total` — the direction is
preserved, only the length is cut. It returns the norm it saw **before** clipping,
which is the number to log; pass `max_norm=float("inf")` to measure without
clipping.

![Gradient norm and training loss with and without clipping, same seed, measured](assets/nn/gradient-norm-with-and-without-clipping.png)

A one-layer tanh `nn.RNN` (hidden 128) on the adding problem with 60 time steps,
plain SGD at `lr = 0.5`, same seed for both runs. Without clipping the gradient
norm goes 4.0, 23, $6.3\times10^{3}$, … , $2.1\times10^{18}$ at step 10 and `inf`
at step 11; the loss is `nan` for the remaining 1 489 steps. With
`max_norm=1.0` the identical run trains: the pre-clip norm exceeds 1.0 on 7 of
1 500 steps — 0.5% — and sits at a median of 0.14 over the last 500. Clipping did
not slow anything down. It removed seven steps out of 1 500, and those seven
steps were the whole difference between a model and a `nan`.

| Grad norm | Reading | Action |
|---|---|---|
| $10^{-7}$ and falling | vanishing | residuals, check activations, check init |
| stable, 0.1–10 | healthy | nothing |
| occasional spike to $10^{3}$ | one bad batch, or a genuinely sharp region | clip at 1.0 |
| above `max_norm` on most steps | the learning rate is too high | lower it; clipping is hiding it |
| `nan` | too late — a previous step destroyed the weights | crash on it instead |

`max_norm=1.0` is the standard value, and for recurrent models and transformers
clipping is not optional — a single exploding step is unrecoverable, as above.
Under mixed precision the gradients are scaled, so call `scaler.unscale_(opt)`
before clipping; that plumbing belongs to
[Performance and Profiling](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/performance-and-profiling).
Log `grad_norm` next to the loss and most of this lesson becomes visible from one
panel.

---

## The block that works

> `Linear(bias=False) → BatchNorm → ReLU → Dropout(0.1)`, He-initialized, wrapped
> in a residual once you pass ~15 layers, cross-entropy on logits, clipped at 1.0.
> LayerNorm instead of BatchNorm for sequences, for RL, and for anything scored
> one sample at a time.

Every choice in that line is a section above, or one of
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization),
[Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections)
and
[Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout),
and the combination trains at almost any depth. Change one thing at a time and
keep the run that justified the change — three variants at once tell you
nothing, and
[Monitoring and Debugging](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging)
is how you keep them apart.

Lab 4 puts this against a deadline: the
[1 Minute Permuted MNIST](https://ml-arena.com/viewchallenge/8) challenge gives
your agent 60 seconds to train and 60 more to predict, on 3 CPU cores and no GPU.
A pure-numpy softmax regression scores 0.9256 there. A small ReLU MLP — He
initialization, BatchNorm, no dropout, AdamW, gradient clipping — reaches 0.9834
inside the same budget, and every point of that gap comes from this lesson.

---

## Check yourself

1. A network's loss sits at exactly 2.303 for twenty epochs on a ten-class
   problem. Name three causes from this lesson and the three it hands over to,
   and say which figure shows each one.

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   torch.manual_seed(0)
   blk = nn.Sequential(nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU())
   print(sum(p.numel() for p in blk.parameters()))     # -> 66048
   x = torch.randn(8, 256)
   blk.train(); a = blk(x)
   blk.eval();  b = blk(x)
   print(torch.allclose(a, b))                         # -> False
   ```

   Where do the 66 048 parameters come from, why is the `Linear` bias absent on
   purpose, and what would make `a` and `b` equal?

3. You replace `BatchNorm1d` with `LayerNorm` in a model you evaluate one sample
   at a time, and the accuracy gap between `train()` and `eval()` disappears.
   Explain both halves of that sentence from the axis figure.

4. `clip_grad_norm_(model.parameters(), 1.0)` returns a value above 1.0 on 80% of
   your steps. Is clipping working? What do you change, and what would the same
   diagnostic look like if the problem were vanishing rather than exploding?

5. Your regression model tracks the bulk of the data well but is dragged by a
   handful of extreme targets. Which loss do you switch to, what does it charge
   at a residual of 4 versus MSE, and what else should you check about the target
   before blaming the loss?

6. A classmate reports 0.9834 on challenge 8 with a two-hidden-layer MLP, and
   yours — same architecture, same budget — reaches 0.60 and a gradient norm of
   $10^{-7}$. List the four things you would check, in order.
