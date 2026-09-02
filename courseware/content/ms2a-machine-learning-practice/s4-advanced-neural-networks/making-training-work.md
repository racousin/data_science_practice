# Making Training Work

A loop that runs is not a loop that learns. Between the two sit six choices:
activation, initialization, normalization, regularization, depth, and what you do
when the gradients misbehave.

<!-- notes: 40 minutes, the core of the session. Every item here is the
difference between a network that reaches 0.92 and the identical network that
sits at 0.60 forever. Show the dead-ReLU count live. -->

---

## Activations

| | Range | Use it for |
|---|---|---|
| ReLU | [0, ∞) | the default, everywhere |
| LeakyReLU(0.01) | (−∞, ∞) | when units are dying |
| GELU | (−0.17, ∞) | transformers, modern CNNs |
| tanh | (−1, 1) | RNN internals, bounded outputs |
| sigmoid | (0, 1) | binary output only |

**Default to ReLU. Use GELU if you are copying a transformer.** Never put sigmoid
in a hidden layer: its derivative peaks at 0.25, so ten stacked layers multiply
the gradient by at most $0.25^{10} \approx 10^{-6}$ and the early layers receive
nothing. tanh saturates the same way, more slowly.

---

## Dead ReLU units

If a unit's pre-activation is negative for every input in the dataset, its
gradient is exactly zero forever — it is not learning slowly, it is gone. The
cause is usually a learning rate large enough to drive the bias hard negative.

```python
h = model.fc1(xb)
dead = (h <= 0).all(dim=0).float().mean()
print(f"{dead:.0%} of fc1 units are dead on this batch")
```

Above roughly 40% dead in a layer: lower the learning rate, then switch to
`LeakyReLU` or `GELU`, which keep a gradient on the negative side.

---

## Initialization decides whether the signal survives

![Weight distributions across layers for three initialization schemes](assets/nn/weight_distributions.png)

Three schemes across five layers of shrinking width. The fixed $\sigma = 0.1$
normal is identical in every layer; Xavier and He widen as the layer narrows,
because both scale the variance to the fan-in.

---

## Xavier and He

$$
\sigma^2_{Xavier} = \frac{2}{fan_{in} + fan_{out}} \qquad
\sigma^2_{He} = \frac{2}{fan_{in}}
$$

Xavier holds the activation variance constant through a symmetric activation; He
doubles it to compensate for ReLU discarding half of it.

```python
nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
nn.init.zeros_(m.bias)
```

PyTorch's defaults for `Linear` and `Conv2d` are sensible. Where initialization
bites is a hand-rolled parameter: `torch.randn(512, 784)` has unit variance,
roughly 28× too wide for that fan-in, and gives `nan` in three steps.

---

## Batch normalization

Normalise each feature over the batch, then let the network scale and shift it
back if it wants to:

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \qquad y = \gamma \hat{x} + \beta
$$

$\gamma$ and $\beta$ are learned. Running estimates of $\mu$ and $\sigma^2$
accumulate during training and are used at evaluation — the reason `model.eval()`
exists.

```python
nn.Sequential(nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU())
```

`bias=False`, because BatchNorm subtracts the mean and that bias is redundant.

---

## Where batch normalization breaks

- **Small batches.** Below ~16 the statistics are noisy; at batch size 1 in
  `train()` mode `BatchNorm1d` raises. `drop_last=True` prevents that.
- **A gap that closes in `train()` mode.** The running statistics are wrong — too
  few updates, or a shift between splits.
- **Dropout immediately before it.** Dropout's output variance differs between
  train and eval, so the norm learns statistics that are wrong at inference.

Default order inside a block: **Linear/Conv → Norm → Activation → Dropout.**

---

## Layer normalization

```python
nn.LayerNorm(256)      # over the last dimension, within one sample
```

| | BatchNorm | LayerNorm |
|---|---|---|
| Statistics over | the batch | one sample's features |
| Depends on batch size | yes | no |
| train ≠ eval | yes | no |
| Standard in | CNNs | transformers, RNNs, RL |

Never looking across samples makes it correct for sequences, for batch size 1,
and for RL, where the "batch" is whatever the environment just produced.

---

## Dropout

Each unit is zeroed with probability $p$ at training time and the survivors
scaled by $1/(1-p)$, so the expected activation is unchanged and evaluation needs
no correction.

```python
nn.Dropout(p=0.5)      # wide fully-connected head
nn.Dropout(p=0.1)      # convolutional and transformer blocks
```

Place it where the parameters are. On a narrow bottleneck it destroys information
rather than regularising, and it always slows convergence — reach for it after
you have observed overfitting, not before.

---

## Residual connections

$$
y = F(x) + x
$$

The derivative of that sum with respect to $x$ contains a term equal to 1: the
signal reaching layer 1 no longer has to survive a product of forty Jacobians.

```python
def forward(self, x):
    h = self.norm(x)
    return x + self.drop(self.fc2(self.act(self.fc1(h))))
```

The block must return the shape it received or the addition is illegal — project
the shortcut with a `1x1` convolution when the width changes. Normalising
*before* the block, as above, trains without warmup and tolerates a far wider
range of initializations.

The 2015 ResNet result is the one to remember: a plain 56-layer network had
*higher training* error than a plain 20-layer one. An optimization failure, not a
generalisation one, and the identity path fixed it.

> Beyond about 15 weight layers, add residual connections or do not add depth.

---

## Measure the gradient, then clip it

```python
loss.backward()
total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
writer.add_scalar("grad_norm", total, step)
optimizer.step()
```

Between `backward` and `step`, never anywhere else. It rescales the gradient
vector to norm at most `max_norm`, preserving direction, and returns the norm it
saw. Pass `1e9` to measure without clipping.

| Grad norm | Reading |
|---|---|
| $10^{-7}$ and falling | vanishing — add residuals, check activations |
| stable, 0.1–10 | healthy |
| spikes to $10^3$ | exploding — clip, and lower the LR |
| `nan` | too late; see the debugging lesson |

`1.0` is standard for transformers and recurrent models, where clipping is not
optional. If it fires on most steps, your learning rate is too high and clipping
is hiding it.

---

## Custom loss functions

Any function of tensors returning a scalar, built from differentiable torch
operations, is a loss.

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()
```

Focal loss down-weights the examples the model already gets right — the standard
answer to class imbalance in detection (Session 6). Three ways to break one
silently: `.item()` or `.detach()` inside it, a NumPy round-trip, a missing
reduction to a scalar. Assert once:

```python
assert loss.ndim == 0 and loss.requires_grad
```

---

## The block that works

> `Linear(bias=False) → BatchNorm → ReLU → Dropout(0.1)`, wrapped in a residual,
> He-initialized, clipped at 1.0.

Every choice in that block is one of the six above, and it trains at almost any
depth. Start there; change one thing at a time; keep the run that justified the
change.
