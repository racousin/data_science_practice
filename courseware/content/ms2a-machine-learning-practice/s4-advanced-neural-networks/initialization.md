# Parameter Initialization

Before the first step, every weight in the network already has a value, chosen
by a line of code most people never read. That value has two jobs. It must
**differ between units**, or they stay copies of each other forever. And it must
have **the right scale**, or the signal dies or explodes before it reaches the
loss. Get either wrong and no optimizer, schedule or regularizer later in this
session can recover it.

This lesson covers both jobs, what PyTorch does when you say nothing, when to
override it, and the handful of layers that are deliberately *not* initialized at
random. Where it sits in the session: the gradient that reaches the first layer
is a product of one Jacobian per layer; initialization sets that product at
step 0,
[Normalization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/normalization)
holds it during training, and
[Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections)
is what you add when initialization alone is no longer enough.

Every figure marked *measured* comes from a run made while writing this lesson,
on MNIST, in torch 2.x on a laptop CPU.

<!-- notes: 25 minutes. Two demos carry the lesson: the constant-init network
whose 256 units stay identical (print the rank), and the first loss of a
N(0, 1) network against ln 10. Derive the variance argument on the board in
three lines; do not derive Xavier's harmonic mean. The special-cases slide is
reference: point at it, do not read it. -->

---

## Job one: break the symmetry

If two hidden units start with the same incoming weights, they compute the same
output on every input, receive the same gradient, and take the same step. After
any number of steps they are still identical: the layer has the width of one
unit, however many you declared.

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.01)     # every unit the same
        nn.init.zeros_(m.bias)
```

*Measured*, on a 784-256-256-10 MLP trained for 1,500 steps of AdamW on 10,000
MNIST images, three seeds:

| Initialization | Test accuracy | Rank of the first layer's 256 × 784 weights |
|---|---|---|
| all zeros | 11.35% | 0 — every weight still exactly 0 |
| every weight 0.01 | 25.9% | 1 — all 256 rows identical |
| PyTorch default (random) | 96.0% | 256 |

All zeros is the degenerate case: every hidden unit outputs 0, so no weight
below the output layer receives any gradient, and after 1,500 steps the first
layer is still exactly zero. Only the output bias learns, and the network
predicts the most frequent digit for every image — 11.35% is the share of 1s in
the test set.

The constant is subtler, and more instructive. The weights *do* move — the
largest ends near 0.66 — but all 256 units move together. The first layer's
weight matrix has rank 1: 256 copies of one unit, and a network of width one
reaches 25.9%.

The fix is randomness, and any random draw breaks the symmetry. What the
distribution *is* matters for the second job.

---

## Job two: set the scale

One unit of a `Linear` layer sums $n_{in}$ products. With zero-mean weights
drawn independently of zero-mean inputs,

$$
\operatorname{Var}(y) = n_{in}\,\operatorname{Var}(w)\,\operatorname{Var}(x)
$$

so each layer multiplies the variance of the signal by $n_{in}\operatorname{Var}(w)$.
Stack thirty layers and that factor is raised to the thirtieth power: it has to
be 1, or the signal is gone. Setting it to 1 gives the three classic rules:

$$
\operatorname{Var}(w) = \frac{1}{n_{in}}\ \text{(LeCun)} \qquad
\frac{2}{n_{in} + n_{out}}\ \text{(Xavier)} \qquad
\frac{2}{n_{in}}\ \text{(He)}
$$

Xavier (Glorot and Bengio, 2010) averages the forward and the backward
condition, for symmetric activations like tanh. He (He et al., 2015) doubles
LeCun's variance because a ReLU sets half of its inputs to zero and so halves
the second moment of what it passes on. That factor of two is the whole
difference, and it compounds with depth.

![Weight distributions of a 784-512-256-128-64-32 network under a fixed N(0, 0.1²), Xavier and He](assets/nn/weight_distributions.png)

Read the figure by column. A fixed $\sigma = 0.1$ draws the same distribution in
every layer, whatever its fan-in. Xavier and He widen as the fan-in shrinks, from
$\sigma = 0.051$ for He at 784 inputs to 0.175 at 64: a layer with fewer inputs
needs larger weights to produce an output of the same size.

---

## What the scale does to thirty layers, measured

![Activation and gradient standard deviation across a 30-layer stack for four initializations, measured](assets/nn/signal-and-gradient-std-across-depth-by-init.png)

Thirty `Linear(512, 512)` layers, unit-variance input, one forward pass and one
backward pass — nothing trained. Read the ReLU column on the right. `torch.randn`
with $\sigma = 1$ blows the forward signal to $8.5 \times 10^{11}$ by layer 10 and
`inf` by layer 30. PyTorch's own default decays it to $1.6 \times 10^{-12}$, and
the gradient arriving at layer 1 is $5.3 \times 10^{-12}$. Xavier is better and
still ends at $1.9 \times 10^{-5}$. He holds the signal at 0.63 and the gradient
at 1.12 for the whole depth.

The tanh column is gentler, because tanh is close to linear near zero and keeps
the negative half. Xavier still decays there: tanh squashes what it passes on, so
it needs a gain above 1 — `nn.init.calculate_gain("tanh")` is $5/3$ — and the
rule without the gain falls short of it. Match the rule to the activation.

---

## Does it matter once training starts? Measured

![Training loss of plain ReLU MLPs at three depths and with BatchNorm, under five initializations, measured](assets/nn/init-scale-by-depth.png)

Plain ReLU MLPs of 2, 8 and 16 hidden layers of width 256 — no normalization,
no skip — trained for 1,500 steps of SGD with momentum 0.9 at 0.01 on 10,000
images, three seeds per initialization; the fourth panel adds a BatchNorm after
every `Linear` of the 16-layer one.

At two layers every initialization trains except $\mathcal{N}(0, 1)$, which
starts at a loss of 1,708 and ends at 77.8% test accuracy. He finishes lowest
(95.3%), the default at 94.3%. At eight layers the figure splits: He (96.2%)
and Xavier (95.7%) train, while the default and $\mathcal{N}(0, 0.01^2)$ never
leave $\ln 10$ — 11.35% — and $\mathcal{N}(0, 1)$ goes to `nan`. At sixteen,
the same split, He ahead of Xavier, 95.6% to 94.5%. The two rules that scale
with the fan-in are the only ones that train a deep plain network at all.

The fourth panel is why modern networks forgive a lot: with BatchNorm after
every layer, every initialization but $\mathcal{N}(0, 1)$ trains, and the tiny
$\mathcal{N}(0, 0.01^2)$ is now the best of them (96.3%). Normalization rescales
each layer's output whatever the scale of its weights, which is most of what
initialization was for.

`AdamW` at 1e-3 softens the same grid, because it divides every step by the
gradient's own scale: at sixteen layers the default still trains, slowly, to
91.6%, and only $\mathcal{N}(0, 0.01^2)$ stays stuck. It cannot undo a scale
that is far too large: its steps are about the learning rate in size, so weights
drawn at $\sigma = 1$ take a thousand steps to come down, and that network ends
at 68.8%.

---

## The loss at step 0 is a free test

At initialization, a classifier over $K$ classes should be unsure: every class
near $1/K$, so the cross-entropy of the first batch is near $\ln K$ — 2.303 for
ten classes. *Measured*, the first batch of each network above:

| Initialization | 2 layers | 8 layers | 16 layers | 16 + BatchNorm |
|---|---|---|---|---|
| PyTorch default | 2.309 | 2.303 | 2.300 | 2.399 |
| Xavier | 2.327 | 2.300 | 2.303 | 2.737 |
| He | 2.375 | 2.328 | 2.493 | 2.752 |
| $\mathcal{N}(0, 0.01^2)$ | 2.303 | 2.303 | 2.303 | 2.313 |
| $\mathcal{N}(0, 1)$ | 1,708 | $2.7 \times 10^{9}$ | $1.3 \times 10^{18}$ | 17.1 |

Print the first loss before anything else. Far from $\ln K$ means the
initialization — or the scale of the inputs — is wrong, and no learning rate
will fix it: a first loss of 1,708 on ten classes means the network is certain,
and wrong, about most of the batch before it has seen a single label. Every
sensible initialization above starts within 0.5 of $\ln 10$.

*Exactly* $\ln K$ deserves a second look too. A first loss of 2.303 to three
decimals means every logit is near zero: the signal vanished on its way up.
That is how the default and $\mathcal{N}(0, 0.01^2)$ started at eight and sixteen
layers, and under SGD neither ever left it. Xavier started there too, at 2.300,
and trained: it is a warning to check the activations, not a verdict.

---

## PyTorch's defaults, layer by layer

You never call an initializer and the network still trains, because every layer
initializes itself in its constructor. Read off torch 2.x:

| Layer | Weights | Bias | Example |
|---|---|---|---|
| `Linear`, `Conv*d` | $\mathcal{U}(\pm 1/\sqrt{n_{in}})$ | $\mathcal{U}(\pm 1/\sqrt{n_{in}})$ | `Linear(784, 512)`: std 0.0206 |
| `Embedding` | $\mathcal{N}(0, 1)$ | — | std 1.0, whatever the dimension |
| `LSTM`, `GRU`, `RNN` | $\mathcal{U}(\pm 1/\sqrt{h})$, every matrix | the same | `LSTM(32, 64)`: bound 0.125 |
| `MultiheadAttention` | Xavier uniform | zeros | `d = 64`: std 0.088 |
| `BatchNorm`, `LayerNorm` | $\gamma = 1$ | $\beta = 0$ | the identity, before training |

The `Linear` rule is `kaiming_uniform_` with a leaky-ReLU slope of $\sqrt{5}$, a
historical setting that works out to a standard deviation of
$1/\sqrt{3\,n_{in}}$ — LeCun's variance divided by three, and He's divided by
six. That is why the default decays through the thirty-layer stack above, and
why it is still fine for the few-layer networks of this course: the training
figure shows where it stops being fine.

```python
lin = nn.Linear(784, 512)
lin.weight.std()        # 0.0206 = 1 / sqrt(3 * 784)
lin.reset_parameters()  # draw again, with the same default rule
```

---

## Choosing, and applying it

| Activation after the layer | Initializer | In PyTorch |
|---|---|---|
| ReLU, LeakyReLU, GELU, SiLU | He | `kaiming_normal_(w, nonlinearity="relu")` |
| tanh, sigmoid, none | Xavier | `xavier_normal_(w, gain=calculate_gain("tanh"))` |
| SELU | LeCun | `kaiming_normal_(w, nonlinearity="linear")` |

`calculate_gain` returns the factor that corrects for the activation: $\sqrt{2}$
for ReLU, $5/3$ for tanh, 1 for a linear layer. Apply the choice once, after
building the model and before the optimizer sees it:

```python
def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)      # visits every submodule, recursively
```

Three rules come with it:

- **`isinstance`, not "every parameter".** `for p in model.parameters():
  nn.init.normal_(p, std=0.02)` also redraws every norm layer's $\gamma$, from 1
  to about 0, which shrinks that layer's output to almost nothing — silently.
  `kaiming_normal_` in the same loop at least fails loudly: it raises
  `ValueError` on the first one-dimensional tensor.
- **`mode="fan_in"` (the default) preserves the forward signal,
  `mode="fan_out"` the backward one.** torchvision's ResNets use `fan_out` for
  their convolutions; for an MLP the two differ by the ratio of the widths.
- **A parameter you create yourself gets no default.** `torch.randn(512, 784)`
  has $\sigma = 1$, twenty times wider than He for that fan-in, and produces
  `nan` in a handful of steps. Scale it by hand:

```python
self.w = nn.Parameter(torch.randn(512, 784) * (2 / 784) ** 0.5)  # He
```
