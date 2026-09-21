# Skip Connections

The gradient that reaches the first layer of a deep network is a product of one
Jacobian per layer, and
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization)
and
[Normalization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/normalization)
spend their length holding each factor near 1. Past a certain depth that is not
enough. A skip connection changes the product itself: it hands a block's input
past the block, so the signal and the gradient have a path that no layer
multiplies.

This lesson is the mechanism, measured: why the path helps, how deep a network
trains with and without it, what the addition does to the scale of the signal,
where the normalization goes, what to do when the shape changes, and the
concatenating variant of U-Net and DenseNet. The architectures built on it —
ResNet in
[Session 5](https://ml-arena.com/courses/ms2a-machine-learning-practice/s5-computer-vision-1/pooling-and-architectures),
U-Net in Session 6, the transformer in Session 7 — take it as given.

Every figure marked *measured* comes from a run made while writing this lesson,
on MNIST, in torch 2.x on a laptop CPU.

<!-- notes: 35 minutes. The two live demos: the 24-block pair (the plain one sits
at 2.303 on screen and nobody forgets it) and deleting one block from each
trained network. The depth sweep is the ResNet paper in one figure — say so. The
scale and pre/post-norm slides are for the students who will build transformers
in Session 7; keep them brisk. -->

---

## Two ways to hand the input past a block

![A block B whose input is added to its output, and the same block whose input is concatenated to it](assets/nn/d2l-add-vs-concatenate.png)

*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

| | Add: $x + F(x)$ | Concatenate: $[x, F(x)]$ |
|---|---|---|
| Used by | ResNet, every transformer | DenseNet, U-Net |
| Shape after the join | unchanged | wider, by the width of $F(x)$ |
| Parameters of the join | none | none, but the next layer pays for the width |
| What it buys | a gradient path through depth | the next layer sees the raw input and the processed one |

Addition is the one that makes depth trainable, and most of this lesson is about
it. Concatenation comes at the end.

---

## The residual block

![A regular block that learns f(x) directly, and a residual block that learns g(x) = f(x) − x and adds x back](assets/nn/d2l-regular-vs-residual-block.png)

*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

On the left, the two weight layers must learn the whole mapping $f(x)$. On the
right they learn only the **residual** $g(x) = f(x) - x$, and the block adds $x$
back. The two can represent exactly the same functions. What changes is which
function is easy to reach: if the best a block can do is nothing, the residual
block gets there by driving its weights to zero, while the plain block has to
fit an exact identity through two weight layers and a ReLU.

```python
class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))
```

One character of the forward, `x +`, is the difference between the two
networks measured next.

---

## Why adding a block cannot hurt

![Non-nested function classes, where a larger class can be further from the target, and nested ones, where it cannot](assets/nn/d2l-nested-function-classes.png)

*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Read each $\mathcal{F}_i$ as the set of functions a network of depth $i$ can
reach, and $f^*$ as the one you want. On the left, making the network deeper
changes the set without containing the old one, and the deeper network can end
up *further* from $f^*$. On the right each set contains the previous one, so
depth can only move you closer. A residual block is what makes the sets nested:
a new block that sets $g = 0$ returns the previous network exactly.

That is an argument about what is *representable*. Whether the optimizer finds
it is the next two slides.

---

## The gradient has a path that nothing multiplies

$$
y = x + g(x) \quad\Longrightarrow\quad \frac{\partial y}{\partial x} = I + \frac{\partial g}{\partial x}
$$

Unroll it over a stack of blocks and the output of block $L$ is
$x_L = x_\ell + \sum_{i=\ell}^{L-1} g_i(x_i)$ — every block writes into one
running sum, and the gradient of the loss reaches every earlier $x_\ell$ through
the identity term without being multiplied by any weight.

![Gradient reaching each block at initialization, and training loss, for a 24-block stack with and without the identity path, measured](assets/nn/residual-identity-path-gradient-and-loss.png)

Two 24-block networks, identical weights, identical data, identical optimizer.
Each block is `LayerNorm → Linear(128,128) → ReLU → Linear(128,128)`; the only
difference is `x = block(x)` against `x = x + block(x)`. At initialization the
gradient arriving at block 1 is $5.1 \times 10^{-5}$ in the plain stack and
$5.2 \times 10^{-4}$ with the skip — ten times larger, and flat across depth
instead of decaying toward the input. After 600 steps of AdamW the residual
network is at a training loss of $2 \times 10^{-4}$ and 97.1–97.8% test accuracy
over three seeds; the plain one sits at 2.30 — $\ln 10$, the loss of guessing —
and 10% accuracy, on data a two-layer MLP solves to 97%.

---

## The degradation problem, measured

![Training loss and test accuracy against depth for plain and residual stacks, each at its best of three learning rates, measured](assets/nn/residual-depth-sweep.png)

Plain and residual stacks of 1 to 32 blocks — each block a pre-norm
`LayerNorm → Linear(128,128) → ReLU → Linear(128,128)` — trained for 2,000
steps of AdamW on 10,000 MNIST images, each depth at the best of three learning
rates (3e-4, 1e-3, 3e-3) on validation, three seeds.

The plain stack gets worse at *training* as it gets deeper. Its final training
loss goes from $3.7 \times 10^{-4}$ at one block to 0.037 at eight and 0.19 at
sixteen, and at 32 blocks no learning rate moves it off $\ln 10$: 11.35% test
accuracy, which is the share of the most common digit. The residual stack's
training loss stays between $2.5 \times 10^{-3}$ and 0.018 at every depth, and its
test accuracy between 95.7% and 96.4% from one block to 32.

At one block the plain network is the better of the two, by a point. From two
blocks on it is worse, and the gap widens with depth: 0.5 points at four, 1.3 at
eight, 2.5 at sixteen. Depth buys nothing on MNIST — one block does as well
as 32 — which is what makes the experiment clean: there was nothing to gain, so
everything the plain network loses with depth is lost to optimization.

This is the ResNet result in miniature, and the point of the 2015 paper: a
plain 56-layer network had *higher training* error than a plain 20-layer one.
The failure is optimization, not generalization — the plain network is not
overfitting, it cannot fit — and the identity path fixes it.

---

## A residual network behaves like many shallow ones

Delete one block of a trained network at test time and pass its input straight
through. In a plain network that removes a stage of a pipeline that every later
layer depends on. In a residual network it removes one term of the running sum.

![Test accuracy after deleting each single block of a trained plain and residual network, measured](assets/nn/residual-delete-one-block.png)

The two 8-block networks of the depth sweep, each at its best learning rate,
three seeds. The plain one scores 95.1% whole. Delete any one of its eight blocks
and it falls to between 6% and 11% on average — around the 10% of guessing:
every later layer was fitted to the representation the missing block produced.
The residual one scores 96.4% whole. Delete any of its last four blocks and it
loses less than half a point; block 4 costs 0.8, blocks 3 and 2 cost 2.4 and 4.0.
Only the first block is expensive — 78.8% without it: it writes the most into
the stream, and every later block builds on what it wrote.

Veit, Wilber and Belongie (2016) made the same measurement on ImageNet ResNets
and read it this way: the $2^L$ paths through a residual network, each taking or
skipping each block, behave like an ensemble of networks of many depths, most of
them short. Removing a block removes half the paths and leaves the rest working.

---

## The sum grows: keep it in check

Every residual block *adds* to the stream. If each branch adds roughly
independent signal of variance $v$, after $L$ blocks the variance is about
$\operatorname{Var}(x_0) + Lv$, and without any normalization in the branch it
compounds instead.

![Standard deviation of the residual stream across 64 blocks at initialization, for three ways of initializing the branch, measured](assets/nn/residual-stream-std-at-init.png)

64 residual blocks of width 256, a batch of MNIST images, no training. With
pre-norm and PyTorch's default initialization, the stream's standard deviation
grows from 0.19 after the stem to 0.96 at block 16 and 1.90 at block 64 — ten
times, and roughly as $\sqrt{L}$, because each branch sees a normalized input
and so adds about the same variance every time. Scaling each branch's last layer
by $1/\sqrt{2L}$ holds it to 0.38. Zero-initializing that layer keeps it at
0.19, exactly the stem's output, because every block starts as the identity.

Without any normalization the growth is multiplicative instead: each block
scales the stream by one plus its branch's gain. With PyTorch's conservative
default that came to 7.4× over the same 64 blocks; with larger weights it is
faster, and it compounds with depth the way the plain stack's product did.

---

## Three ways to hold the sum

Three standard answers, often combined:

- **Normalize the branch's input** (pre-norm, next slide), so each block sees a
  signal of unit scale whatever the stream has grown to.
- **Scale the branch's last layer** by $1/\sqrt{2L}$ at initialization — GPT-2's
  rule for its residual projections, with $L$ the number of blocks.
- **Zero-initialize the branch's last layer**, or the $\gamma$ of its last norm,
  so every block starts as the identity and the network starts as a shallow one
  that grows deeper as training moves the zeros:

```python
nn.init.zeros_(block.fc2.weight)
nn.init.zeros_(block.fc2.bias)       # the block now returns x exactly
```

Zero is safe here and not in a plain MLP because the branch's *inputs* already
differ from unit to unit — see
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization).

---

## Pre-norm or post-norm

$$
\text{post-norm: } x \leftarrow \mathrm{LN}\big(x + g(x)\big)
\qquad
\text{pre-norm: } x \leftarrow x + g\big(\mathrm{LN}(x)\big)
$$

Post-norm is the 2017 transformer. It normalizes the stream itself, after every
addition, so the identity path passes through a LayerNorm at every block and is
no longer an identity. Pre-norm normalizes only what enters the branch and
leaves the stream untouched; it needs one final LayerNorm before the head.

![Final training loss and test accuracy against learning rate for a 24-block residual MLP with pre-norm, post-norm and no norm, measured](assets/nn/pre-norm-vs-post-norm.png)

The same 24-block residual MLP three ways — pre-norm, post-norm, and no
normalization at all — trained without warmup at five learning rates, 1,500
steps, three seeds. Up to `lr = 3e-3` all three train, and at this size none is
clearly ahead: the best test accuracies are 95.8% pre-norm, 96.0% post-norm and
96.1% with no norm. At `1e-2` they separate. Pre-norm still trains, to 94.3%.
Post-norm stalls at $\ln 10$ in all three seeds; with no norm, all three go to
`nan`.

What pre-norm buys here is headroom, not accuracy: it is the only variant that
survives the largest rate. That is the small version of the transformer result
(Xiong et al., 2020), where the gap is much wider — post-norm needs a warmup and
a small rate to get through its first steps, and pre-norm does not.

Most large transformers since GPT-2 are pre-norm, and it is the default
[Session 7](https://ml-arena.com/courses/ms2a-machine-learning-practice/s7-nlp-1/the-transformer)
recommends.
Post-norm is not wrong; it is the variant that needs a learning-rate warmup to
survive its first hundred steps — see
[Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules).

---

## When the shape changes: project the shortcut

The addition needs $x$ and $g(x)$ to have the same shape. A block that halves
the resolution or doubles the channels breaks that, and the fix is to transform
the shortcut too — by the cheapest layer that produces the new shape.

![A ResNet block with an identity shortcut, and the same block with a 1×1 convolution on the shortcut](assets/nn/d2l-resnet-block-and-projection.png)

*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

---

## The projection, in code

```python
class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out))
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.branch(x) + self.shortcut(x))
```

`BasicBlock(64, 128, stride=2)` maps `(8, 64, 32, 32)` to `(8, 128, 16, 16)`.
The branch holds 221,696 parameters and the projection 8,448:
a 1×1 convolution is the cheapest way to change the channel count, and it is the
only reason the block is ever more than two convolutions. Project only where the
shape changes — a projection on every block turns the identity path back into a
stack of layers.

This is the 2015 block, with the ReLU *after* the addition, and it is the one
torchvision's `resnet18` uses. The 2016 follow-up (He et al., *Identity Mappings
in Deep Residual Networks*) moved normalization and ReLU into the branch —
"pre-activation" — so that the shortcut is a pure identity from input to
output, and trained a 1001-layer network that way.

---

## Concatenation: DenseNet and U-Net

`torch.cat([x, g(x)], dim=1)` keeps both instead of summing them. Nothing is
lost, and nothing is free: the width after the join is the sum of the two, and
the layer after it pays for every extra channel.

![U-Net: an encoder that downsamples, a decoder that upsamples, and a concatenation at every resolution](assets/cv/unet-architecture.png)

**U-Net** (Session 6,
[Segmentation](https://ml-arena.com/courses/ms2a-machine-learning-practice/s6-computer-vision-2/segmentation))
concatenates *across* the network: each decoder stage receives the encoder's
feature map at the same resolution, which carries the fine spatial detail that
the downsampling threw away. Without those long skips the decoder has to
reconstruct object boundaries from a map 16 times smaller, and does it badly.

**DenseNet** (2017) concatenates *within* a stage: every layer receives the
feature maps of all the layers before it, and adds $k$ new channels — the growth
rate — to the pile. Its channel count grows linearly with depth, which is why it
alternates dense blocks with "transition" layers — a 1×1 convolution and a
pooling — that cut the width back down.

```python
conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
x = torch.randn(8, 64, 32, 32)
h = conv(x)                          # (8, 32, 32, 32): k = 32 new channels
torch.cat([x, h], dim=1).shape       # (8, 96, 32, 32)
```

---

## Stochastic depth: dropout for whole blocks

Because a residual block can be removed and the network still works, it can
also be removed *during training*: drop the entire branch of a block, for each
sample independently, with probability $p$, and keep only the identity.

```python
def drop_path(h: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p == 0.0:
        return h
    keep = torch.rand(h.shape[0], *[1] * (h.ndim - 1),
                      device=h.device) >= p          # one draw per sample
    return h * keep / (1 - p)

# in the block's forward:
return x + drop_path(self.branch(x), self.p, self.training)
```

It is dropout — the same mask, the same $1/(1-p)$ rescaling, the same identity at
evaluation — applied to a whole block instead of one unit, and it only exists
because of the skip: in a plain network, dropping a layer drops the signal.
Vision transformers and ConvNeXt train with rates of 0.1 to 0.4, usually rising
linearly from the first block to the last. The unit-level version and its rules
are [Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout).

---

## The rules

- **The branch must return the shape it received**, or the addition fails. When
  the width or the stride changes, project the shortcut rather than drop it.
- **Keep the identity path clean.** No normalization, no activation, no dropout
  *on the skip itself*: whatever sits on it multiplies the gradient again.
- **Normalize before the branch, not after** (pre-norm), and put one LayerNorm
  before the head.
- **Start the branch small or at zero**, so a deep network starts as a shallow
  one.
- **Add the skip, then add depth.** Beyond about fifteen weight layers, a
  network without residual connections is a network you are debugging, not
  training.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn, torch.nn.functional as F
   torch.manual_seed(0)
   fc1, fc2 = nn.Linear(16, 16), nn.Linear(16, 16)
   nn.init.zeros_(fc2.weight); nn.init.zeros_(fc2.bias)
   x = torch.randn(4, 16)
   y = x + fc2(F.relu(fc1(x)))
   print(torch.equal(y, x))                       # -> True
   y.sum().backward()
   print(fc1.weight.grad.abs().sum().item())      # -> 0.0
   print(fc2.weight.grad.abs().sum().item() > 0)  # -> True
   ```

   The block is exactly the identity and `fc1` receives no gradient. Is it
   stuck? What changes after one optimizer step, and why does an MLP whose
   *every* layer starts at zero stay stuck instead?

2. The derivative of $y = x + g(x)$ is $I + \partial g/\partial x$. Why does
   that make the gradient at the first block of a 50-block network independent
   of how small the $\partial g_i/\partial x$ of the other 49 blocks are?

3. Your plain 20-layer network reaches a *training* loss of 0.9 and a
   10-layer version of it 0.3. Is the deep one overfitting? What do you add, and
   which figure of this lesson predicts the result?

4. You delete block 5 of a trained 12-block network at test time and accuracy
   drops from 97% to 20%. Was the network residual? What would you expect if it
   were?

5. A block maps `(B, 64, 56, 56)` to `(B, 128, 28, 28)`. Write the shortcut, and
   say why a 3×3 convolution there would be a worse choice than a 1×1.

6. Why can stochastic depth drop a whole block of a ResNet but not a whole layer
   of a plain MLP?
