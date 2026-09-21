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
