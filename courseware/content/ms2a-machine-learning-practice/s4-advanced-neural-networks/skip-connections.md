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


<!-- notes: 35 minutes. The two live demos: the 24-block pair (the plain one sits
at 2.303 on screen and nobody forgets it) and deleting one block from each
trained network. The depth sweep is the ResNet paper in one figure — say so. The
scale and pre/post-norm slides are for the students who will build transformers
in Session 6; keep them brisk. -->

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
