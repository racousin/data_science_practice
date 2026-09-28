# Dropout

Dropout switches off a random percentage of a layer's units
on every training step, and switches them all back on to predict. It was
invented for neural networks, it is two lines of PyTorch.


<!-- notes: 30 minutes. Draw the thinned network on the board before the
figure. The F.dropout slide is the one that pays off in their projects. The
BatchNorm slide is honest on purpose: the shift is real and measurable, the
cost on this MLP was nothing — say so. MC dropout is five minutes and an
optional demo. -->

---

## One random sub-network per step

![A two-layer MLP, and the same MLP after dropout has removed two of its five hidden units](assets/nn/d2l-dropout-before-and-after.png)

*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

On the right, units $h_2$ and $h_5$ have been dropped for this step: their
outputs are zero, so the output layer cannot use them and no gradient flows back
through them. The next step draws a new mask and a different sub-network trains.
With $n$ units that can each be on or off there are $2^n$ such sub-networks, all
sharing one set of weights.

```python
self.net = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, 10),                   # never after the output layer
)
```

`nn.Dropout` is a module with no parameters. Its behaviour depends on one flag,
the one `model.train()` and `model.eval()` set.

---

## Inverted scaling: why `eval()` needs no correction

![Five training passes of Dropout(0.5) over twelve units, and the expected activation against the eval output, measured](assets/nn/dropout-masks-and-inverted-scaling.png)

Each unit is zeroed with probability $p$ on every training pass, independently,
and the survivors are multiplied by $1/(1-p)$ — *inverted* dropout. That scaling
is what makes the expected training activation equal the evaluation activation.
