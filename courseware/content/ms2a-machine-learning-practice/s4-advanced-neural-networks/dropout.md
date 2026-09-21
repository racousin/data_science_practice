# Dropout

Dropout switches off a random half — or a tenth, or a third — of a layer's units
on every training step, and switches them all back on to predict. It was
invented for neural networks, it is two lines of PyTorch, and it is easy to use
in the wrong place.

Measured against the other regularizers on the same 2,000 images, it gained 1.1
points, behind every regularizer that changes the data. This lesson is why it works, how much of
it to use and where, the bug that makes it silently wrong, the placement that
can, and the one thing it gives you that no other regularizer does — an estimate
of its own uncertainty.

Every figure marked *measured* comes from a run made while writing this lesson,
on MNIST, in torch 2.x on a laptop CPU, in one setting: two
hidden ReLU layers, 2,000 training images, AdamW at 1e-3, three seeds.

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
is what makes the expected training activation equal the evaluation activation:
in the right panel, the mean of 20 000 training passes matches the `eval()`
output to within 0.009 on every unit. Evaluation therefore needs no correction at
all; `nn.Dropout` is exactly the identity once you call `model.eval()`.

$$
\tilde{h}_i = \frac{m_i}{1-p}\,h_i,\quad m_i \sim \text{Bernoulli}(1-p)
\qquad\Longrightarrow\qquad
\mathbb{E}[\tilde{h}_i] = h_i
$$

The original papers (Hinton et al., 2012; Srivastava et al., 2014) scaled the
other way — no rescaling in training, weights multiplied by $1-p$ at test time.
Same expectation, but every inference path had to remember to do it, which is
why every framework now does the inverted version.

---

## Why it regularizes: an ensemble you never have to average

No unit can rely on a particular other unit being present, so none can learn a
feature that is only useful in combination with one specific partner — Hinton's
"co-adaptation". Each unit has to be useful alongside whichever others happen to
survive.

The other reading is an ensemble. Each step trains one of the $2^n$
sub-networks; evaluation with all units on and inverted scaling approximates the
*average* of all of them in one pass. That approximation can be checked: keep
dropout on at test time, run the test set through $T$ random masks, and average
the probabilities.

![Test accuracy of the average of T random dropout masks against the single eval() pass, and accuracy after rejecting the least certain predictions, measured](assets/nn/dropout-ensemble-and-uncertainty.png)

Width 1024, `Dropout(0.5)` after both hidden layers, 2,000 images, three seeds.
A single random mask scores 92.5% on average; averaging the probabilities of
five masks, 93.5%; of twenty, 93.8% — and the one `eval()` pass scores 93.8%.
The weight-scaling rule is as accurate as averaging twenty sub-networks, at the
cost of one: that is why evaluation is done that way.

It matches the average on *accuracy*, not on *confidence*. The test log-loss of
the `eval()` pass is 0.54; the average of fifty masks gets 0.29, half of it. The
single pass is more confident than the average it approximates, and
cross-entropy charges every confident mistake heavily — which matters as soon as
someone reads the probabilities rather than the argmax.

---

## How much, measured

![Validation accuracy against the dropout rate for networks of width 64, 256 and 1024, measured](assets/nn/dropout-rate-by-width.png)

Two hidden layers at three widths, six rates, 2,000 images, three seeds, chosen
on validation. Every width gains from some dropout, and none gains much: the best
rate is worth 1.0 point at width 64 ($p = 0.2$), 1.7 at width 256 ($p = 0.5$) and
1.3 at width 1024 ($p = 0.3$), against the 4 points data augmentation bought on
the same images.

The widths disagree about *how much*. The narrow network is best at 0.2 and at
0.7 falls below no dropout at all, 90.1% against 90.9%; the two wider ones still
gain more than a point at 0.7. Dropout removes capacity on every step, and a
network with little to spare pays for it. Width and dropout go together rather
than instead of each other: the widest network at its best rate, 94.3%, is 2.5
points above the narrowest at its best.

Start at 0.1–0.3 for a new network, higher only for wide layers and a real gap.

---

## Where it goes

| Where | Rate | Why |
|---|---|---|
| after the activation of a wide fully-connected layer | 0.3–0.5 | the parameters are there, and so is the overfitting |
| on the input features | 0–0.2 | a mild noise; higher destroys the signal |
| convolutional feature maps | `Dropout2d`, 0.1 | drops whole channels; neighbouring pixels are too correlated for per-pixel masks |
| transformer: attention weights, residual branch | 0.1 | the standard since 2017 |
| large-scale pre-training | 0 | a model that sees each example about once has little to memorize |
| after the output layer | never | the loss needs every logit |
| before a normalization layer | avoid | the variance shift, below |
| narrow bottlenecks, small networks | none | there is no spare capacity to drop |

Two consequences come with any placement. Dropout always **slows convergence** —
you are training a different sub-network every step — and it puts a **floor under
the training loss**, which is why the overfit-one-batch test of
[Monitoring and Debugging](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging)
is run with dropout off.

---

## Dropout before BatchNorm: the shift, measured

In training, dropout's zeros and its $1/(1-p)$ scaling make the signal leaving
it *more variable* than the same signal in `eval()`, where dropout is off.
A BatchNorm downstream accumulates its running variance during training — from
the noisy version — and applies it at evaluation to the quiet one.

`Linear → ReLU → Dropout(p) → Linear → BatchNorm → ReLU → Linear` on 10,000
images, three seeds, against the same network with dropout moved after the last
BatchNorm. With dropout before, the variance BatchNorm meets in `eval()` is 17%
*below* the running estimate it stored in training at $p = 0.5$, and 4.5% below
at $p = 0.2$ — the `Linear` between them does not remove the shift. With dropout
after, the two agree to within 1–2%.

On this network the shift cost nothing: 96.9% test accuracy with dropout before
at $p = 0.5$, 96.2% after, and re-estimating the statistics with dropout off
moved none of the four by more than 0.3 points. Li et al. (2019), who named the
effect, measured real drops in deep convolutional networks, where the shift
repeats at every block and compounds. So treat it as a placement rule that
costs nothing, not an emergency: put dropout after the last normalization of a
block, as in `Linear → BatchNorm → ReLU → Dropout`, and if you inherit the other
order, check whether it matters by re-estimating the statistics with dropout
off and comparing:

```python
bn.reset_running_stats(); bn.momentum = None   # cumulative average
model.eval(); bn.train()                       # dropout off, BN collecting
with torch.no_grad():
    for xb, _ in train_loader:
        model(xb)
model.eval()
```

---

## The bug: `F.dropout` is on by default

The functional form takes the mode as an argument, and its default is
`training=True`:

```python
h = F.relu(self.fc1(x))
x = F.dropout(h, 0.5)                    # always on — the bug
x = F.dropout(h, 0.5, self.training)     # follows train() and eval()
```

`model.eval()` sets `self.training` on every module; it cannot reach an argument
you did not pass. *Measured* on the two-hidden-layer network of this
lesson, three seeds: with the bug, two `eval()` passes over the same five images
give different outputs, and test accuracy is 91.3% against 93.0% for the same
network written correctly — every prediction is made by one random sub-network
instead of the average of all of them.

Use the module, `nn.Dropout`, and the question never comes up.

---

## Monte Carlo dropout: uncertainty for free

The same ensemble gives more than an average. If the $T$ masks disagree about an
input, the network is unsure about it. Keep **only** the dropout modules in
training mode — `model.train()` would also switch BatchNorm to batch statistics:

```python
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()
with torch.no_grad():
    probs = torch.stack([model(x).softmax(-1) for _ in range(50)])
mean = probs.mean(0)                                  # the prediction
entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)   # its uncertainty
```

The right panel of the ensemble figure uses it to sort the test predictions by
certainty and reject the least certain. Rejecting 10% of the test set lifts the
accuracy on the rest from 93.8% to 97.7%; rejecting 20%, to 98.9%. On this data
the `eval()` pass's own top probability sorts the predictions just as well until
about a third are rejected, so for triage in-distribution, the single pass is
enough.

MC dropout earns its cost on inputs unlike the training data. Fed Fashion-MNIST
images — shoes and shirts — the digit model's top softmax probability averages
0.89: it is confidently wrong about things it has never seen. Told apart from
real digits by that probability, the two sets separate with an AUROC of 0.82;
by the MC entropy, 0.85. Better, not solved: an uncertainty estimate is not an
out-of-distribution detector.

---

## The variants

| Variant | What is dropped | In PyTorch | Use |
|---|---|---|---|
| Dropout | single units | `nn.Dropout` | fully-connected layers |
| Spatial dropout | whole channels | `nn.Dropout1d/2d/3d` | after a convolution |
| DropPath, stochastic depth | a whole residual branch, per sample | a few lines, see [Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections) | ViTs, ConvNeXt, 0.1–0.4 |
| Attention dropout | attention weights | `nn.MultiheadAttention(dropout=0.1)` | transformers |
| DropConnect | single weights | not built in | rarely used |
| AlphaDropout | units, keeping SELU's statistics | `nn.AlphaDropout` | SELU networks only |

`Dropout2d` needs a 4-D input `(B, C, H, W)`; on a 3-D sequence tensor use
`Dropout1d` with channels in dimension 1. DropPath is the variant most current
vision architectures train with, and it exists only because a residual block can
be removed without breaking the network.
