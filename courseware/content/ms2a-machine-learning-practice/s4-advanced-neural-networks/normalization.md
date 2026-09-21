# Normalization

The gradient that reaches the first layer of an $L$-layer network is a product
of $L$ Jacobians, one per layer. Multiply numbers slightly below 1 thirty times
over and you get zero; slightly above 1 and you get `inf`.
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization)
sets that product near 1 at step 0. Normalization keeps it there once the
weights start to move: every layer's input is rescaled to zero mean and unit
variance, whatever the layers before it did.

This lesson is the two normalization layers a practitioner uses — BatchNorm and
LayerNorm — the one axis that separates them, and what BatchNorm costs you when
the batch is small.
[Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections)
takes it from there: where the norm goes inside a residual block.

Every figure marked *measured* comes from a run made while writing this lesson,
on scikit-learn's 1 797-image digits set, small enough to reproduce on a laptop
in under a minute.

<!-- notes: 20 minutes. Draw the axis figure on the board before showing it:
the whole lesson is which axis the mean is taken over. Run check-yourself
question 2 live — the train/eval mismatch on eight samples is the demo. -->

---

## BatchNorm and LayerNorm

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
*first*, inside the residual block, for the reason given in Skip Connections. `BatchNorm1d`
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

## Where the norm goes

> `Linear(bias=False) → BatchNorm → ReLU → Dropout`, He-initialized, wrapped
> in a residual once you pass ~15 layers. LayerNorm instead of BatchNorm for
> sequences, for RL, and for anything scored one sample at a time.

Every choice in that line is this lesson or one of
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization),
[Skip Connections](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/skip-connections)
and
[Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout).
Lab 4 puts it against a deadline: on the
[1 Minute Permuted MNIST](https://ml-arena.com/viewchallenge/8) challenge a
pure-numpy softmax regression scores 0.9256, and a small ReLU MLP — He
initialization, BatchNorm, no dropout, AdamW, gradient clipping — reaches 0.9834
inside the same 60-second budget.
