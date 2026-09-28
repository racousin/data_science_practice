# Normalization

The gradient that reaches the first layer of an $L$-layer network is a product
of $L$ Jacobians, one per layer. Multiply numbers slightly below 1 thirty times
over and you get zero; slightly above 1 and you get `inf`.
[Parameter Initialization](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/initialization)
sets that product near 1 at step 0. Normalization keeps it there once the
weights start to move: every layer's input is rescaled to zero mean and unit
variance, whatever the layers before it did.

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

| | BatchNorm | LayerNorm |
|---|---|---|
| Statistics over | the batch, per feature | one sample, per sample |
| Depends on batch size | yes | no |
| `train()` ≠ `eval()` | yes | no |
| Extra state | running mean and variance | none |
| Sequence-friendly | needs `(B, C, T)` | native on `(B, T, D)` |
| Standard in | CNNs | transformers, RNNs, RL |
