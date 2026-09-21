# Essential Layers

A layer is a shape contract with parameters attached. The parameter half is what
gets taught; the shape half is what costs you the afternoon. PyTorch tells you
immediately when a matrix multiplication does not line up, and says nothing at all
when a tensor lines up for the wrong reason.

This lesson is the catalogue a practitioner actually assembles — `Linear`,
`Embedding`, `Conv2d`, pooling, recurrent — and the shape each one accepts and
returns. What goes *between* layers (activations, normalisation, dropout,
initialization, clipping) is
[Making Training Work](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/making-training-work);
what drives them is
[Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules);
how the tensors arrive is
[Data Pipelines and the Training Loop](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/data-pipelines-and-training-loop).

Every shape, dtype and parameter count printed below was read off a real torch 2.x
call, not worked out on paper.

<!-- notes: 30 minutes. They know nn.Linear and nn.Module from the 12h module — do
not re-teach those. Spend the time on shapes: put a wrong Conv2d on screen, run it,
and read the error out loud before fixing it. Convolution itself belongs to Session
5 — stop at the shape contract and say so. The permuted-pixels figure is the one to
dwell on: it is the argument for the MLP they will submit in Lab 4. -->

---

## A layer is a contract

Every layer promises three things: the shape it accepts, the shape it returns, and
the parameters it owns.

| Layer | Input | Output | Parameters |
|---|---|---|---|
| `Linear(d_in, d_out)` | `(..., d_in)` float | `(..., d_out)` | $d_{in}d_{out} + d_{out}$ |
| `Embedding(n, d)` | `(...)` **int64** | `(..., d)` | $nd$ |
| `Conv1d(c_in, c_out, k)` | `(B, c_in, T)` | `(B, c_out, T')` | $k c_{in} c_{out} + c_{out}$ |
| `Conv2d(c_in, c_out, k)` | `(B, c_in, H, W)` | `(B, c_out, H', W')` | $k^2 c_{in} c_{out} + c_{out}$ |
| `MaxPool2d(k)` | `(B, C, H, W)` | `(B, C, H/k, W/k)` | none |
| `LSTM(d_in, h)` | `(B, T, d_in)` | `(B, T, h)` and the final state | $4(d_{in}h + h^2 + 2h)$ |

The `LSTM` row assumes `batch_first=True`, which is **not** the default — see below.

Two layouts dominate, and they disagree about where the channel axis goes.

![The two dominant tensor layouts: (B, C, H, W) for images and (B, T, D) for sequences](assets/nn/tensor-layouts-vision-vs-sequence.png)

**`(B, C, H, W)`** for images — channels *second*. **`(B, T, D)`** for sequences —
features *last*. That asymmetry causes a large fraction of all PyTorch shape errors,
and neither convention is going away: the vision one exists because convolution
kernels want contiguous spatial planes, the sequence one because everything that
consumes a sequence ends in a `Linear`.

---

## Linear: the last axis, and nothing else

`Linear` is the exception to the layout problem. It reads the last dimension and
carries every dimension in front of it through untouched.

![nn.Linear applies one weight matrix to every row of the last axis; the leading dimensions are carried through](assets/nn/linear-acts-on-the-last-dimension.png)

```python
import torch, torch.nn as nn

lin = nn.Linear(3, 5)
lin(torch.randn(2, 4, 3)).shape        # torch.Size([2, 4, 5])
lin(torch.randn(7, 3)).shape           # torch.Size([7, 5])
lin.weight.shape, lin.bias.shape       # (5, 3), (5,)
```

`weight` is stored as `(out_features, in_features)` — transposed relative to the
`x @ W` you would write by hand, because `F.linear` computes `x @ W.T + b`. Print it
once and stop second-guessing it.

The input must be floating point. An integer tensor does not get promoted:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float
```

That error means a label, an index or an unconverted `uint8` image reached a
`Linear`. The fix is `.float()` at the boundary, in the `Dataset`, not in `forward`.

---

## Embedding: a lookup table that learns

For a *category* — a token id, a customer id, a permuted MNIST class — there is
nothing to multiply. `Embedding` stores one row per category and returns the rows you
index.

![nn.Embedding gathers one row of a learned matrix per integer index; unindexed rows get no gradient](assets/nn/embedding-lookup-table.png)

```python
emb = nn.Embedding(num_embeddings=5000, embedding_dim=32, padding_idx=0)
idx = torch.tensor([[4, 17, 0], [9, 9, 0]])      # int64, shape (2, 3)
emb(idx).shape                                    # torch.Size([2, 3, 32])
sum(p.numel() for p in emb.parameters())          # 160000
```

Three facts do all the damage:

- **The index dtype is `int64`.** A float index raises
  `Expected tensor for argument #1 'indices' to have one of the following scalar
  types: Long, Int`.
- **An index outside `[0, num_embeddings)` is an error, never a zero.** On CPU it is
  a clean `IndexError: index out of range in self`; on GPU it is an unreadable
  device-side assert several kernels later. Validate the vocabulary size where the
  ids are built, in the `Dataset`.
- **A row that no batch indexes gets no gradient.** Rare categories train slowly
  because they are seen rarely, not because the layer is doing anything clever.
  `padding_idx=0` goes further and pins row 0 at zero for good.

Dimension is the one hyper-parameter, and it is not sensitive:

| Cardinality | Dimension |
|---|---|
| < 10 | 3–8 |
| ~100 | 8–16 |
| ~1,000 | 16–32 |
| 10,000+ | 32–64 |

Past that it buys little and overfits fast. The same layer serves high-cardinality
tabular columns — the learned alternative to the encodings of Session 2 — and token
ids in a language model in Session 7.

---

## Convolution: one small weight set, reused at every position

A `Conv2d` holds a kernel of $k \times k \times c_{in}$ weights per output channel and
slides it across the image, computing one number per window.

![A 3x3 kernel slid over an MNIST digit, and the feature map it produces](assets/nn/convolution-kernel-and-feature-map.png)

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
conv(torch.randn(8, 3, 224, 224)).shape          # torch.Size([8, 64, 224, 224])
conv.weight.shape                                 # (64, 3, 3, 3)
sum(p.numel() for p in conv.parameters())         # 1792  = 3*3*3*64 + 64
```

Read `weight.shape` as *(out channels, in channels, kernel height, kernel width)*.
Every output channel sees **all** input channels — a `Conv2d` is not per-channel
filtering — and the same 1,792 numbers are used at all 50,176 positions. That reuse
is the whole idea, and Session 5's
[Convolution](https://ml-arena.com/courses/ms2a-machine-learning-practice/s5-computer-vision-1/convolution)
takes it apart properly. Here, only the shape contract matters.

---

## The output size, and the two settings that decide it

$$
H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

![Conv2d output size for a 6x6 input and a 3x3 kernel under three padding and stride settings](assets/nn/conv-output-size-padding-stride.png)

Two settings are worth memorising:

- **`kernel_size=3, padding=1, stride=1` keeps the size.** More generally
  `padding = k // 2` for odd `k`. This is the default of every modern CNN, which is
  why you can stack twenty of them without doing arithmetic.
- **`stride=2` halves it**, and is how a network downsamples without a pooling layer.

`Conv1d` is the same operation over `(B, C, T)` and is a badly under-used baseline
for sensor and signal data: `nn.Conv1d(6, 32, kernel_size=5, padding=2)` maps
`(8, 6, 240)` to `(8, 32, 240)` with 992 parameters. Note the layout — a sequence fed
to a `Conv1d` is `(B, C, T)`, channels second, not the `(B, T, D)` an LSTM wants. One
`.transpose(1, 2)` between them, every time.

---

## What the locality prior buys, and when it is false

Compare the two ways to map a 224×224 RGB image to 64 features:

| | `Linear` on the flattened image | `Conv2d(3, 64, 3, padding=1)` |
|---|---|---|
| Parameters | 150,528 × 64 + 64 = **9,633,856** | **1,792** |
| A pixel moved 3 px right | a different input entirely | the same features, 3 px along |
| Another resolution | rebuild the layer | works unchanged |

Flattening destroys adjacency: pixels (10, 10) and (10, 11) become arbitrary
positions in a 150,528-vector. Convolution *assumes* they are neighbours. That
assumption is worth five thousand times fewer parameters — when it is true.

When it is false, you pay for it. Below, the same MNIST is trained twice: once as
pixels arrive, once with a single fixed permutation applied to all 784 pixel
positions of every image, train and test alike.

![Test accuracy of a small conv net and an MLP on MNIST with natural and with permuted pixels, and the wall clock of each](assets/nn/conv-prior-under-a-pixel-permutation.png)

Three seeds each, 3 epochs, AdamW at 1e-3, batch 128, on 3 CPU cores. On natural
pixels the conv net wins on accuracy — 0.989 against 0.977 — with a quarter of the
MLP's parameters. Permute the pixels and it falls to 0.964, *below* the MLP, which
does not notice the permutation at all (0.977 → 0.976): a `Linear` has no notion of
adjacency to lose. The conv net also cost 86 seconds against 2.

That is exactly the setting of
[1 Minute Permuted MNIST](https://ml-arena.com/viewchallenge/8), where the pixel
positions *are* permuted and `train()` gets 60 seconds. A convolutional model there
pays the full price of the prior and collects none of the benefit — which is why the
reference MLP reaches 0.9834 inside the budget while the pure-numpy benchmark sits at
0.9256. Pick the layer whose assumption matches your data, not the layer with the
best reputation.

---

## Pooling: downsampling with no parameters

![MaxPool2d, AvgPool2d and AdaptiveAvgPool2d applied to the same 4x4 feature map](assets/nn/max-average-and-global-pooling.png)

```python
x = torch.randn(8, 64, 27, 27)
nn.MaxPool2d(2)(x).shape                      # torch.Size([8, 64, 13, 13])
nn.AdaptiveAvgPool2d((1, 1))(x).shape         # torch.Size([8, 64, 1, 1])
nn.Conv2d(64, 64, 3, stride=2, padding=1)(x).shape   # torch.Size([8, 64, 14, 14])
```

`MaxPool2d(2)` on an odd size floors: 27 becomes 13, and two rows and two columns are
simply dropped. The stride-2 convolution on the same input gives 14 — it pads. A
model that mixes the two and then hard-codes a flattened size will break on the first
input whose resolution you did not test.

`AdaptiveAvgPool2d((1, 1))` is the one to know by name. It averages each channel over
whatever spatial extent it is given, so the classifier that follows has a fixed input
size of `C` regardless of resolution — the reason a pretrained backbone accepts
224×224 and 384×384 without a code change. Which pooling to use, and why stride-2
convolutions have largely replaced max pooling, is
[Pooling and CNN Architectures](https://ml-arena.com/courses/ms2a-machine-learning-practice/s5-computer-vision-1/pooling-and-architectures)
in Session 5.

---

## Recurrent layers: one cell, applied T times

`RNN`, `LSTM` and `GRU` consume a sequence step by step, carrying a hidden state, and
return the full output sequence **and** the final state.

![A recurrent layer folded and unrolled, with the shapes of a real nn.LSTM call](assets/nn/recurrent-layer-unrolled.png)

```python
lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
out, (h, c) = lstm(torch.randn(8, 100, 32))
out.shape, h.shape, c.shape      # (8, 100, 64), (2, 8, 64), (2, 8, 64)
```

`out` is every step's hidden state, `h` is only the last one — `(num_layers, B, H)`,
with the batch in the *middle* whatever `batch_first` says. For a
sequence-classification head you want `out[:, -1]` or `h[-1]`, and for a
sequence-to-sequence head you want all of `out`.

> **`batch_first=False` is the default.** Without it the layer reads `(8, 100, 32)`
> as 8 time steps of a 100-sample batch, and nothing raises: the output shape is
> `(8, 100, 64)` either way. Only `h` gives it away — `(2, 100, 64)` instead of
> `(2, 8, 64)`. The model trains, converges, and scores plausibly badly.

An `LSTM` layer holds $4(d_{in}h + h^2 + 2h)$ parameters — 25,088 for
`LSTM(32, 64)` — four gates' worth. A `GRU` has three gates, 18,816 for the same
sizes, and is the better first try: fewer parameters, and in practice the same score
on sequences shorter than a few hundred steps.

---

## Assembling: `Sequential`, or a `forward` you wrote

```python
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(32, 10),
)
```

`Sequential` is right for a straight line and wrong for everything else. The moment
you need two inputs, a skip connection or a branch, write `forward`. The mixed
tabular model below — continuous columns straight through, categorical columns via an
embedding, concatenated — has no `Sequential` form at all:

![Shapes through a mixed tabular model: continuous columns and an embedding concatenated, then two Linear layers](assets/nn/shape-flow-embedding-concat-linear.png)

```python
class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 8)
        self.head = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_num, x_cat):                  # (B, 6) float32, (B, 3) int64
        e = self.emb(x_cat).flatten(1)                # (B, 3, 8) -> (B, 24)
        return self.head(torch.cat([x_num, e], dim=1))    # (B, 30) -> (B, 1)

Mixed()(torch.randn(16, 6), torch.randint(0, 50, (16, 3))).shape    # (16, 1)
```

The `30` in `Linear(30, 64)` is `6 + 3 × 8`, and it is the number that breaks when
someone adds a categorical column. Derive it — `n_num + n_cat * emb_dim` — rather
than typing it.

Use `x.flatten(1)`, not `x.view(x.size(0), -1)`. After a `permute` or a
`transpose`, `view` raises
`RuntimeError: view size is not compatible with input tensor's size and stride`,
because the tensor is no longer contiguous. `flatten` and `reshape` handle it.

---

## When the shapes do not line up

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x8192 and 512x128)
```

Read it right to left. `512x128` is the `Linear` you declared. `32x8192` is what
arrived: batch 32, and a flatten that produced 8192 features where you predicted 512.
The bug is upstream, and the fastest fix is not arithmetic — it is printing the
shapes:

```python
x = torch.randn(2, 3, 32, 32)
for name, layer in model.named_children():
    x = layer(x)
    print(f"{name:2s} {type(layer).__name__:18s} {tuple(x.shape)}")
# 0  Conv2d             (2, 32, 32, 32)
# 1  BatchNorm2d        (2, 32, 32, 32)
# 2  ReLU               (2, 32, 32, 32)
# 3  AdaptiveAvgPool2d  (2, 32, 1, 1)
# 4  Flatten            (2, 32)
# 5  Linear             (2, 10)
```

Then count the parameters, on every architecture change:

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total:,} total, {trainable:,} trainable")      # 1,290 total, 1,290 trainable
```

A count that jumps thirty-fold because a pooling layer was dropped takes one second
to read here and forty invisible minutes otherwise. The two numbers differing when
you froze nothing means a submodule sits in a plain Python list instead of an
`nn.ModuleList`, so its parameters are not registered — and are not being trained.

> Then declare the shape you expect, at the top of `forward`, while you are still
> developing.

```python
assert x.ndim == 4 and x.shape[1] == self.in_ch, x.shape
```

Microseconds of runtime, and the message names the actual tensor instead of
surfacing eight layers later as an unreadable matmul complaint.

---

## Check yourself

1. Which layout does vision use, which do sequences use, and which of the layers in
   this lesson does not care?

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   x = torch.randn(8, 100, 32)          # batch 8, 100 steps, 32 features
   for bf in (True, False):
       out, (h, c) = nn.LSTM(32, 64, num_layers=2, batch_first=bf)(x)
       print(bf, tuple(out.shape), tuple(h.shape))
   # -> True (8, 100, 64) (2, 8, 64)
   # -> False (8, 100, 64) (2, 100, 64)
   ```

   The output shape is identical either way. Which line tells you that 100 was read
   as the batch dimension, and what would you see in the loss curve if you never
   looked?

3. Run this too, and explain the ratio.

   ```python
   import torch.nn as nn
   lin  = nn.Linear(150528, 64)
   conv = nn.Conv2d(3, 64, 3, padding=1)
   print(sum(p.numel() for p in lin.parameters()))    # -> 9633856
   print(sum(p.numel() for p in conv.parameters()))   # -> 1792
   ```

   What does the convolution assume about the input that buys the difference, and
   what happens to the two counts if the image is 384×384 instead of 224×224?

4. You feed `(B, 3)` integer category ids into `nn.Embedding(50, 8)` and want to
   concatenate the result with 6 continuous columns before a `Linear`. What are the
   shapes at each step, and what is the `in_features` of that `Linear`?

5. After a `permute`, `x.view(x.size(0), -1)` raises. What is the error, and what do
   you write instead?

6. Your model's parameter count is 203,530 but only 197,000 are trainable, and you
   froze nothing. Where do you look?
