# Essential Layers

A layer is a shape contract with parameters attached. Most of the time you lose
to a network is spent on the shape half, not the parameter half.

<!-- notes: 30 minutes. They know nn.Linear and nn.Module from the 12h module —
do not re-teach those. Spend the time on shapes: put a wrong Conv2d on screen,
run it, and read the error out loud before fixing it. -->

---

## A layer is a contract

Every layer promises three things: the shape it accepts, the shape it returns,
and the parameters it owns.

| Layer | In | Out |
|---|---|---|
| `Linear(d_in, d_out)` | `(..., d_in)` | `(..., d_out)` |
| `Conv1d(c_in, c_out, k)` | `(B, c_in, T)` | `(B, c_out, T')` |
| `Conv2d(c_in, c_out, k)` | `(B, c_in, H, W)` | `(B, c_out, H', W')` |
| `LSTM(d_in, h)` | `(B, T, d_in)` | `(B, T, h)` |
| `Embedding(n, d)` | `(B, T)` int64 | `(B, T, d)` |

Two conventions dominate: **`(B, C, H, W)`** for images, **`(B, T, D)`** for
sequences. Channels second for vision, last for sequences — an asymmetry that
causes a large fraction of all PyTorch shape errors. `Linear` is the exception
that acts on the last dimension only, whatever precedes it.

---

## Convolution

A kernel slid across the input, sharing weights at every position.

```python
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
conv(torch.randn(8, 3, 224, 224)).shape    # (8, 64, 224, 224)
```

Output spatial size, floored:

$$
H_{out} = \frac{H + 2p - k}{s} + 1
$$

`padding = k // 2` with `stride=1` leaves the size unchanged — which is why
`kernel_size=3, padding=1` is the default of every modern CNN.

---

## Why convolution beats a Linear on images

| | Linear on flattened 224×224×3 | `Conv2d(3, 64, 3)` |
|---|---|---|
| Parameters | 150,528 × 64 ≈ 9.6 M | 3·3·3·64 + 64 = 1,792 |
| Translation invariance | learned from scratch | built in |
| Accepts another resolution | no | yes |

Flattening destroys adjacency: pixels (10,10) and (10,11) become arbitrary
positions in a 150k vector. Convolution assumes they are neighbours — Session 5
takes this apart properly. `Conv1d` is the same idea over `(B, C, T)`, and a
badly under-used baseline for sensor data.

---

## Recurrent layers

`RNN`, `LSTM` and `GRU` consume a sequence step by step, carrying a hidden state.
They return the full output sequence **and** the final state.

```python
lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
out, (h, c) = lstm(torch.randn(8, 100, 32))
out.shape, h.shape          # (8, 100, 64), (2, 8, 64)
```

**`batch_first=False` is the default.** Without it the layer expects `(T, B, D)`,
silently treats your batch dimension as time, and trains to a plausible-looking
mediocre score. Always pass `batch_first=True`.

An LSTM layer holds $4(d_{in}h + h^2 + 2h)$ parameters, four gates' worth. GRU has
three gates and is the better first try.

---

## Embedding

A lookup table with gradients: row $i$ of a learned matrix, for integer category
$i$. It replaces one-hot encoding above a handful of levels.

```python
emb = nn.Embedding(num_embeddings=5000, embedding_dim=32, padding_idx=0)
emb(torch.tensor([[4, 17, 0], [9, 9, 0]])).shape    # (2, 3, 32)
```

Input must be `int64` indices inside `[0, num_embeddings)`. An index one past the
end raises a clear `IndexError` on CPU and an unreadable device-side assert on
GPU, so validate the vocabulary size in your `Dataset`, not in the model.

| Cardinality | Dimension |
|---|---|
| < 10 | 3–8 |
| ~100 | 8–16 |
| ~1,000 | 16–32 |
| 10,000+ | 32–64 |

Past this it buys little and overfits fast. The same layer serves high-cardinality
tabular columns — an alternative to the encodings in Session 2 — and token ids in
a language model (Session 7).

---

## Normalization, dropout, pooling

```python
nn.BatchNorm2d(64)               # (B, 64, H, W) — normalises across the batch
nn.LayerNorm(256)                # normalises the last dimension, per sample
nn.Dropout(p=0.3)
nn.MaxPool2d(2)                  # halves H and W
nn.AdaptiveAvgPool2d((1, 1))     # any H, W -> 1, 1
```

The first three behave differently under `train()` and `eval()`; the next lesson
covers when to reach for which. `AdaptiveAvgPool2d((1, 1))` before the classifier
fixes the head's input size at `C` regardless of resolution, which is why a
pretrained backbone accepts 224×224 or 384×384 without a code change.

Use `x.flatten(1)`, not `x.view(x.size(0), -1)`: `view` raises on the
non-contiguous tensor a `permute` leaves behind.

---

## Reading a shape error

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x8192 and 512x128)
```

Read it right to left. `512x128` is the `Linear` you declared; `32x8192` is what
arrived — batch 32, and a flatten producing 8192 features where you predicted 512.
The bug is upstream, and the fastest fix is not arithmetic:

```python
x = torch.randn(2, 3, 32, 32)
for name, layer in model.named_children():
    x = layer(x)
    print(f"{name:12s} {tuple(x.shape)}")
```

---

## Sequential or a custom forward

```python
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(32, 10),
)
```

`Sequential` is right for a straight line and wrong for everything else. The
moment you need a skip connection, two inputs or a branch, write `forward`.

```python
def forward(self, x_num, x_cat):
    e = self.emb(x_cat).flatten(1)
    return self.head(torch.cat([x_num, e], dim=1))
```

A mixed tabular model — continuous columns straight through, categoricals via
embeddings, concatenated — has no `Sequential` form.

---

## Count, then assert

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
```

Print this on every architecture change: a count that jumps 30× because a pooling
layer was dropped takes one second to read here and forty invisible minutes
otherwise. The two differing when you froze nothing means a module sits in a
plain Python list instead of an `nn.ModuleList`.

> Then declare the shape you expect, at the top of `forward`, while you are still
> developing.

```python
assert x.ndim == 4 and x.shape[1] == self.in_ch, x.shape
```

Microseconds of runtime, and the error names the actual tensor instead of
surfacing eight layers later as an unreadable matmul complaint.

---

## Check yourself

1. Which layout does vision use, which do sequences use, and which layer does
   not care?

   **Answer.** `(B, C, H, W)` for images — channels second — and `(B, T, D)`
   for sequences, channels last. `Linear` is the exception: it acts on the last
   dimension only, whatever precedes it.

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

   The output shape is identical either way, which is why forgetting
   `batch_first=True` never raises. Only `h` shows that 100 was read as the
   batch dimension.

3. After a `permute`, `x.view(x.size(0), -1)` raises. What is the error, and
   what do you write instead?

   **Answer.** `RuntimeError: view size is not compatible with input tensor's
   size and stride` — `permute` leaves a non-contiguous tensor. Use
   `x.flatten(1)`.
