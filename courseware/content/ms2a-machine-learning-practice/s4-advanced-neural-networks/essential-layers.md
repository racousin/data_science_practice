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
