# Tensor Mechanics

Creating, reshaping, and combining tensors. Nearly every PyTorch bug you will
hit this year is a shape bug, so this is the lesson to actually type along with.

<!-- notes: 35 minutes, hands on keyboards throughout. The broadcasting rules
and the matmul shape rule are the two things to drill. -->

---

## Attributes

Four things describe any tensor:

```python
import torch

x = torch.randn(32, 3, 224, 224)

x.shape     # torch.Size([32, 3, 224, 224])
x.dtype     # torch.float32
x.device    # cpu
x.ndim      # 4
```

`(batch, channels, height, width)` — the standard image layout, and the one you
will meet again in *MS2A - Machine Learning Practice*.

---

## Creating

```python
torch.tensor([[1, 2], [3, 4]])     # from data — infers dtype
torch.zeros(3, 4)
torch.ones(2, 3)
torch.arange(0, 10, 2)             # tensor([0, 2, 4, 6, 8])
torch.linspace(0, 1, 5)            # 5 points, 0 to 1 inclusive
torch.eye(3)                       # identity
```

Random:

```python
torch.rand(2, 3)        # uniform [0, 1)
torch.randn(2, 3)       # standard normal
torch.randint(0, 10, (2, 3))
```

---

## Reproducibility

```python
torch.manual_seed(42)
```

Set it at the top of every training script. An unseeded run is not an experiment
— you cannot tell an improvement from a lucky initialisation.

---

## Like-constructors

```python
torch.zeros_like(x)      # same shape, dtype AND device
torch.randn_like(x)
```

Prefer these over `torch.zeros(x.shape)` — they carry the device across, which
removes a whole class of the error from the last lesson.

---

## Data types

| dtype | Bytes | Use |
|---|---|---|
| `torch.float32` | 4 | the default for everything |
| `torch.float16` / `bfloat16` | 2 | mixed-precision training |
| `torch.int64` | 8 | indices, class labels |
| `torch.bool` | 1 | masks |

```python
x = torch.randn(3).to(torch.float64)
y = torch.tensor([1, 2, 3], dtype=torch.float32)
```

**Loss functions want `float32` inputs and `int64` labels.** A dtype mismatch
here produces an error message that does not mention dtypes.

---

## Indexing

Numpy semantics, throughout:

```python
x = torch.randn(4, 5)

x[0]           # first row       -> shape (5,)
x[:, 0]        # first column    -> shape (4,)
x[1:3]         # rows 1 and 2    -> shape (2, 5)
x[x > 0]       # boolean mask    -> 1-D, length varies
x[..., -1]     # last along the final axis
```

Slices are **views** — they share memory with the original. Writing to a slice
writes to the source.

---

## Reshaping

```python
x = torch.arange(12)

x.view(3, 4)        # requires contiguous memory
x.reshape(3, 4)     # copies if it has to — safer
x.reshape(3, -1)    # -1 = "work it out"
```

Use `reshape` unless you have a reason. `view` fails on a non-contiguous tensor
with an error that sends people to Stack Overflow.

---

## Flatten, squeeze, unsqueeze

```python
x = torch.randn(32, 3, 8, 8)

x.flatten(1)          # (32, 192) — keep the batch dim, flatten the rest
x.unsqueeze(0)        # (1, 32, 3, 8, 8) — add a dimension
x.squeeze()           # drop every dimension of size 1
```

`flatten(1)` is how you get from a convolutional stack into a linear layer.
`unsqueeze(0)` is how you feed a single example to a model expecting a batch.

---

## Transpose and permute

```python
x = torch.randn(2, 3, 4)

x.transpose(0, 1)      # (3, 2, 4) — swap two dims
x.permute(2, 0, 1)     # (4, 2, 3) — arbitrary reorder
x.T                    # 2-D only
```

`permute` is what converts between image layouts: `(H, W, C)` from an image
library into `(C, H, W)` for PyTorch.

---

## Element-wise operations

```python
a, b = torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])

a + b        # tensor([5., 7., 9.])
a * b        # tensor([ 4., 10., 18.])  — element-wise, NOT matrix product
a ** 2
torch.exp(a)
torch.sqrt(a)
```

`*` is element-wise. This is the single most common source of silently wrong
results for people arriving from a maths background.

---

## Matrix multiplication

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

A @ B                 # (3, 5) — use this
torch.matmul(A, B)    # identical
```

**The rule:** `(n, k) @ (k, m) -> (n, m)`. The inner dimensions must match and
they disappear.

Batched, which is what actually happens in a network:

```python
A = torch.randn(32, 3, 4)
B = torch.randn(32, 4, 5)
(A @ B).shape          # torch.Size([32, 3, 5])
```

The leading dimensions are treated as a batch and carried through.

---

## Reductions

```python
x = torch.randn(3, 4)

x.sum()                  # scalar
x.sum(dim=0)             # (4,) — collapse rows
x.sum(dim=1)             # (3,) — collapse columns
x.mean(dim=1, keepdim=True)   # (3, 1) — keep the axis
x.max(dim=1)             # (values, indices)
x.argmax(dim=1)          # indices only — this is your prediction
```

`dim=k` means *the dimension that disappears*. Say that to yourself each time
and you will stop guessing.

`keepdim=True` preserves the axis as size 1, which keeps broadcasting working.

---

## Broadcasting

Operations on different shapes work when the shapes are compatible. Compare
dimensions **right to left**; each pair must be equal, or one of them must be 1.

```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)
(a + b).shape        # torch.Size([3, 4])
```

```text
(3, 1)
(1, 4)
------
(3, 4)
```

---

## Broadcasting in practice

Normalising each feature by its column mean:

```python
x = torch.randn(100, 5)
x_centred = x - x.mean(dim=0)          # (100,5) - (5,) -> (100,5)
```

Adding a per-example bias:

```python
bias = torch.randn(100, 1)
x + bias                                # (100,5) + (100,1) -> (100,5)
```

No loops, no copies. This is the vectorised style the whole library assumes.

---

## Broadcasting bites

```python
pred   = torch.randn(100, 1)
target = torch.randn(100)

(pred - target).shape      # (100, 100)  — almost certainly not what you meant
```

`(100,1)` against `(100,)` broadcasts to `(100,100)`, and your loss is the mean
of ten thousand wrong numbers. It does not raise; it just trains badly.

**Fix:** `pred.squeeze()` or `target.unsqueeze(1)`. Assert your shapes.

<!-- notes: This exact bug appears in student RL code every single year. Show it,
name it, and tell them to assert. -->

---

## Numpy interop

```python
import numpy as np

t = torch.from_numpy(np.array([1, 2, 3]))   # shares memory
a = t.numpy()                                # shares memory (CPU only)
a = t.detach().cpu().numpy()                 # the safe incantation
```

`from_numpy` and `.numpy()` **share** the buffer — modifying one modifies the
other. `.detach().cpu().numpy()` is what you want when a tensor might be on a
GPU or carrying gradients.

---

## Debugging shapes

When something breaks, print shapes rather than values:

```python
print(f"{x.shape=}, {w.shape=}, {y.shape=}")
```

Or assert them, which turns a silent broadcast into an immediate crash:

```python
assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"
```

Fail fast: a wrong shape should stop the script, not quietly change your loss.
