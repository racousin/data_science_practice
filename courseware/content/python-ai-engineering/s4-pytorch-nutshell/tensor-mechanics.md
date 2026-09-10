# Tensors

The data structure every later lesson uses. Three attributes and a dozen
operations cover the training loop, and many PyTorch errors come down to a
shape, a dtype or a device that is not what the code assumed.

<!-- notes: 9 minutes. Run the snippets live, line by line: a notebook cell
displays only its last value. Most of this is NumPy, which the students used in
Sessions 2 and 3: go fast on creation and indexing. Slow down on the three
slides that pay off in the lab: "The float64 trap" (a crash on the first
forward pass), "Reductions" (which axis disappears) and "Broadcasting bites" (a
silent (100, 100) that trains on the wrong loss). -->

---

## Three attributes

```python
X = torch.randn(256, 12)    # a batch: 256 rows processed together
X.shape                     # torch.Size([256, 12])
X.dtype                     # torch.float32
X.device                    # device(type='cpu')
```

- **shape**: for tabular data, $(n, p)$ as in Session 2, one row per sample and
  one column per feature. A **batch** is a group of rows processed together in
  one training step;
  [lesson 7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end)
  explains why training goes batch by batch.
- **dtype**: the number type of every element: `float32` by default, `int64`
  when the data are integers.
- **device**: the memory the values live in, CPU or GPU
  ([lesson 1](/courses/python-ai-engineering/s4-pytorch-nutshell/course/why-tensors)).

When something fails, print these three first.

---

## Creating tensors

```python
torch.tensor([[1., 2.], [3., 4.]])    # from data
torch.arange(0, 10, 2)                # tensor([0, 2, 4, 6, 8])
torch.manual_seed(0)                  # fixes the draws that follow
torch.randn(2, 3)                     # samples from N(0, 1)
```

- `torch.tensor` copies existing data and infers its dtype (next slide).
  `torch.zeros(2, 3)` and `torch.ones(2, 3)` fill a shape with 0 or 1.
- `randn` draws from a random-number generator. `torch.manual_seed(0)` at the
  top of a notebook makes every draw, initial weights included, repeatable on
  the same machine. Without it, an improvement cannot be told apart from a
  lucky initialisation.
- Across machines, library versions, or CPU against GPU, the last digits can
  still differ: libraries add numbers in different orders, and floating-point
  addition is not associative. `(0.1 + 0.2) + 0.3` is `0.6000000000000001`,
  `0.1 + (0.2 + 0.3)` is `0.6`.

---

## Data types

| dtype | Bytes | Use it for |
|---|---|---|
| `float32` | 4 | features, weights, regression targets: the default |
| `int64` | 8 | class labels and indices |
| `bool` | 1 | masks: `True` or `False` per element, e.g. `x > 2` |
| `float16`, `bfloat16` | 2 | halving memory on a GPU (lesson 1) |

```python
torch.tensor([1, 2]).dtype        # torch.int64: inferred from integers
torch.tensor([1., 2.]).dtype      # torch.float32
torch.tensor([1, 2]).float()      # tensor([1., 2.]): converted
```

The rule for this module: features and regression targets in float32, class
labels in int64. `.float()` converts to float32, `.long()` to int64.

---

## The float64 trap

```python
import numpy as np
X_np = np.random.rand(256, 12)                  # float64 by default
torch.from_numpy(X_np).dtype                    # torch.float64
torch.tensor(X_np, dtype=torch.float32).dtype   # torch.float32
```

NumPy and pandas store decimals as **float64**, and `torch.from_numpy` or
`torch.tensor` keep that dtype. A network's weights are float32, so its first
layer then fails with `mat1 and mat2 must have the same dtype, but got Double
and Float` (Double is float64, Float is float32). From a DataFrame, write
`torch.tensor(df.values, dtype=torch.float32)`.

---

## Indexing and reshaping

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
x[:, 0]                          # tensor([1., 4.]): column 0
x[x > 2]                         # tensor([3., 4., 5., 6.]): a mask
torch.arange(4.).view(-1, 1).shape    # torch.Size([4, 1]): a column
```

- Indexing follows NumPy: `x[0]` is row 0. A bool tensor used as an index, a
  **mask**, keeps the elements where it is `True`, in one flat tensor.
- `view` lays the same values out in a new shape; `-1` means "whatever size
  makes it fit". `view(-1, 1)` turns $n$ values into an $(n, 1)$ column, the
  shape a regression model outputs
  ([lesson 5](/courses/python-ai-engineering/s4-pytorch-nutshell/course/modules-and-optimizers)).
  `.squeeze(1)` goes back to $(n,)$.
- `reshape` does the same as `view`, copying the values when `view` cannot.

---

## Element-wise and matrix products

```python
a = torch.tensor([1., 2., 3.])
a * a                    # tensor([1., 4., 9.]): element by element
a @ a                    # tensor(14.): the dot product 1 + 4 + 9
(X @ torch.randn(12, 64)).shape      # torch.Size([256, 64])
```

- `+`, `-`, `*`, `/`, `**`, `torch.exp`, `.abs()` and `torch.maximum(a, b)`
  (the larger value at each position) all work element by element.
- `@` is the matrix product. The inner sizes must match, and they disappear:

$$
(n \times k)\ (k \times m) \rightarrow (n \times m)
$$

The last line has the shapes of a layer of 64 neurons applied to the batch `X`
of the first slide: 256 rows of 12 features in, 256 rows of 64 values out.
Lesson 5 builds the layer itself.

---

## Reductions: `dim` is the axis that disappears

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
x.sum(dim=0)                 # tensor([5., 7., 9.]): one per column
x.mean(dim=1)                # tensor([2., 5.]): one per row
x.argmax(dim=1)              # tensor([2, 2]): where each row peaks
```

- `dim=0` collapses the rows and leaves one value per column. With no `dim`,
  every axis goes: `x.mean()` is `tensor(3.5000)`.
- `argmax` gives the position of the largest value: for a classifier's scores,
  the predicted class.
- A loss is a reduction: the per-row losses $L(y_i, \hat{y}_i)$ of a batch,
  averaged down to one number.

---

## Broadcasting

Shapes are compared from the right. Equal sizes match; a size of 1, or a missing
dimension, is stretched to the other size without copying. Anything else is an
error. Session 2's standardisation needs no loop:

$$
\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{s_j}
$$

```python
x = torch.tensor([[0., 0.], [1., 10.], [2., 20.]])   # (3, 2)
mean = x.mean(dim=0)                # tensor([ 1., 10.]): shape (2,)
x_std = (x - mean) / x.std(dim=0)   # (3, 2) with (2,): per column
x_std[:, 1]                         # tensor([-1.,  0.,  1.])
```

$\mu_j$ and $s_j$, the mean and standard deviation of column $j$, have no row
index $i$, so the same value serves every row.
`std` divides by $n - 1$; `StandardScaler` divides by $n$.

---

## Broadcasting bites

```python
pred = torch.randn(100, 1)          # a model's output: (n, 1)
y = torch.randn(100)                # a column read from a CSV: (n,)
(pred - y).shape                    # torch.Size([100, 100]): no error
(pred - y.view(-1, 1)).shape        # torch.Size([100, 1])
```

Compared from the right, $(100,)$ counts as $(1, 100)$, and both tensors are
stretched to $(100, 100)$: every target is subtracted from every prediction,
10,000 pairs. Their mean is a number, just the wrong one, and training runs on
it. A loss written by hand gives no warning; `nn.MSELoss`
([lesson 6](/courses/python-ai-engineering/s4-pytorch-nutshell/course/losses))
prints one and carries on. Make the shapes equal before computing a loss, and
let a wrong shape stop the script: `assert pred.shape == y.shape`.

---

## Back to Python and NumPy

```python
t = torch.tensor([1., 2., 3.])
t.mean().item()          # 2.0: a Python float, to print or log
t.numpy()                # array([1., 2., 3.], dtype=float32)
```

- `.item()` turns a one-value tensor into a Python number; on a tensor of
  several values it raises an error.
- `.numpy()` hands the values to NumPy, pandas or a CSV file. It reads CPU memory
  only: a tensor on a GPU is first copied back with `.cpu()`, short for
  `.to("cpu")` (lesson 1): `t.cpu().numpy()`.
- [Lesson 3](/courses/python-ai-engineering/s4-pytorch-nutshell/course/autograd)
  adds one more step, for tensors that track gradients.

---

## Check yourself

1. `x` has shape `(32, 12)`. What are the shapes of `x.mean(dim=0)` and
   `x.mean(dim=1)`?

   **Answer.** `(12,)` and `(32,)`: the axis named by `dim` is the one that
   disappears.

2. Run this. What does it print?

   ```python
   pred = torch.zeros(4, 1)
   y = torch.zeros(4)
   print((pred - y).shape, (pred - y.view(-1, 1)).shape)
   ```

   **Answer.** `torch.Size([4, 4]) torch.Size([4, 1])`. The $(4,)$ target counts
   as $(1, 4)$ and both are stretched; after `view(-1, 1)` the shapes match.

3. `torch.from_numpy(df.values)` on a DataFrame of 12 decimal columns: which
   dtype does it give, and why does a network's first layer then fail?

   **Answer.** `torch.float64`, while the layer's weights are float32. Build the
   tensor with `dtype=torch.float32`.
