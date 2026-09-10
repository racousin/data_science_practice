# Tensors

The data structure of pytorch. Three attributes and a dozen
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
  one column per feature.

- **dtype**: the number type of every element: `float32` by default, `int64`
  when the data are integers.
- **device**: the memory the values live in, CPU or GPU.

When something fails, print these three first.

---

## Creating tensors

```python
torch.tensor([[1., 2.], [3., 4.]])    # from data
torch.arange(0, 10, 2)                # tensor([0, 2, 4, 6, 8])
torch.manual_seed(0)                  # fixes the draws that follow
torch.randn(2, 3)                     # samples from N(0, 1)
torch.zeros(2, 3)
torch.ones(2, 3)

```

---


## Data Type

![int.png](assets/s4-pytorch-nutshell/tensor-mechanics/int.png)

![float.png](assets/s4-pytorch-nutshell/tensor-mechanics/float.png)



---

## Precision consequences


![dtypememory.png](assets/s4-pytorch-nutshell/tensor-mechanics/dtypememory.png)

![float_error.png](assets/s4-pytorch-nutshell/tensor-mechanics/float_error.png)

0.1 is not representable in binary. Each type stores a slightly different neighbour: float64 → 0.1000000000000000055…, float32 → 0.10000000149…, float16 → 0.09997558593…. So there is a bias before any addition happens.

0.1 + 0.1 + ... (10,000 terms)



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


- `view(-1, 1)` turns $n$ values into an $(n, 1)$ column, the
  shape a regression model outputs.
- `.squeeze(1)` goes back to $(n,)$.

---

## Element-wise - matrix products - Reductions

```python
a = torch.tensor([1., 2., 3.])
a * a                    # tensor([1., 4., 9.]): element by element
a @ a                    # tensor(14.): the dot product 1 + 4 + 9
```

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
x.sum(dim=0)                 # tensor([5., 7., 9.]): one per column
x.mean(dim=1)                # tensor([2., 5.]): one per row
x.argmax(dim=1)              # tensor([2, 2]): where each row peaks
```
