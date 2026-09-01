# Why Tensors

Before the API, the reason it exists. Ten minutes here saves you from writing
Python loops over data that should have been one matrix multiply.

<!-- notes: 25 minutes. The timing demo is worth doing live — run it on the
projector, let them see the two orders of magnitude. -->

---

## The workload

Training a neural network is, almost entirely, repeated matrix multiplication.
One layer:

$$
h = \sigma(W x + b)
$$

with $W$ perhaps $1000 \times 1000$. That is $10^6$ multiply-adds per layer, per
example, per step — times thousands of steps.

Two properties save you: the operations are **identical**, and they are
**independent**. That is exactly what specialised hardware exploits.

---

## Why Python alone cannot do it

```python
# pure Python: ~1000x slower than it needs to be
result = [[sum(A[i][k] * B[k][j] for k in range(n))
           for j in range(n)] for i in range(n)]
```

Every element access goes through the interpreter: type checks, reference
counting, bounds checks. The arithmetic is a rounding error next to the
overhead.

---

## What a tensor is

A **contiguous block of memory, of a single dtype, with a shape**.

That definition is what makes speed possible. Because the type and layout are
known ahead of time, the whole operation is dispatched once to compiled code —
BLAS on CPU, cuBLAS on GPU — instead of a million times through Python.

| | Python list | Tensor |
|---|---|---|
| Element type | anything | one dtype |
| Memory | scattered pointers | one contiguous block |
| Operations | interpreted per element | one compiled kernel |
| GPU | no | yes |

---

## Measure it

```python
import torch, time

n = 1000
A, B = torch.randn(n, n), torch.randn(n, n)

t = time.time(); _ = A @ B; print(f"torch cpu: {time.time()-t:.4f}s")
```

Compare against the triple loop above on `n = 200` — do not try `n = 1000`,
you will be waiting. The gap is roughly three orders of magnitude, and it is the
whole reason this library exists.

---

## Hardware

| | CPU | GPU |
|---|---|---|
| Cores | 8–64, complex | thousands, simple |
| Good at | branching, sequential logic | the same operation on a lot of data |
| Memory bandwidth | ~100 GB/s | ~1000+ GB/s |

A GPU is bad at `if` statements and excellent at "multiply these ten million
numbers". Neural networks are the second thing.

---

## Devices in PyTorch

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
x = torch.randn(1000, 1000, device=device)
```

`mps` is Apple Silicon. Write the selection once, at the top, and pass `device`
everywhere — hard-coding `"cuda"` is how a script fails on half the room's
laptops.

---

## The rule that costs people hours

**Every tensor in an operation must be on the same device.**

```python
a = torch.randn(3, device="cpu")
b = torch.randn(3, device="cuda")
a + b        # RuntimeError: Expected all tensors to be on the same device
```

Move, do not guess:

```python
b = b.to("cpu")      # or a = a.to("cuda")
```

`.to()` **copies** across devices and returns a new tensor. It is not in-place.

---

## Why PyTorch specifically

- **Eager execution** — it runs line by line, so you can `print` a tensor and
  debug with a normal debugger.
- **Autograd** — gradients come for free, which is the next lesson but one.
- **Ecosystem** — Gymnasium, PettingZoo, torchvision, Hugging Face all speak it.

That last one is the practical reason for this course: everything you touch for
the rest of the year takes and returns `torch.Tensor`.

---

## What to take away

1. A tensor is typed, contiguous memory with a shape.
2. That layout is what lets one Python line become one compiled kernel.
3. Loops over tensor elements in Python defeat the entire point.
4. Device mismatches are the most common runtime error you will hit.

If you find yourself writing `for i in range(len(tensor))`, stop — there is a
vectorised way, and the next lesson is about finding it.
