# What PyTorch Is, and Why

PyTorch is an open-source Python library that does two jobs: it runs
arithmetic on large arrays fast, on whatever hardware (cpu/gpu), and it
computes the gradients that training needs.

<!-- notes: 12 minutes. The argument, not the API: the workload is matrix
multiplication, it is parallel, the hardware that exploits it has its own
memory, and the gradients have to come without derivations by hand. One live
moment is worth it: the pure-Python product against `A_t @ B_t` (about 0.3 s
against 15–22 µs). Ask who is on a Mac (`mps`), an NVIDIA machine (`cuda`) or
Colab's default runtime (`cpu`): the same-device slide behaves differently on
each. The GPU slides are conceptual; the lab needs no GPU. Every timing here was
measured on an Apple M4 (torch 2.14.0, 4 threads) on 10 September 2026, with
other jobs running: the ranges are the spread over repeated runs. The four
values of the device-time figure are the figure's own, made on the same machine
by tools/figures/s4_pytorch_nutshell.py (device_time). -->

![Pytorch_logo.png](assets/s4-pytorch-nutshell/why-tensors/Pytorch_logo.png)

---

## The workload

A layer of an MLP multiplies its input by $W^k$, adds $b^k$ and applies
$\sigma$. For many inputs at once, that multiplication is a product
of two matrices, so training is, almost entirely, repeated matrix
multiplication:

$$
C_{ij} = \sum_{m=1}^{n} A_{im} B_{mj}
$$

- **The work is large.** For $n = 1000$: $n^3$ multiply-adds, that is
  $2n^3 = 2 \times 10^{9}$ floating-point operations (FLOP). Speed is counted in
  FLOP per second: the CPU of an Apple M4 sustains about $1.5 \times 10^{12}$
  FLOP/s (measured: 1.3–1.7 ms for this product).
- **The entries are independent.** Each $C_{ij}$ needs only row $i$ of $A$ and
  column $j$ of $B$: a million dot products that could all run at once.

---

## How much work

![Training compute of large models, 2017–2024, log scale (source: Epoch AI)](assets/s4-pytorch-nutshell/why-tensors/flop.jpeg)

Training Llama 3.1 405B (July 2024, not in the figure) took
$3.8 \times 10^{25}$ FLOP. On the M4's CPU:

$$
\frac{3.8 \times 10^{25}\ \mathrm{FLOP}}{1.5 \times 10^{12}\ \mathrm{FLOP/s}} = 2.5 \times 10^{13}\ \mathrm{s} \approx 8 \times 10^{5}\ \mathrm{years}
$$

It was trained in months, on up to 16,000 GPUs, because the work splits.

---

## Why Python alone cannot do it


![speed.png](assets/s4-pytorch-nutshell/why-tensors/speed.png)

```python
n = 200
A_t = torch.randn(n, n)            # a tensor: 200 x 200 random values
B_t = torch.randn(n, n)
A, B = A_t.tolist(), B_t.tolist()  # the same values, as Python lists
```

```python
C = [[sum(A[i][m] * B[m][j] for m in range(n))  # pure Python
      for j in range(n)] for i in range(n)]
C_t = A_t @ B_t                                 # one torch call
```

Measured on an Apple M4: **0.25–0.36 s** for the Python version, **15–22 µs**
for `A_t @ B_t`: 16,000 to 20,000 times faster (eighteen runs).

- Python sends every element through the interpreter (type checks, reference
  counts), one multiplication at a time.
- `@`, the matrix product, is one call into BLAS: a library of compiled
  linear-algebra routines that runs on several cores at once and uses the CPU's
  vector instructions (several numbers per instruction).





---


## What a tensor is

A **block of memory holding numbers of one type (the dtype, e.g. 32-bit
floats), with a shape** such as (200, 200).

| | Python list | Tensor |
|---|---|---|
| Element type | anything | one dtype |
| Memory | scattered objects | one contiguous block |
| Operations | interpreted, per element | one compiled kernel |
| Runs on a GPU | no | yes |


![tensor_layout.jpeg](assets/s4-pytorch-nutshell/why-tensors/tensor_layout.jpeg)


---

## CPU and GPU

![A CPU has a few strong cores; a GPU has thousands of simple ones](assets/s4-pytorch-nutshell/why-tensors/cpu-vs-gpu.png)

| | CPU | GPU |
|---|---|---|
| Cores | 4–64, powerful | thousands, simple |
| Built for | branching logic, low latency | one operation on many numbers |
| Reads its memory at | ~100 GB/s | ~1,000 GB/s |

---

## Two memories, one narrow bridge

![Data goes from disk to CPU memory, is copied to GPU memory, computed on, and copied back](assets/s4-pytorch-nutshell/why-tensors/gpu-workflow.jpg)

A GPU computes only on data in **its own memory** (VRAM), 16–80 GB. Data gets
there over the PCIe link, **~30 GB/s**: often the slowest step.



---

## Devices in PyTorch

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(1000, 1000, device=device)    # created on that device
```


```python
a = torch.randn(3)              # in CPU memory
b = a.to(device)                # a copy on the GPU
print(a.device, b.device)
```


---


## CPU vs GPU

```python
import time

# Without synchronization - misleading timing
start = time.time()
gpu_result = gpu_tensor @ gpu_tensor  # Returns immediately
print(f"Time: {time.time() - start:.6f}s")  # Too fast! Operation still running

# With synchronization - accurate timing
start = time.time()
gpu_result = gpu_tensor @ gpu_tensor
torch.cuda.synchronize()  # Wait for GPU to finish
print(f"Actual time: {time.time() - start:.6f}s")
```



![device-time2.png](assets/s4-pytorch-nutshell/why-tensors/device-time2.png)



---

## The main other reason for PyTorch

Training updates every parameter by gradient descent (Session 2):

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \ell(\theta_t)
$$

- **Gradients, computed for you.** Every step needs $\partial \ell / \partial \theta$
  for every parameter of a complex deep neural network architecure. PyTorch records
  the operations that produced $\ell$ and runs backpropagation in one
  call  without the bookkeeping.


- **The ecosystem.** Most research code uses it; JAX and TensorFlow do the same
  job with other trade-offs.
