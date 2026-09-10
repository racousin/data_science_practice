# What PyTorch Is, and Why

PyTorch is an open-source Python library that does two jobs: it runs
arithmetic on large arrays fast, on whatever hardware is present, and it
computes the gradients that training needs. This lesson gives the reason for
both; the API starts in the next one.

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

---

## The workload

A layer of an MLP multiplies its input by $W^k$, adds $b^k$ and applies
$\sigma$ (Session 3). For many inputs at once, that multiplication is a product
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

Type and layout are known in advance, so one operation on a whole tensor runs
as one **kernel**, a compiled routine (BLAS on a CPU, cuBLAS on an NVIDIA GPU),
instead of a million steps through Python.

| | Python list | Tensor |
|---|---|---|
| Element type | anything | one dtype |
| Memory | scattered objects | one contiguous block |
| Operations | interpreted, per element | one compiled kernel |
| Runs on a GPU | no | yes |

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

## What fills the memory

- **Bytes per value**: 4 in float32, 2 in float16. A million float32 values
  take 4 MB; in float16, 2 MB.
- **The model.** 7 billion parameters are 28 GB in float32. Training with
  Adam, an optimizer that keeps two running averages per weight
  ([lesson 4](/courses/python-ai-engineering/s4-pytorch-nutshell/course/optimizers)),
  holds four copies (weights, gradients, the two averages): 112 GB.
- **The batch**: the rows processed together in one training step
  ([lesson 7](/courses/python-ai-engineering/s4-pytorch-nutshell/course/training-loop-end-to-end)),
  and everything computed from them.
- Weights, gradients, optimizer state and the batch must all fit in VRAM at
  once: that is what caps the batch size.

---

## Measured, not promised

![Matrix multiplication time against matrix size on the CPU and the GPU of an Apple M4, log scale](assets/s4-pytorch-nutshell/why-tensors/device-time.png)

A GPU wins only when the work outweighs the copy and the kernel launch (starting
a kernel on the GPU). On the Apple M4 of the figure, the GPU takes 0.2 ms for a
$100 \times 100$ product the CPU does in 0.003 ms, and 46 ms for a
$4000 \times 4000$ product that takes the CPU 85 ms.
[Measure it yourself in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s4-cpu-gpu-benchmark.ipynb).

---

## Devices in PyTorch

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(1000, 1000, device=device)    # created on that device
```

- `cuda`: NVIDIA GPUs, including Colab's GPU runtime. `mps`: the GPU of Apple
  silicon; it shares the CPU's memory, but PyTorch still treats it as a
  separate device. `cpu`: everywhere, including Colab's default runtime.
- Choose once, at the top, and pass `device` everywhere. Hard-coding `"cuda"`
  is how a script fails on half the room's laptops.

---

## The same-device rule

```python
a = torch.randn(3)              # in CPU memory
b = a.to(device)                # a copy on the GPU, if there is one
print(a.device, b.device)
```

- On a GPU machine this prints `cpu mps:0` on the M4 (`cpu cuda:0` with an
  NVIDIA GPU), and `a + b` raises a `RuntimeError`: "Expected all tensors to be
  on the same device, but found at least two devices, mps:0 and cpu!". PyTorch
  never copies behind your back: the copy is the expensive part.
- On a CPU-only runtime `device` is `"cpu"`: `b` is `a`, no copy is made, the
  line prints `cpu cpu`, and `a + b` runs.
- `tensor.to(device)` returns a **new** tensor: reassign it. A model
  ([lesson 5](/courses/python-ai-engineering/s4-pytorch-nutshell/course/modules-and-optimizers))
  is the exception: `model.to(device)` moves it in place.
- GPU calls return before the work is done. Time them after
  `torch.cuda.synchronize()` (`torch.mps.synchronize()` on a Mac), or you time
  the queueing: on the M4, a $1000 \times 1000$ product "takes" 0.03–0.18 ms
  without the call and 1.0–3.3 ms with it.

---

## Why PyTorch

Training updates every parameter by gradient descent (Session 2):

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \ell(\theta_t)
$$

- **Gradients, computed for you.** Every step needs $\partial \ell / \partial \theta$
  for every parameter: 235,146 of them in a 784-256-128-10 MLP. PyTorch records
  the operations that produced $\ell$ and runs Session 3's backpropagation in one
  call ([lesson 3](/courses/python-ai-engineering/s4-pytorch-nutshell/course/autograd)),
  without the bookkeeping.
- **Tensors on any hardware.** A Python front end over compiled C++ and CUDA
  kernels; the `device` argument moves the work.
- **Ordinary Python.** Code runs line by line, so `print`, `if` and a debugger
  all work.
- **The ecosystem.** Most research code uses it; JAX and TensorFlow do the same
  job with other trade-offs.

---

## Check yourself

1. How many FLOP is one product of two $1000 \times 1000$ matrices? Why can a
   GPU run it in parallel, and why does the M4's GPU lose to its CPU at
   $100 \times 100$?

   **Answer.** $2n^3 = 2 \times 10^{9}$. Every entry $C_{ij}$ is an independent
   dot product. At $100 \times 100$ the work, $2 \times 10^{6}$ FLOP, is too
   small to pay for the kernel launch.

2. Run this. What does it print?

   ```python
   w = torch.zeros(1_000_000)                  # float32
   h = torch.zeros(1_000_000, dtype=torch.float16)
   print(w.element_size(), h.element_size())   # bytes per value
   print(w.numel() * w.element_size())         # values x bytes
   ```

   **Answer.** `4 2`, then `4000000`: 4 bytes per float32 value, 2 per float16
   value, so a million float32 values take 4 MB.

3. On a GPU machine, `x = torch.rand(3)` is followed by `x.to(device)`. Where
   is `x`, and what happens when it is added to a tensor created with
   `device=device`?

   **Answer.** Still on the CPU: nothing kept the copy `.to()` returned, so the
   addition raises the same-device `RuntimeError`. Write `x = x.to(device)`.
