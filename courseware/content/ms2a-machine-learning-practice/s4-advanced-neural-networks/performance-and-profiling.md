# Performance and Profiling

Every optimization in this lesson is worthless until you know where the time
goes. Measure first: the bottleneck is almost never where the intuition puts it.

<!-- notes: 30 minutes. Run the empty-dataloader test live on a real loop — in
most student projects it accounts for more than half the epoch. Multi-GPU is in
the Reference module and is not lectured. -->

---

## Time goes to three places

Data loading, compute, host-to-device transfer. A "slow model" is usually a
starved GPU. Timing it naively gives the wrong answer, because CUDA calls are
asynchronous — the Python line returns before the kernel has run.

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    train_step()
torch.cuda.synchronize()
print(f"{50 * batch_size / (time.perf_counter() - t0):.0f} samples/s")
```

Two synchronisations, and ten warm-up iterations before the timed ones.

![Average time per training-loop component](assets/nn/time.png)

The optimizer step being the largest bar here is a small-model artefact: its cost
is fixed per parameter and independent of batch size.

---

## Forward, backward, and the 2× rule

![Forward versus backward FLOPs per layer](assets/nn/flop.png)

A `Linear(n, m)` costs about $2nm$ FLOPs per sample forward. Backward computes
the weight gradient and the gradient passed back, so it costs twice that.

$$
FLOPs_{step} \approx 3 \times FLOPs_{forward} + 8 N_{params}
$$

The last term is Adam, and it does not scale with batch size: inference is three
times cheaper than a training step, and a small-batch model with many parameters
is optimizer-bound, not compute-bound.

---

## Is the dataloader the bottleneck?

```python
t0 = time.perf_counter()
for xb, yb in train_dl:
    pass
print(f"loader-only epoch: {time.perf_counter() - t0:.1f}s")
```

Compare with a full epoch. If they are within 20% of each other, your GPU is idle
most of the time and the fix is `num_workers`, `persistent_workers` and decoding
less per sample. Highest-value measurement here, and almost nobody runs it.

---

## torch.profiler

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    for _ in range(5):
        train_step()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
```

Sort by `self_cuda_time_total`, not `cuda_time_total`: *self* excludes children,
so the top row is the operation actually consuming the time rather than the
`aten::linear` containing it. Wrap sections in `record_function("backward")` for
your own labels. `prof.export_chrome_trace("trace.json")` opens in
`chrome://tracing`, where a starved GPU shows as gaps in the CUDA row.

---

## Batch size

![CPU versus GPU forward time across batch sizes](assets/nn/cpugpubatch.png)

CPU time grows roughly linearly with the batch. GPU time is nearly flat from 1 to
64, because at those sizes the device is mostly idle waiting for kernel launches.
**A small batch wastes a GPU.** Raise it until memory or accuracy stops you — and
remember that raising it means re-tuning the learning rate.

---

## Host-to-device transfer

![CPU-GPU memory transfer time by payload size](assets/nn/overhead.png)

500 MB costs 113 ms up and 350 ms back: transfers are not free, and the return
trip is worse.

Move the model to the device once, never per batch. Accumulate metrics on the
device and call `.item()` once per epoch — it forces a synchronisation, and a
`print(loss.item())` in the inner loop can cost more than the backward pass. Pair
`pin_memory=True` with `.to(device, non_blocking=True)`.

---

## Mixed precision

![Mixed-precision training: FP16 compute with an FP32 master copy](assets/nn/mixed.jpg)

Run the matmuls in 16-bit, keep the weights and the update in 32-bit: roughly 2×
throughput on tensor-core hardware, and half the activation memory.

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    loss = criterion(model(xb), yb)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer); scaler.update()
```

The scaler multiplies the loss before `backward` so small gradients do not
underflow in fp16, then divides it out before the step. `unscale_` is mandatory
before clipping — clipping scaled gradients clips the wrong thing.

`autocast("cuda")` on a machine without CUDA does **not** raise: it warns once
and runs the block in fp32, and `GradScaler("cuda")` disables itself the same
way. Pick the device once and pass it in — `dev = "cuda" if
torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else
"cpu"`, then `torch.amp.autocast(dev, dtype=torch.float16)`. If your measured
speedup is exactly 1.00x, check that autocast engaged before reporting it.

---

## fp16, bf16, and when it does not pay

| | fp16 | bf16 |
|---|---|---|
| Exponent range | narrow | same as fp32 |
| Needs `GradScaler` | yes | no |
| Hardware | Volta and later | Ampere and later |

Prefer `bfloat16` where it exists: same speed, no scaler, no underflow. AMP buys
nothing when the model is small enough to be launch-bound, when the bottleneck is
the dataloader, or on hardware without tensor cores. Measure the speedup — one
reported without a batch size and a device name is not a measurement.

---

## torch.compile

```python
model = torch.compile(model)
```

Traces the model, fuses element-wise operations into single kernels, and
generates code for your device: typically 1.2–2× on a real network, occasionally
nothing. The first call pays seconds to minutes of compilation, so exclude it
from your timing. Changing input shapes triggers a recompile — use `drop_last`
and fixed resolutions, or you pay that cost every epoch.

---

## Where the memory goes

![Memory distribution of a small MLP trained with Adam](assets/nn/distribution.png)

Parameters, gradients, optimizer state, activations. With Adam in fp32 the first
three are fixed at **16 bytes per parameter** — 4 for the weight, 4 for its
gradient, 8 for the two moments — so a 100 M-parameter model needs 1.6 GB before
a single activation exists. Activations are the part that scales with batch size,
and the only part you can trade against throughput.

```python
torch.cuda.reset_peak_memory_stats()
train_one_epoch()
print(f"peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

`nvidia-smi` reports what the caching allocator *reserved*, which is larger and
does not shrink when tensors are freed. Never conclude you are out of memory from
`nvidia-smi`.

---

## When you are genuinely out of memory

Lower the batch size and accumulate gradients; enable AMP; then reach for
activation checkpointing, which recomputes activations during the backward pass
instead of storing them — about 30% slower for about 60% less activation memory.

```python
h = torch.utils.checkpoint.checkpoint(self.block, x, use_reentrant=False)
```

It is the last resort: a batch half the size costs only a learning-rate change.

---

## Report throughput honestly

State **samples per second**, with the batch size, the precision and the device.
"Three minutes per epoch" is not comparable across batch sizes, datasets or
machines. Measure the same way every time: ten warm-up steps, fifty timed steps,
two synchronisations, the median of three repeats. Scaling beyond one GPU is in
the Reference module — you do not have the hardware, and it is almost never the
thing that was slow.

---

## Check yourself

1. How do you find out whether the dataloader is the bottleneck, and what is
   the fix when it is?

   **Answer.** Time a loop that only iterates `train_dl` and does nothing, and
   compare it with a full epoch. Within 20% of each other means the device is
   idle most of the time; the fix is `num_workers`, `persistent_workers`, and
   decoding less per sample.

2. Run this. You should get exactly the output shown.

   ```python
   N = 100_000_000
   print(f"{N * 16 / 1e9:.1f} GB")      # -> 1.6 GB
   ```

   Sixteen bytes per parameter with Adam in fp32 — 4 for the weight, 4 for its
   gradient, 8 for the two moments — before a single activation exists.

3. You measure a 1.00x AMP speedup. What does this lesson tell you to check
   first, and what makes a speedup reportable at all?

   **Answer.** Check that autocast actually engaged: on a machine without CUDA
   `autocast("cuda")` warns once and runs in fp32. AMP also genuinely buys
   nothing when the model is launch-bound, when the dataloader is the
   bottleneck, or on hardware without tensor cores. A speedup is reportable
   only with the batch size, the precision and the device name attached.
