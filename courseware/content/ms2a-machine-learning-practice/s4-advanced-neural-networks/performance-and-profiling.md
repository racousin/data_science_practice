# Performance and Profiling

Every optimization in this lesson is worthless until you know where the time
goes. Measure first: the bottleneck is almost never where the intuition puts it.
The measurements below were made for this lesson on a laptop CPU pinned to three
threads — the same budget as the challenge-8 agent — so the numbers are real but
*yours will differ*; the method is what transfers.

<!-- notes: 30 minutes. Run the empty-dataloader test live on a real loop — in
most student projects it accounts for more than half the epoch. Open a run's
System section in W&B first and read the utilisation panel before touching the
profiler. Multi-GPU is in the Reference module and is not lectured. The last
section is the bridge to Lab 4 and challenge 8. -->

---

## The always-on monitor

Every W&B run records system metrics without a line of code: CPU utilisation of the
process, its resident memory, disk and network traffic, and for each GPU its
utilisation, memory, temperature and power, sampled every 15 seconds and shown in
the **System** section of the run page. Read that section before anything else in
this lesson. A GPU utilisation panel that sits at 30%, or oscillates between 0 and
100, is a starved GPU: the fix is in the data pipeline, and no kernel-level
optimisation will touch it. A process memory line that climbs epoch after epoch is
a leak — usually a list of `loss` tensors kept with their graphs instead of
`loss.item()`.

![Left, the resident memory of an evaluation loop that collects its outputs without no_grad, against the same loop under no_grad; right, the CPU utilisation of a starved and an in-memory training pipeline, which is the same line for both](assets/nn/system-metrics-leak-and-utilisation.png)

Both signatures above were measured with the same two series, sampled four times a
second on the three-thread laptop. Panel **A** is an evaluation loop over MNIST that
collects its predictions — `preds.append(model(xb))` — with no `torch.no_grad()`
and no `.detach()`: every kept output drags its autograd graph, and the activations
behind it, so the process grows by two gigabytes a second until the machine starts
swapping. The same loop under `torch.no_grad()` sits flat at 0.64 GB.

```python
with torch.no_grad():                      # no graph is built at all
    preds.append(model(xb).argmax(1))      # and nothing keeps one alive
```

Panel **B** is the warning label on utilisation. The starved `DataLoader` pipeline
and the in-memory one draw the *same* line at 110–130% of one core, while the second
does 2.4× the work of the first in the same thirty seconds. On a GPU, utilisation is
the first panel to read; on a CPU it barely moves, and the number that tells you
something is throughput — samples per second, which you log yourself
([Monitoring and Debugging](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/monitoring-and-debugging)).

---

## Time goes to three places

Data loading, compute, host-to-device transfer. A "slow model" is usually a
starved device. Timing it naively gives the wrong answer, because CUDA calls are
asynchronous — the Python line returns before the kernel has run.

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    train_step()
torch.cuda.synchronize()
print(f"{50 * batch_size / (time.perf_counter() - t0):.0f} samples/s")
```

Two synchronisations, and ten warm-up iterations before the timed ones. On a CPU
there is nothing to synchronise, but the warm-up still matters: the first
iterations pay for allocator growth and thread-pool start-up.

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

Compare with a full epoch. If they are within 20% of each other, your device is
idle most of the time and the fix is `num_workers`, `persistent_workers`, and
decoding less per sample. Highest-value measurement here, and almost nobody runs
it.

![One MNIST epoch with three input pipelines, loader only versus loader plus training step; and training throughput against batch size, both on three CPU threads](assets/nn/dataloader-and-batch-size-on-three-cores.png)

Panel **A** is that test on 60 000 MNIST digits with three threads. The textbook
pipeline — `torchvision.datasets.MNIST` with a `ToTensor` + `Normalize`
transform through a `DataLoader` — spends 84.3 s per epoch decoding PIL
images one by one, against 20.0 s for loading *and* training: the loader is
most of the epoch. Two worker processes bring the loader to 18.8 s, but they
also compete with the training threads for the same cores, so the full epoch takes
31.0 s. The third pipeline converts the whole dataset to one normalised
tensor once and slices it by index: 0.5 s to iterate, 8.4 s with
training — the loader has disappeared and the epoch is compute. When the dataset
fits in memory, and 60 000 × 784 floats is 188 MB, the `DataLoader` is overhead
you are paying for nothing.

---

## torch.profiler

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    for _ in range(10):
        with record_function("forward"):
            loss = criterion(model(xb), yb)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            opt.step(); opt.zero_grad(set_to_none=True)
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

Sort by `self_device_time_total` (`self_cpu_time_total` on a CPU-only run), not
`device_time_total`: *self* excludes children, so the top row is the operation
actually consuming the time rather than the `aten::linear` containing it.
`record_function` labels give you your own rows. The exported trace opens in
Perfetto at `ui.perfetto.dev`, where a starved GPU shows as gaps in the CUDA row
and an optimizer-bound step as a long tail of tiny element-wise kernels after each
backward. `with_stack=True` maps every op to the Python line that issued it.

![The ten operations with the largest self CPU time in one training step of the MLP, coloured by phase, from torch.profiler](assets/nn/profiler-top-ops.png)

Profiled: the same MLP at batch 256 on three threads. The two matmul kernels
(`addmm` forward, `mm` backward) are 18% of the step. The AdamW update — `mul_`,
`addcmul_`, `sqrt`, `addcdiv_`, `lerp_`, six element-wise passes over every
parameter, plus its Python overhead — is more than a third. This step is
optimizer-bound, exactly as the $8N_{params}$ term predicts, and the profiler said so
in one table: the fix is a larger batch, not a faster matmul.

---

## Batch size

![CPU versus GPU forward time across batch sizes](assets/nn/cpugpubatch.png)

CPU time grows roughly linearly with the batch. GPU time is nearly flat from 1 to
64, because at those sizes the device is mostly idle waiting for kernel launches.
**A small batch wastes a GPU.** It wastes a CPU too, for the reason the profiler
just gave: panel **B** of the figure above shows the same MLP going from
304 samples/s at batch 8 to 35,978 at batch 1024 on three threads — more
than a hundredfold — because the per-step optimizer and framework cost is paid once
per batch whatever its size. Raise it until memory or accuracy stops you — and
remember that raising it means re-tuning the learning rate
([Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules)).

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
("CUDA is not available … Disabling autocast") and runs the block in fp32, and
`GradScaler("cuda")` disables itself the same way. Pick the device once and pass
it in — `dev = "cuda" if torch.cuda.is_available() else "mps" if
torch.backends.mps.is_available() else "cpu"`, then
`torch.amp.autocast(dev, dtype=...)`. If your measured speedup is exactly 1.00×,
check that autocast engaged before reporting it.

---

## fp16, bf16, and when it does not pay

![Bit layout of bfloat16, float32 and float16: sign, exponent and mantissa widths](assets/nn/fp32-fp16-bf16-bit-layout.png)

| | fp16 | bf16 |
|---|---|---|
| Exponent bits | 5 (range ±65 504) | 8 (same range as fp32) |
| Mantissa bits | 10 | 7 |
| Needs `GradScaler` | yes | no |
| Hardware | Volta and later | Ampere and later; recent CPUs |

The layout is the whole story: bf16 keeps fp32's eight exponent bits and gives up
mantissa, so it cannot overflow where fp32 would not, and needs no scaler; fp16
keeps precision and gives up range, which is what the scaler compensates for.
Prefer `bfloat16` where it exists. On a CPU, `torch.amp.autocast("cpu",
dtype=torch.bfloat16)` is real too, but it pays only on processors with bf16 matrix
instructions — measure it, it can be slower.

AMP buys nothing when the model is small enough to be launch-bound, when the
bottleneck is the dataloader, or on hardware without tensor cores. Measure the
speedup — one reported without a batch size and a device name is not a
measurement.

---

## torch.compile

```python
model = torch.compile(model)
```

Traces the model, fuses element-wise operations into single kernels, and
generates code for your device: typically 1.2–2× on a real network, occasionally
nothing. The first call pays for compilation — on the laptop CPU behind this
lesson's figures, **75 s** for the MLP, which then ran at
12.4 ms per step against 25.8 ms eager, a 2.1×
speedup that needs 5,574 steps to pay for itself. Exclude that first
call from your timing, and do not compile a model that trains for a minute. A
change of input shape triggers a recompile; since 2.1 the second one marks that
dimension dynamic, so you pay it once or twice, not every epoch — `drop_last=True`
and fixed resolutions avoid even that. `mode="reduce-overhead"` adds CUDA graphs
for launch-bound small-batch models.

---

## Where the memory goes

Parameters, gradients, optimizer state, activations. With Adam in fp32 the first
three are fixed at **16 bytes per parameter** — 4 for the weight, 4 for its
gradient, 8 for the two moments — so a 100 M-parameter model needs 1.6 GB before
a single activation exists. Activations are the part that scales with batch size,
and the only part you can trade against throughput.

![Memory of fp32 Adam training for an MLP and for ResNet-18 at several batch sizes: parameters, gradients, the two Adam moments and the activations kept for backward](assets/nn/memory-breakdown-adam-fp32.png)

The two models are opposite regimes. The MLP has 0.67 M parameters — 10.7 MB
of states — and keeps only 7 KB of activations per sample, so it stays
optimizer-dominated until batch 1500 and is still just 41 MB at batch 4096. ResNet-18 has 11.7 M parameters — 187 MB
of states — but saves 22 MB of activations per 224 × 224 image for the
backward pass: at batch 128 that is 3.0 GB, and the states are a rounding
error. The activation numbers were measured with
`torch.autograd.graph.saved_tensors_hooks` on a CPU; on a GPU add the framework's
workspace and the caching allocator's slack.

```python
torch.cuda.reset_peak_memory_stats()
train_one_epoch()
print(f"peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

`nvidia-smi` reports what the caching allocator *reserved*, which is larger and
does not shrink when tensors are freed. Never conclude you are out of memory from
`nvidia-smi`; `torch.cuda.memory_summary()` breaks the reservation down.

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

## Sixty seconds on three cores

```mlarena:challenge id=8
```

Challenge 8 is this lesson as an exam. `train(X_train, y_train)` gets 60 s on
three CPU cores and 3 GiB of RAM, `predict(X_test)` gets another 60 s, a timeout
scores zero, and `train()` must build a fresh model on every call. Everything above
applies, with the signs flipped by the budget:

- **`torch.set_num_threads(3)`** at the top of `train()`, and the same when you
  time locally; a laptop with ten cores will lie to you otherwise.
- **No `DataLoader`.** `X_train` arrives as a `(60000, 28, 28)` uint8 array
  already in memory. Convert it once — `torch.from_numpy(X).float().div_(255)`,
  normalise, flatten — and slice it by index, as in panel A above. Do not
  standardise with constants copied from an MNIST tutorial: the pixels and labels
  are permuted and the environment may call `train()` again on data that is not
  what you expect, so compute the mean and standard deviation from `X_train`
  itself, and build the model inside `train()`, never at import time.
- **A large batch.** Panel B is the argument: on three threads the step overhead
  dominates below batch 256. At batch 1024 the MLP trained 30 epochs
  of 60 000 digits in 50 s on this laptop. Measure the first epoch inside `train()`
  and set the epoch count from that rate, not from a constant.
- **A deadline, not a hope.** `t_end = time.perf_counter() + 50`; check it every
  few hundred steps and stop with margin, keeping the best-so-far weights. A 59.9 s
  `train()` that gets killed at 60 s is a zero.
- **No `torch.compile`** — 75 s of compilation is the whole budget —
  and no W&B import: the agent container has no network.
- **`predict()` is a forward pass** over 10 000 rows; batch it in a few thousand
  at a time under `torch.no_grad()` and it takes well under a second.

Lab 4 walks through the local harness that times all of this for you.

---

## Report throughput honestly

State **samples per second**, with the batch size, the precision, the device and
the thread count. "Three minutes per epoch" is not comparable across batch sizes,
datasets or machines. Measure the same way every time: ten warm-up steps, fifty
timed steps, two synchronisations, the median of three repeats. Scaling beyond
one GPU is in the Reference module — you do not have the hardware, and it is
almost never the thing that was slow.

---

## Check yourself

1. How do you find out whether the dataloader is the bottleneck, and what are the
   two different fixes when it is — one for a dataset that fits in memory, one for
   a dataset that does not?

2. Run this. You should get exactly the output shown.

   ```python
   N = 100_000_000
   print(f"{N * 16 / 1e9:.1f} GB")      # -> 1.6 GB
   ```

   Sixteen bytes per parameter with Adam in fp32 — 4 for the weight, 4 for its
   gradient, 8 for the two moments — before a single activation exists. Which of
   the four does `model.eval()` plus `torch.no_grad()` inference still need?

3. You measure a 1.00× AMP speedup. What does this lesson tell you to check first,
   and what makes a speedup reportable at all?

4. The profiler's top rows are `aten::mul_`, `aten::addcmul_`, `aten::sqrt` and
   `aten::addcdiv_`. Which phase of the step is that, and what is the cheapest
   change that shrinks it?

5. Your `train()` for challenge 8 compiles the model and trains at batch 32.
   Name the two decisions that lose the most time, and what you would measure to
   replace them.

6. A GPU utilisation panel in W&B oscillates between 0% and 100% every few
   seconds while the loss curve looks fine. What is happening, and which two
   `DataLoader` arguments do you try first?
