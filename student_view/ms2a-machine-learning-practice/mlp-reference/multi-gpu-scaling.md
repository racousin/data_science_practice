# Multi-GPU Scaling

Reference only — not covered in class, because nobody in the room has four
GPUs. Read it before you use a cluster or a multi-device cloud instance, or
when you want to know what `torchrun` in a paper's README actually does.
Session 4 covers the single-device training loop this builds on.

<!-- notes: Self-study, linked from Session 4. Never lectured. The one slide
that applies to a laptop is gradient accumulation. -->

---

## Four kinds of parallelism

| Strategy | What is split | Per-GPU memory | Communication | Use when |
|---|---|---|---|---|
| **Data** | the batch | full model | all-reduce gradients | model fits on one GPU |
| **Model** | the layers | model / N | activations, point to point | model does not fit |
| **Pipeline** | layers, plus micro-batches | model / N | activations between stages | deep model, balanced stages |
| **Tensor** | inside a single layer | model / N | all-reduce per layer | very wide transformer layers |

That is also the order in which to reach for them: the last three exist because
a model stopped fitting, not because they are faster.

---

## Data parallelism, concretely

Every GPU holds a complete copy of the model. The batch is split $N$ ways, each
device runs its own forward and backward pass, and the gradients are averaged
before the update.

$$
\nabla_{avg} = \frac{1}{N} \sum_{i=1}^{N} \nabla_i
$$

Because the update is identical everywhere, replicas stay bit-identical — the
invariant that lets DDP synchronise gradients only, never parameters.

---

## `DataParallel` is deprecated

`nn.DataParallel` runs one process driving all devices: it scatters the batch,
gathers the outputs back on GPU 0 and computes the loss there. GPU 0 runs out
of memory first while the others idle, one process means one GIL for every
device's Python work, and the model is replicated on *every* forward pass.
`nn.parallel.DistributedDataParallel` runs one process per GPU, replicates once
at construction, and overlaps the all-reduce with the backward pass — faster on
a single machine too. Treat `DataParallel` as legacy; do not copy it.

---

## All-reduce

Averaging gradients is a collective, not a broadcast to a master. Ring
all-reduce passes each bucket around a ring twice — once reducing, once
distributing — at a cost of

$$
T = \alpha + \beta \cdot \frac{2(N-1)}{N} \cdot M
$$

with $\alpha$ the latency, $\beta$ the inverse bandwidth, $M$ the gradient size
and $N$ the device count. The bandwidth term tends to $2 \beta M$ and stops
growing with $N$; that is why it scales. DDP buckets gradients (25 MB by
default) and fires each bucket's all-reduce as soon as it is complete, hiding
most communication under the backward pass.

---

## The launcher and the vocabulary

```bash
torchrun --nproc_per_node=4 train.py --epochs 30
```

`torchrun` starts one process per GPU and injects the environment the script
reads but never computes: **world size** (total processes across all machines),
**rank** (this process's global index), **local rank** (its index *on this
machine*, which is the GPU id), plus `MASTER_ADDR` and `MASTER_PORT`.

---

## Reading the environment

```python
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])   # KeyError if not under torchrun
torch.cuda.set_device(local_rank)
model = DDP(model.to(local_rank), device_ids=[local_rank])
```

`os.environ[...]`, not `os.environ.get("LOCAL_RANK", 0)`. A default of zero
turns "you forgot the launcher" into "four processes fighting over GPU 0",
which looks like a hardware fault for an hour. Rank 0 logs and checkpoints;
every other rank stays silent, or you get eight copies of every line.

---

## Effective batch size and the learning rate

$$
B_{eff} = B_{gpu} \times N, \qquad \eta_{new} = \eta_{base} \times N
$$

A gradient averaged over $N$ times more samples carries about $\sqrt{N}$ times
less noise, so the old learning rate now steps too timidly. The linear scaling
rule (Goyal et al., 2017) compensates, with a few epochs of warmup up to
$\eta_{new}$ because the rule breaks at initialisation when the surface is
steep.

**Failure mode.** Moving from 1 GPU to 8 and keeping the learning rate is the
most common reason a distributed run converges *worse* than the single-device
baseline. Nothing is broken; the optimiser is under-stepping by 8×.

---

## `DistributedSampler` and the seeding trap

Without a distributed sampler every rank iterates the whole dataset, and you
have run four epochs while reporting one.

```python
sampler = DistributedSampler(train_ds, shuffle=True)
loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
for epoch in range(n_epochs):
    sampler.set_epoch(epoch)      # without this: identical order every epoch
```

`set_epoch` is the trap. The sampler seeds its permutation with `seed + epoch`;
never update the epoch and every epoch replays the same order on every rank —
shuffling silently does nothing, with no warning. Second trap: the sampler pads
the dataset so all ranks get equal batch counts, and on the *validation* set
those duplicates are counted twice. Evaluate on rank 0 only.

---

## Gradient accumulation — the part that applies to you

You cannot buy devices, but you can buy the *effect* of a larger batch by
summing gradients over several small batches before stepping.

```python
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

The division by `accum_steps` is not optional: without it you hand the
optimiser a sum where it expects a mean, and the effective learning rate is
`accum_steps` times too large. This buys large-batch convergence at
small-batch memory cost, not speed — the compute is identical.

---

## Where the memory goes, and ZeRO

![Training memory components](/api/academic_courses/assets/lessons/125/distribution.png)

With Adam in float32, parameters cost $4|\theta|$ bytes, gradients another
$4|\theta|$ and the two Adam moments $8|\theta|$ — 16 bytes per parameter
before a single activation, replicated on every device by DDP. ZeRO (DeepSpeed)
and FSDP shard exactly that state across the data-parallel group and all-gather
each piece only when it is needed.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model)
```

Memory falls towards $1/N$ of the DDP figure while communication rises. Worth
it above a few billion parameters, rarely below one billion.

---

## Model, pipeline and tensor parallelism

**Model parallelism** puts different layers on different devices; naively one
device runs at a time, so utilisation is $1/N$ — memory, not speed. **Pipeline
parallelism** fixes the idling by splitting the batch into $M$ micro-batches so
every stage works on a different one, leaving a *bubble* of $(P-1)/M$ for $P$
stages — use at least $4P$ micro-batches. **Tensor parallelism** splits
individual weight matrices, column-parallel for an MLP's first linear and
row-parallel for the second, giving one all-reduce per block; it needs
NVLink-class bandwidth so it stays *inside* a node. Frontier-scale training
combines all three: tensor within a node, pipeline across nodes, data on top.

---

## When scaling out is not the answer

Watch `nvidia-smi` during a run before adding a device. At 30% utilisation more
GPUs give you more idle GPUs.

| Symptom | Real cause | Fix |
|---|---|---|
| Low utilisation, spiky | dataloader starving the device | more workers, `pin_memory`, cache decodes |
| Out of memory only | model or batch too large | AMP, gradient checkpointing, accumulation |
| Slow, GPU at 95% | genuinely compute-bound | now scaling out helps |
| 8 GPUs give 3× | communication bound | bigger buckets, faster interconnect, fewer syncs |

**Rule.** Profile first, parallelise second. Most "we need a cluster" problems
in a student project are `num_workers=0`, a per-sample JPEG decode that should
have been a cached tensor, or a `.item()` inside the loop forcing a device
synchronisation every step.

