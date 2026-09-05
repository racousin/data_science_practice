# Lab 4 — Train Something Deep

Build a network deeper than the MLP from the 12h module, on your own dataset, and
prove with measurements that it trains correctly and finishes efficiently.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.

<!-- notes: They should reuse the processed dataset from Lab 2 so the comparison
against Lab 3's gradient boosting is meaningful. "The GBM still wins" is a
correct and publishable outcome — say so early or half of them will fudge it. -->

---

## Setup

```text
src/nn/
    data.py         <- Dataset + DataLoader construction
    model.py        <- the network
    train.py        <- the loop, CLI-configurable
    seed.py         <- seed_everything + seed_worker
tests/test_nn.py
runs/               <- TensorBoard event files
checkpoints/        <- gitignored
RESULTS.md
```

Use the processed dataset from Lab 2. Its Lab 3 gradient-boosting score is your
baseline; beating it is not required, reporting the comparison honestly is.

---

## Part A — Dataset and target (5 min)

Wire a `torch.utils.data.Dataset` over the Parquet from Lab 2. Declare in
`RESULTS.md` what one row is, the target, the metric, and the Lab 3 baseline.

- features `float32`, class labels `int64`
- the train/val/test split from Lab 2, unchanged — do not re-split
- normalisation statistics computed on train only
- `drop_last=True` if you use BatchNorm

---

## Part B — A network worth calling deep (10 min)

At least four weight layers, and at least three of the following:

- a residual block
- BatchNorm or LayerNorm
- Dropout, placed after the norm
- an `nn.Embedding` for a high-cardinality categorical column
- `Conv1d` or `Conv2d`, if your data has an axis that justifies it

Print the parameter count and the per-layer output shapes at construction and put
both in `RESULTS.md`. A parameter count you cannot explain is a model you have
not read.

---

## Part C — Make it train (10 min)

1. `seed_everything(seed)` with the seed passed on the command line, plus a
   `generator` and `worker_init_fn` on the `DataLoader`.
2. Overfit one batch: 32 samples, augmentation and dropout off, 200 steps, loss
   below 0.01. Commit the printed output as evidence.
3. Sweep three learning rates an order of magnitude apart, each logged to its own
   `runs/` directory.

```python
writer.add_scalar("loss/train", tr, epoch)
writer.add_scalar("loss/val", va, epoch)
writer.add_scalar("grad_norm", gn, epoch)
```

Log the gradient norm and the learning rate as well as the losses, plus one
weight histogram per epoch.

---

## Part D — Make it finish (10 min)

- **Mixed precision.** `torch.amp.autocast` plus `GradScaler` (fp16), or bf16
  without one. Measure samples/sec with and without: ten warm-up steps, fifty
  timed steps, `torch.cuda.synchronize()` on both sides.
- **Early stopping** on the validation metric, best checkpoint written on
  improvement and restored before the test evaluation.
- **A resumable checkpoint** carrying model, optimizer, scheduler, scaler, epoch
  and seed.

No CUDA device? Run the same measurement on CPU or MPS and report the result,
including "no speedup" with the numbers that show it.

---

## Part E — Tests and RESULTS.md (10 min)

Three tests, on a fixture of a few dozen rows committed under `tests/fixtures/`:

```python
def test_forward_shape():
    """model(batch) returns (batch_size, n_classes) for a batch of 4."""

def test_overfits_one_batch():
    """8 samples, 150 steps, no dropout: final loss < 0.01."""

def test_checkpoint_round_trip():
    """save(), fresh model, load(): identical logits on the same input."""
```

The third uses `torch.testing.assert_close` on the logits, in `eval()` mode. It
catches a `state_dict` loaded with `strict=False`, a buffer that was never saved,
and a model left in `train()` mode.

---

## The write-up

`RESULTS.md` contains:

- the dataset, the metric, and the Lab 3 baseline
- the model: layer table, parameter count, per-layer shapes
- the overfit-one-batch output
- the LR sweep: three rates, three validation scores, one curve screenshot
- throughput with and without AMP: samples/sec, batch size, device
- the epoch the best checkpoint came from, and that checkpoint's test score
- whether the network beat the gradient-boosting baseline, and why either way

The pull-request description states the learning rate you chose and the evidence
for it, one thing that was broken and how the instrumentation showed it, the
measured AMP speedup with device and batch size, and what you would change with a
GPU-hour instead of ten minutes.

---

## Grading

| Criterion | Weight |
|---|---|
| Four or more weight layers, three or more components from Part B | 15% |
| Seeded run, reproducible to a stated tolerance | 10% |
| Overfit-one-batch proof committed | 10% |
| LR sweep over three rates, with curves | 15% |
| TensorBoard logs committed: scalars, grad norm, one histogram | 10% |
| Mixed precision with a measured, fully specified speedup | 15% |
| Early stopping with best-checkpoint restore | 10% |
| Three tests passing | 15% |

---

## Automatic deductions

- test scores reported from the last epoch instead of the restored checkpoint
- `model.eval()` missing on the validation or test path
- a softmax before `CrossEntropyLoss`
- a bare `except` anywhere in the training loop
- a speedup quoted without batch size and device
- checkpoints, event files over 10 MB, or data committed to git
- normalisation statistics computed on the full dataset

---

## Carry it forward

The loop you wrote here is the one you will run in Session 5 with a pretrained
backbone, in Session 7 with a transformer and in Session 10 with a policy
gradient. Only the model and the data change.

Make it configurable now — seed, learning rate, batch size and epochs as
arguments, not constants — and you will not rewrite it four more times.
