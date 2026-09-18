# Lab 4 — Train Something Deep

Build a network deeper than the MLP from the 12h module, on your own dataset, and
prove with measurements that it trains correctly and finishes efficiently.

**Time:** 45 minutes in the room, plus the five-minute Part F.
**Deliverable:** a merged PR in your project repository.

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

```bash
uv add torch tensorboard
```

`tensorboard` is a **separate package** from `torch`: without it
`from torch.utils.tensorboard import SummaryWriter` raises
`ModuleNotFoundError: No module named 'tensorboard'`, and 10% of the grade is
committed event files.

Lab 2 persisted a fitted `Pipeline`, not a transformed matrix — you apply it to
Lab 1's Parquet here. Its Lab 3 gradient-boosting score is your baseline;
beating it is not required, reporting the comparison honestly is.

---

## Part A — Dataset and target (5 min)

Lab 2's artefact is `models/pipeline_<date>.joblib`, a **fitted `Pipeline`** —
not a matrix. Load it, apply it to Lab 1's Parquet, and wire a
`torch.utils.data.Dataset` over the result.

```python
import joblib, pandas as pd
pipe = joblib.load("models/pipeline_2026-09-14.joblib")   # your dated artefact
X = pipe.transform(pd.read_parquet("data/lab1.parquet"))
```

Lab 2 split two ways — train and test, `test_size=0.2, random_state=0` — and
never made a validation split. Reproduce that boundary exactly, then carve a
validation set off the training part with a seed you record. All three sizes go
in `RESULTS.md`, because Parts C and D both report numbers on the validation
split.

- features `float32`, class labels `int64`
- the Lab 2 train/test boundary is **reproduced, not redrawn**: the same
  `train_test_split(X, y, test_size=0.2, random_state=0)` call
- normalisation statistics computed on train only — the Lab 2 pipeline already
  did that, so do not scale a second time
- `drop_last=True` if you use BatchNorm

Declare in `RESULTS.md` what one row is, the target, the metric, and the Lab 3
baseline.

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

- **Mixed precision.** Pick the device once and pass it to autocast; a
  `GradScaler` is only needed for CUDA fp16.

  ```python
  dev = ("cuda" if torch.cuda.is_available()
         else "mps" if torch.backends.mps.is_available() else "cpu")
  with torch.amp.autocast(dev, dtype=torch.float16):
      loss = criterion(model(xb), yb)
  ```

  Measure samples/sec with and without: ten warm-up steps, fifty timed steps,
  and a **device-aware barrier** on both sides —
  `torch.cuda.synchronize()` on CUDA, `torch.mps.synchronize()` on MPS, nothing
  on CPU. A bare `torch.cuda.synchronize()` on a machine without CUDA raises
  `AssertionError: Torch not compiled with CUDA enabled`.
- **Early stopping** on the validation metric, best checkpoint written on
  improvement and restored before the test evaluation.
- **A resumable checkpoint** carrying model, optimizer, scheduler, scaler, epoch
  and seed.

No CUDA device? Run the same measurement on CPU or MPS and report the result,
including "no speedup" with the numbers that show it. Check that autocast
engaged before you report anything: `torch.amp.autocast("cuda", ...)` on a
machine without CUDA emits one `UserWarning` and silently runs in fp32, so a
speedup of exactly 1.00x usually means the block did nothing.

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

## Part F — Put it on the board (5 min)

The session's competition is **1 Minute Permuted MNIST**, ML-Arena competition
**8**, and it is deliberately not your project dataset. Each episode hands the
agent 60,000 labelled and 10,000 unlabelled images with the pixel positions
*and* the label meanings freshly permuted, and gives it **60 seconds** to train
and predict. Ten episodes, mean accuracy, higher is better; a timeout scores
0.0 for that episode. It is this session's throughput lesson with a deadline
attached.

The submission is one `agent.py` on the competition's own contract — `class
Agent` with `train(X_train, y_train)` and `predict(X_test) -> list` — not the
training script from Parts A to E. Start from the template on the competition
page and replace its two methods. The arrays cross a JSON boundary and arrive
as plain Python lists, so `np.asarray(...)` before any array maths.

Size to the machine that actually runs it: the competition record says
`agent_memory_limit: 3Gi` and `agent_cpu_limit: 3000m`. The overview page's
"4 GB RAM / 2 cores" is stale — budget for **3 GiB and 3 cores**.

```bash
uv pip install mlarena-sdk        # the PyPI name; it imports as `mlarena`
```

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")   # Profile -> API Keys
client.submit(competition_id=8, files=["agent.py"])
print(client.leaderboard(8, top=5))
```

---

## Part F — the ladder

The first row is the board's own `__benchmark__` agent. The other three were
measured on 60,000 train / 10,000 test permuted MNIST with BLAS pinned to three
threads, to match `agent_cpu_limit: 3000m`:

| agent | accuracy | wall clock, of a 60 s budget |
|---|---|---|
| the template's random labels | 0.099 | — |
| `LogisticRegression(max_iter=30)` | 0.921 | 3 s |
| `LogisticRegression(max_iter=200)` | 0.926 | 17 s |
| one hidden layer of 256 units, 20 epochs | 0.980 | 36 s |

The board agrees: its `__benchmark__` row scores **0.0989** and ranks 1038th of
1084, the median of the 1,075 scored entries is **0.982**, and the best is
**0.999**. Your bar is **0.926** — whatever you learned this session should beat
plain logistic regression on flattened pixels. Leave margin on the clock: those
wall times are a laptop's, the grading pod is slower, and 61 seconds scores
zero.

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

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] `RESULTS.md` states the three split sizes, and the train/test boundary is
      Lab 2's `random_state=0` split reproduced, not a fresh one
- [ ] `RESULTS.md` carries the parameter count and the per-layer output shapes,
      and I can account for the count layer by layer
- [ ] The overfit-one-batch output is committed and its final loss is below 0.01
- [ ] `runs/` holds three event files, one per learning rate, each containing
      `loss/train`, `loss/val`, `grad_norm` and one weight histogram
- [ ] The AMP measurement names its device, batch size and precision — and I
      confirmed autocast engaged rather than reporting a 1.00x no-op
- [ ] The reported test score comes from the restored best checkpoint, and
      `test_checkpoint_round_trip` passes
- [ ] My submission is on the leaderboard of 1 Minute Permuted MNIST (#8)
- [ ] My score beats the baseline: **accuracy > 0.926** — plain logistic
      regression on flattened pixels

If the last two are not ticked you have not finished the lab, however good the
code is.
