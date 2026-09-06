# Lab 4 — Train and Submit

The module's closing exercise. Everything from Sessions 1–4 in one deliverable:
a packaged, tested, version-controlled PyTorch training pipeline, submitted to
ML-Arena.

**Time:** 45 minutes. **Deliverable:** a merged PR + a leaderboard entry.

---

## Part A — Package it (10 min)

Add to your Lab 1 repository, on a branch:

```text
src/textstats/          <- existing
src/mlp/
    __init__.py
    model.py            <- the nn.Module
    data.py             <- Dataset / DataLoader construction
    train.py            <- the loop, importable and callable
tests/test_mlp.py
```

Add `torch` **and `torchvision`** to `pyproject.toml` dependencies and
re-lock. Part C loads MNIST through `torchvision.datasets`; a fresh clone
with only `torch` fails `uv run pytest` with `ModuleNotFoundError: No module
named 'torchvision'`, which is the 20% row at the top of the grading table.

**Requirement:** `train.py` exposes `train(config: dict) -> dict` returning the
metrics. No work at import time — importing must not train anything.

---

## Part B — Tests that catch real bugs (10 min)

At minimum, four:

```python
def test_forward_shape():
    """Model maps (batch, in_dim) -> (batch, n_classes)."""

def test_overfits_one_batch():
    """200 steps on 32 examples drives the loss below 0.1."""

def test_eval_mode_is_deterministic():
    """Two forward passes under model.eval() give identical output."""

def test_missing_config_key_raises():
    """train({}) raises KeyError — no silent defaults."""
```

The second is the valuable one: it fails if shapes, the loss, the optimizer
wiring, or `zero_grad` are wrong.

The fourth enforces fail-fast — `config["lr"]`, never `config.get("lr", 1e-3)`.

---

## Part C — Train (10 min)

Train an MLP on MNIST. Requirements:

- seeded
- train/validation split, stratified
- both losses printed every epoch
- early stopping with best-checkpoint restore
- final metrics written to `results.json`

```python
from torchvision import datasets, transforms

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train = datasets.MNIST("data/", train=True, download=True, transform=tf)
```

`data/` is gitignored. Committing MNIST is an automatic deduction.

**What good looks like.** A correctly wired 784-128-10 MLP with Adam at
`lr=1e-3` reaches **97% validation accuracy within 5 epochs** — measured on a
laptop CPU: 0.954 after epoch 1, 0.972 after epoch 5, about 13 seconds an
epoch. Below 95%, stop and re-read the failure table in *Training Loop End to
End*: the cause is almost always the normalisation, a softmax before
`CrossEntropyLoss`, or a missing `model.eval()`. Multinomial logistic
regression on the same pixels gets 91.3% — under that, the network is not
learning at all.

Overfit one batch **before** the full run. Say in your PR description what it
told you.

---

## Part D — Submit to ML-Arena (10 min)

**AIE S4 — MNIST Warm-up** (competition `182`) takes a single
`submission.csv` of predictions on 5,000 held-out digits. Download `X_test.csv`
from the competition's data tab; the columns `p0 … p783` are the image
flattened row-major as `uint8` 0-255, i.e. what `datasets.MNIST` gives you
before `ToTensor()`.

**The submission schema is `id,label`** — not `id,prediction`. Lab 3 used
`prediction`; this competition does not, and upload validation rejects the file
before anything runs (`Column mismatch. Expected: ['id', 'label']`). Every test
id must appear exactly once, and `label` is the predicted digit, 0-9.

```python
import pandas as pd

pd.DataFrame({"id": test_ids, "label": predictions}).to_csv("submission.csv", index=False)
```

---

## Part D — send it

```bash
uv pip install mlarena-sdk
```

The distribution is `mlarena-sdk` and it imports as `mlarena`. `uv pip install
mlarena` gets you an unrelated package with no `connect`.

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.submit(competition_id=182, files=["submission.csv"])
print(client.leaderboard(182).head())
```

---

## Part D — the numbers

Ranking is on **accuracy — higher is better**. Predicting the
single most common digit for all 5,000 images scores **0.108**. The
multinomial logistic regression on raw pixels that ships as the competition's
benchmark scores **0.913** — that is the bar. The 784-128-10 MLP from Part C,
Adam at `lr=1e-3` for five epochs, scores **0.982**. Under 0.913 means
something is broken, not under-tuned.

Read that number honestly: 4,300 of the 5,000 evaluation images are
byte-identical to images in the torchvision train split Part C has you train
on, so the leaderboard is an **upper bound**. The same model measured 0.982
there against 0.972 on its own held-out validation split. The validation
number is the honest one.

Getting on the board matters; your position does not. This is the dry run for
the project, and the point is that the submission path works before it counts.

---

## Part E — Pull request (5 min)

Description must contain:

- what the one-batch overfit test told you
- your final train and validation numbers, and whether they indicate overfitting
- the learning rate you settled on and how you chose it
- your competition 182 accuracy next to your own validation accuracy, and
  which of the two you believe
- one thing you tried that did not help

That last one is not filler. A PR with only successes describes a process that
did not happen.

---

## Grading

| Criterion | Weight |
|---|---|
| `uv sync && uv run pytest` green on a fresh clone | 20% |
| The four required tests present and meaningful | 25% |
| Training loop correct: eval mode, no_grad, early stopping + restore | 25% |
| Fail-fast config: no defaults for required keys | 10% |
| ML-Arena submission accepted | 10% |
| PR description covers all five points | 10% |

---

## Automatic deductions

- `data/` or `*.pt` committed
- a softmax before `CrossEntropyLoss`
- `model.eval()` missing from the validation path
- any bare `except`

---

## What you should now have

A repository that:

- installs and tests from a clean clone in three commands
- has a readable history on feature branches with reviewed PRs
- contains a documented `CLAUDE.md` your agent actually uses
- trains a neural network with an honest validation protocol
- produces a submission the platform accepts

That is the engineering floor for *MS2A - Machine Learning Practice*, and half of
the project grade is this repository staying that way for ten more weeks.

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone — cloned into a new
      directory, not the one I worked in
- [ ] `torch` and `torchvision` are both in `pyproject.toml` and in the lockfile,
      and `python -c "import mlp.train"` returns immediately without training
- [ ] All four tests named in Part B exist and pass, and
      `test_overfits_one_batch` drives the loss below 0.1
- [ ] My epoch log prints train and validation loss on every epoch, and
      `results.json` holds the numbers from the restored best checkpoint, not
      from the last epoch
- [ ] My best validation accuracy over the five epochs is **≥ 0.96** (a
      correctly wired 784-128-10 MLP with Adam at `lr=1e-3` lands between 0.965
      and 0.976 depending on seed and split; 0.972 at epoch 5 is one such run, and
      measured reference; below 0.95 is a bug, not a tuning problem)
- [ ] `git status` is clean, and neither `data/` nor any `*.pt` is tracked
- [ ] My PR description contains all five points from Part E, including the one
      thing that did not help
- [ ] My submission is on the leaderboard of AIE S4 — MNIST Warm-up (#182)
- [ ] My score beats the baseline: **accuracy ≥ 0.913**

If the last two are not ticked you have not finished the lab, however good the
code is.
