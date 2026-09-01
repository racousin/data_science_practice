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

Add `torch` to `pyproject.toml` dependencies and re-lock.

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

## Part C — Train (15 min)

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

Overfit one batch **before** the full run. Say in your PR description what it
told you.

---

## Part D — Submit to ML-Arena (10 min)

The warm-up competition takes a single `submission.csv` of predictions on the
held-out test set.

```bash
uv pip install mlarena
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.submit(competition_id=<id>, path="submission.csv")
print(client.leaderboard(<id>).head())
```

Getting on the board matters; your position does not. This is the dry run for
the project, and the point is that the submission path works before it counts.

---

## Part E — Pull request

Description must contain:

- what the one-batch overfit test told you
- your final train and validation numbers, and whether they indicate overfitting
- the learning rate you settled on and how you chose it
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

That is the engineering floor for *Machine Learning en pratique*, and half of
the project grade is this repository staying that way for ten more weeks.
