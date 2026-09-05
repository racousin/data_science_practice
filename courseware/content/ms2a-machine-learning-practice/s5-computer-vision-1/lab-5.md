# Lab 5 — Classify with a Pretrained Backbone

Fine-tune a pretrained CNN on a small image dataset, against a from-scratch
baseline, and say in numbers what the pretraining was worth.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.

<!-- notes: They will all want to skip the baseline. Do not let them — it is 20%
of the grade and it is the only thing that makes the headline number mean
anything. Have a GPU runtime ready for the room — it is not optional here.
On a laptop CPU one 13-epoch ResNet-18 fine-tune at 128 pixels is about
seventeen minutes, and Parts B and C are two of those before the ablation even
starts. -->

---

## Setup

```text
src/vision/
    data.py         <- dataset, splits, the two transform pipelines
    model.py        <- build_baseline(), build_pretrained(freeze=...)
    train.py        <- one train/eval loop, used by both models
    evaluate.py     <- metrics + confusion matrix
tests/test_vision.py
data/images/        <- gitignored
runs/               <- metrics json + confusion matrix png, committed
REPORT.md
```

The image files stay out of git. The metrics do not — `runs/*.json` is the
evidence for every claim in the PR.

**This lab needs a GPU.** On a free Colab T4 the runs below fit inside the
session. On a laptop CPU they do not: measured with torch 2.7.1 on two machines,
a ResNet-18 fine-tune costs **4.6–6.1 s per batch of 32 at 224 px** and **2.0 s
at 128 px**, against **0.8 s** at 128 px with the backbone frozen. At six
classes × 300 images that is 40 batches an epoch, so a single 13-epoch fine-tune
is about 17 minutes at 128 px and 40 minutes to an hour at 224. Without a GPU,
run every model at 128 px with the backbone frozen and say so in the PR.

---

## Part A — The dataset (5 min)

4 to 10 classes, 100 to 500 images per class: your own project's images if your
track is a vision track, otherwise a subset of a `torchvision.datasets` set —
`Flowers102`, `OxfordIIITPet`, `EuroSAT`, `FGVCAircraft`.

Split **stratified**, 70/15/15, fixed seed, written to disk. If images arrive in
bursts from one session or one source, split by group — otherwise near-duplicates
straddle train and test and the test accuracy is fiction.

```python
assert set(train_paths) & set(test_paths) == set()
```

---

## Part B — From-scratch baseline (10 min)

A small CNN, trained on the same data, the same schedule, the same splits:

```python
def build_baseline(num_classes: int) -> nn.Module:
    """3 conv blocks + global average pooling + linear head."""
```

Global average pooling rather than flatten-and-dense, `BatchNorm2d` after every
convolution with `bias=False` on it, and the same number of epochs as Part C.

Record test accuracy and macro-F1 to `runs/baseline.json`. That number is the
only honest reference point you will have.

---

## Part C — Fine-tune (12 min)

```python
def build_pretrained(num_classes: int, freeze: bool) -> nn.Module:
    """resnet18/resnet50 with a new head; freeze=True freezes the backbone."""
```

Two phases, as in the lesson:

1. backbone frozen, `freeze_bn`, head only, 3 epochs at `lr=1e-3`
2. unfreeze, 8–10 epochs, `lr=1e-4` on the backbone and `1e-3` on the head

Use `weights.transforms()` for the evaluation pipeline. Do not hard-code the
ImageNet normalization constants — read them off the checkpoint.

Keep the best-validation checkpoint. Evaluate on test **once**, at the very end,
for each of the three models.

---

## Part D — Augmentation ablation (8 min)

Train the same backbone three times — **backbone frozen, 3 epochs each** —
changing exactly one thing:

| Run | Train transform |
|---|---|
| `none` | resize + centre crop only |
| `basic` | `RandomResizedCrop` + `RandomHorizontalFlip` |
| `strong` | basic + `ColorJitter` + `RandomRotation(15)` |

Same seed, same epochs, same everything else. Report val and test accuracy for
all three in one table in `REPORT.md`.

Frozen and short on purpose: three *full* fine-tunes is over two hours on a CPU,
and the ablation answers the same question either way — which augmentation is
worse than `basic`. With a GPU and time to spare, run them unfrozen and say so.

One of them will be worse than `basic`. Say which, and say why you think so —
"the strong setting rotates and recolours, and my classes are distinguished by
colour" is a real answer.

---

## Part E — Tests (10 min)

```python
def test_model_output_shape():
    """build_pretrained(7, freeze=True) maps (2, 3, 224, 224) -> (2, 7)."""

def test_frozen_backbone_has_no_grad():
    """With freeze=True, every backbone parameter has requires_grad False
    and every head parameter has requires_grad True."""

def test_eval_transform_is_deterministic():
    """Applying the eval transform twice to one image gives identical tensors.
    `weights.transforms()` returns an `ImageClassification` object, not a
    `Compose` — it is not iterable and has no `.transforms`, so test the
    behaviour, not the class names."""

def test_training_is_reproducible():
    """Two 2-step runs from the same seed give bitwise-equal loss values."""
```

Four tests, run on two synthetic images or a fixture of a dozen files — never on
the real dataset, and never on the network. The third is the one that catches
the bug this session is built around.

---

## The confusion matrix

```python
from sklearn.metrics import confusion_matrix, classification_report
```

Save it as `runs/confusion_matrix.png` with class names on both axes, and put the
`classification_report` output in `REPORT.md`.

Then read it. Which class has the worst recall? Which pair is confused in both
directions, and which only in one? A one-directional confusion is usually
imbalance; a symmetric one is two classes that genuinely look alike.

Open the five highest-loss test images. At least one will be mislabelled.

---

## Part F — Put it on the board (5 min here, the training run at home)

The session's competition is **Blood Cell Classification**, `competition_id=173`
— eight cell types, 28×28 RGB, ranked on **F1-macro**, higher is better, 0 to 1.

```bash
uv pip install mlarena-sdk
```

```python
import mlarena, numpy as np, pandas as pd

client = mlarena.connect(api_key="mlk_user_...")     # from your Profile page
client.download_dataset(173, "data/blood/")

train = np.load("data/blood/train_images.npz")   # ["image_id"], ["images"] (N, 28, 28, 3) uint8
y_train = pd.read_csv("data/blood/y_train.csv")  # image_id,label — labels are cell_A ... cell_H
test = np.load("data/blood/test_images.npz")     # ["image_id"], ["images"]

# train on the full training split, predict, write submission.csv with the
# columns image_id,label — one row per test image_id — then:
client.submit(competition_id=173, files=["submission.csv"])
print(client.leaderboard(173).head())
```

The board already tells you what a good score is. The row
`rf-pixel-baseline` — the 28×28×3 pixels flattened into a random forest — scores
**F1-macro = 0.749**, and that is the bar. Five different students sit within a
thousandth of it, at 0.7498, so clearing it is table stakes rather than an
achievement; the best submission on the board scores **0.959**. Ignore the row
named `__benchmark__` at **0.041** — it is a stale creator reference, eighteen
times below the pixel baseline and below anything a wired-up pipeline produces.
It is not the target.

Your Part C recipe transfers directly — a pretrained backbone wants a bigger
input, so upsample the thumbnails — and the class imbalance is the one *Training
CNNs* warned about: F1-macro is the unweighted mean over the eight types, so the
rare ones decide your score. If `download_dataset` answers 404, the dataset
objects are not reachable from your account; report it on the course channel
rather than assuming you mistyped the id.

---

## Pull request

The description states:

- the dataset, the class counts, and how the split was made
- a table: from-scratch, frozen-backbone, fine-tuned — test accuracy and
  macro-F1 for each
- what pretraining was worth, in points, on your data
- the augmentation ablation table and one sentence explaining the loser
- the two most-confused classes and your reading of why
- your competition 173 score, and the leaderboard row it beat
- one thing you would change with a second GPU hour

---

## Grading

| Criterion | Weight |
|---|---|
| Correct splits, stratified, test touched once | 15% |
| From-scratch baseline trained and reported | 20% |
| Fine-tuning: two phases, discriminative rates, best checkpoint kept | 20% |
| Augmentation ablation, three runs, one variable changed | 15% |
| Four tests passing | 15% |
| Confusion matrix read, not just pasted | 15% |

---

## Automatic deductions

- images committed to git
- the eval transform contains a random augmentation
- test set used for model selection, or evaluated more than once
- hard-coded ImageNet mean/std beside a checkpoint that ships its own
- a bare `except` around the training loop
- accuracy reported on an imbalanced set with no per-class metric

---

## Carry it forward

You now have a working image classifier, a baseline it beats, and the evidence
for both. Session 6 keeps the backbone and changes the head: detection and
segmentation are the same features, decoded differently.

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] `assert set(train_paths) & set(test_paths) == set()` passes, and the split and its seed are written to disk (Part A)
- [ ] `runs/baseline.json` exists and holds the from-scratch test accuracy and macro-F1 (Part B)
- [ ] The fine-tuned run kept the best-validation checkpoint, and the test split was scored exactly once for each of the three models (Part C)
- [ ] `REPORT.md` holds the three-row ablation table and names the run that lost to `basic` (Part D)
- [ ] All four tests of Part E pass, `test_eval_transform_is_deterministic` included
- [ ] `runs/confusion_matrix.png` is committed with class names on both axes, and `REPORT.md` names the class with the worst recall
- [ ] My submission is on the leaderboard of Blood Cell Classification (#173) — `client.leaderboard(173)` lists my agent name
- [ ] My score beats the baseline: **F1-macro > 0.749**, the `rf-pixel-baseline` row

If the last two are not ticked you have not finished the lab, however good the
code is.
