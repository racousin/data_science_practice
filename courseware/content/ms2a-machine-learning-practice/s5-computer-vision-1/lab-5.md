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
