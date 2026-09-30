# Lab 6 — Detect or Generate

Two branches, one grading table. Either localise objects and measure it
honestly, or generate images and measure that honestly. The measurement is the
graded part in both cases.

**Time:** 45 minutes. **Deliverable:** a merged PR in your project repository.

<!-- notes: 45 minutes and both branches are ambitious for it — insist that the
model be small and the dataset tiny. A 200-image detection set and a 32x32 VAE
both finish inside the room. Warn early: anyone who starts a 512x512 diffusion
model from scratch will have nothing to show. -->

---

## Starter notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s6-aquarium-detection.ipynb)

[Challenge 195](https://ml-arena.com/viewchallenge/195), *Aquarium Detection*:
7 marine species in 190 test images, boxes and classes, scored by mAP50-95.
The notebook fine-tunes YOLOv8n and submits; start it on a T4 at the
beginning of the lab so it trains while you work (Part F).

---

## Setup

Branch off `main`. Session 5's transforms and training loop are reused, not
rewritten.

```text
src/vision2/
    __init__.py
    data.py         <- dataset + transforms, masks resized NEAREST
    model.py        <- build the model, one function
    train.py        <- the loop, seeded
    metrics.py      <- iou / dice / map, or fid / recon error
tests/test_vision2.py
RESULTS.md
```

Weights, datasets and generated images stay out of git. Commit the figures you
reference in `RESULTS.md`, nothing else.

Branch A needs `segmentation_models_pytorch` (which pulls `timm`) or
`ultralytics`; install it **before** the session —
`smp.Unet("resnet18", encoder_weights="imagenet")` downloads its weights on the
first call and will not work offline.

---

## Part A — Choose a branch and get the data (5 min)

**Branch A — Detect or segment.** A small annotated set you can obtain in one
line: `OxfordIIITPet(root=..., target_types="segmentation", download=True)` —
take 2–5 breeds and 100–300 images. Do not annotate your own images here: at the
5–15 seconds a box costs in the detection lesson, 300 images is an hour, not
five minutes. If you want your own data, annotate it before the session.
`torchvision.datasets` ships no Penn-Fudan loader and Roboflow needs an account,
so neither is a five-minute start.

**Branch B — Generate.** MNIST, Fashion-MNIST, CIFAR-10 or a single-class
subset. 32×32 or 28×28. Resist anything larger.

Before writing code, record in `RESULTS.md`: the dataset, its size, the split,
and the number you intend to report. Pick the metric *before* you see it.

---

## Part B, branch A — Fine-tune a localiser (20 min)

Start from pretrained weights. Training a detector from scratch in 20 minutes
is not a thing.

```python
model = smp.Unet("resnet18", encoder_weights="imagenet", classes=K)
# or: YOLO("yolov8n.pt").train(data="data.yaml", epochs=20, imgsz=416)
```

Requirements:

- a real train/val split by **image**, never by crop or patch
- masks resized with nearest-neighbour interpolation, asserted in `data.py`
- box convention asserted at the boundary — `x2 > x1` and `y2 > y1`
- `torch.manual_seed(...)` set once, and the seed recorded in `RESULTS.md`

If the loss will not move, overfit ten images to near-zero first. A model that
cannot overfit ten images has a data bug, not a capacity problem.

---

## Part B, branch B — Train a generator (20 min)

A convolutional VAE with a 16–64 dimensional latent, or a small DDPM U-Net on
28×28. Both fit in the time budget on a modest GPU; the VAE fits on a CPU.

```python
recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
loss = recon + beta * kl
```

Requirements:

- the reduction convention above, not `reduction="mean"` — state your `beta`
- a fixed noise batch decoded and saved at every epoch, as a filmstrip
- for diffusion: `T`, the beta schedule and the sampler recorded explicitly
- `torch.manual_seed(...)` set once, and the seed recorded in `RESULTS.md`

---

## Part C — Measure (10 min)

Branch A reports **mAP@0.5** for detection or **mean IoU per class** for
segmentation, on the validation split, with the metric implemented in
`metrics.py` — not only read off a framework's progress bar. Then the part that
carries the marks: pick the **five worst validation images** and say what went
wrong. Small objects? Occlusion? An inconsistently annotated class? One
sentence each, with the figure.

Branch B reports reconstruction error on held-out data plus a **latent
interpolation figure**: ten decoded frames between two encoded validation
images. State whether it morphs or crossfades — a crossfade means the latent
space is not structured and is a finding, not a failure.

An unconditional-sample grid is required in both cases for branch B.

---

## Part D — Required tests (10 min)

The same three for both branches. Docstrings only here; you write the bodies.

```python
def test_model_output_shape_and_dtype():
    """One batch through the model returns the documented shape and float32."""

def test_training_step_is_deterministic():
    """Two runs from the same seed give bit-identical loss after 3 steps."""

def test_iou_identical_and_disjoint():
    """iou(b, b) == 1.0 and iou of two disjoint boxes == 0.0."""
```

The third is the one that matters. Substitute the metric your branch reports:
Dice on identical and disjoint masks, or a KL term that is exactly zero when
`mu = 0` and `logvar = 0`. A metric you never tested is a number you cannot
defend.

The determinism test fails most often. Seeding `torch` is not enough — the
`DataLoader` worker seeds, `numpy`, `random` and shuffling must all be pinned.

---

## Part E — RESULTS.md (5 min)

- the dataset, its size, and how the split was made
- the model, whether it was pretrained, and on what
- the metric, with its exact definition and threshold
- the number, on validation, with the seed that produced it
- the figures: worst-five with commentary, or interpolation plus sample grid
- one paragraph: what you would change with another two hours

A metric without its threshold and split is not a result: `mAP` alone is
meaningless, `mAP@0.5 on 40 held-out images, seed 0` is a claim.

---

## Part F — Put it on the board (5 min)

This module's attached competition is **Aquarium Detection**,
`challenge_id=195`: 448 labelled aquarium photos, 7 classes (fish,
jellyfish, penguin, puffin, shark, starfish, stingray), and 190 test images
whose boxes you submit. It is ranked on **mAP50-95**, the metric of Part C
averaged over IoU thresholds 0.50 to 0.95, computed by `detection_metric.py`,
which ships with the data so your validation number is the leaderboard's.

The [starter notebook](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s6-aquarium-detection.ipynb)
fine-tunes YOLOv8n for 30 epochs on a T4 and submits. Start it at the
beginning of the lab, so it trains while you work on your branch.

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")      # from your Profile page
print(client.leaderboard(195))                        # the benchmark and who beat it
```

The benchmark row is the notebook's own recipe. Record your validation
mAP50-95, your leaderboard score and the benchmark's in `RESULTS.md`. Branch A
students may use this dataset for the whole lab: `train.zip` is already in
YOLO format.

---

## Pull request

The description states:

- which branch, and why
- the number, and what a sensible baseline for it would be
- one failure the figures made visible that the metric did not
- one thing you asserted at a data boundary, and what it caught

---

## Grading

| Criterion | Weight |
|---|---|
| Model trains, from a sane pretrained or small architecture | 15% |
| Split and data handling correct, asserted at the boundary | 15% |
| Metric implemented in `metrics.py`, not just read off a log | 20% |
| Required figures present and legible | 15% |
| Three tests passing, including the metric test | 20% |
| `RESULTS.md` complete, seed and threshold stated | 15% |

---

## Automatic deductions

- weights, datasets or generated images committed to git
- a metric reported without its split, threshold or seed
- masks resized with bilinear interpolation
- a bare `except` around a training step
- validation images seen during training, in any form
- a generated sample grid presented without stating the epoch it came from

---

## Carry it forward

Both branches produce the same transferable thing: a metric you implemented,
tested, and can defend under questioning. That is what the project asks for.
Session 7 changes the modality to text; the discipline is identical.

---

## Did you validate this session?

- [ ] `uv sync && uv run pytest` is green on a fresh clone
- [ ] `RESULTS.md` names the dataset, its size, the split and the metric — written down before the first number was produced (Part A)
- [ ] The model trained from pretrained weights or a small architecture, and the seed is recorded in `RESULTS.md` (Part B)
- [ ] The reported metric is computed by my own code in `metrics.py`, not read off a framework's progress bar (Part C)
- [ ] Branch A: the worst-five figure is committed with one sentence per image. Branch B: the ten-frame interpolation and the sample grid are committed, and `RESULTS.md` says morph or crossfade (Part C)
- [ ] All three tests of Part D pass, the metric test on an identical and a disjoint pair included
- [ ] `RESULTS.md` states the number with its split, threshold and seed (Part E)
- [ ] My submission is on the leaderboard of Aquarium Detection (#195) — `client.leaderboard(195)` lists it (Part F)
- [ ] My validation mAP50-95 and my leaderboard score are written in `RESULTS.md` next to the benchmark's (Part F)

Every row is a fact about your repository or the board that you can settle
yourself, without asking anyone.
