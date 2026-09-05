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

---

## Part A — Choose a branch and get the data (5 min)

**Branch A — Detect or segment.** A small annotated set: Penn-Fudan
pedestrians, Oxford-IIIT Pet masks, a Roboflow public set, or 100–300 images
you annotate yourself. 2–5 classes, no more.

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
