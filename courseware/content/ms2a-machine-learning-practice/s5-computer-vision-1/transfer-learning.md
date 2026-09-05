# Transfer Learning

You will not train a vision model from scratch. Someone spent thousands of GPU
hours learning what edges, textures and object parts look like; that work is a
free download, and starting from it is the difference between needing a million
labelled images and needing two thousand.

<!-- notes: 35 minutes and the most directly useful lesson of the session — the
lab is exactly this. Land three things: the head-then-backbone two-phase recipe,
discriminative learning rates, and that frozen requires_grad does not freeze
BatchNorm. -->

---

## Why ImageNet features transfer

The first convolutional layer of a network trained on cats and the first layer
of a network trained on chest X-rays converge to the same thing: oriented edges
and colour blobs. Nobody designed that; it falls out of natural image
statistics.

Features go from **general to specific** with depth. The early layers encode
properties of images; only the last layers encode properties of *ImageNet's
thousand classes*. Transfer learning keeps the first part and rebuilds the
second.

ImageNet-1k is 1.28 million photographs over 1000 categories. It is a broad
enough sample of the visual world that its early features are close to
task-independent.

---

## Two regimes

| | Feature extraction | Fine-tuning |
|---|---|---|
| Backbone weights | frozen | updated |
| Trained parameters | the head only | all, or the last blocks |
| Data needed | 50–100 per class | 500+ per class |
| Cost | minutes on a CPU | an hour on a GPU |
| Overfitting risk | very low | real |
| Typical gain over scratch | large | larger |

They are the two ends of one axis, and the standard recipe is to do both: train
the head with the backbone frozen, then unfreeze and continue at a lower rate.

---

## Feature extraction

```python
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(2048, num_classes)          # new layer, requires_grad=True
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
```

Replacing `model.fc` after freezing works because the new module is created with
`requires_grad=True` by default. Passing only `model.fc.parameters()` to the
optimizer is belt and braces — and it is the version that fails loudly if you
freeze the wrong thing.

With under 500 images total, stop here. Run the frozen backbone once over the
dataset, cache the 2048-dimensional vectors, and fit a `LogisticRegression` on
them. It takes seconds, it cannot overfit much, and it is a baseline that many
fine-tuning runs fail to beat.

---

## Fine-tuning

```python
for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
])
```

Unfreeze from the top down: the head first, then the last block, then earlier
ones if validation is still improving. Every unfreeze needs a lower learning
rate than the one before, because those weights are already good and a large
step destroys them.

The failure has a name — **catastrophic forgetting**. Fine-tune all of ResNet-50
at `lr=1e-3` on 2,000 images and the first epoch erases the pretrained features;
you end up worse than training from scratch, having paid for the download.

---

## Discriminative learning rates

$$
\eta_l = \eta_L \, \gamma^{\,L-l}
$$

One learning rate per depth group, decaying towards the input with
$\gamma \approx 0.3$. Layer 1 needs almost no change; the head needs to be
learned from nothing.

| Group | Rate |
|---|---|
| `conv1`, `layer1` | 1e-5 |
| `layer2`, `layer3` | 3e-5 |
| `layer4` | 1e-4 |
| `fc` (new head) | 1e-3 |

Two groups — backbone and head, a factor of 10 apart — captures most of the
benefit. Four is a refinement, not a requirement.

---

## Freezing does not freeze BatchNorm

```python
def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()                      # stop updating running_mean/var
```

`requires_grad = False` stops the *gradients*. BatchNorm's `running_mean` and
`running_var` are buffers, not parameters — they are updated in the forward pass
whenever the module is in training mode, frozen or not.

So a "frozen" backbone in `model.train()` is quietly drifting its normalization
statistics towards your small dataset, and the frozen weights no longer match
them. Symptom: validation accuracy that is much worse than training accuracy
from the very first epoch, with no other sign of overfitting.

Call `freeze_bn(model)` after every `model.train()` while the backbone is
frozen.

---

## How much data

| Images per class | Strategy |
|---|---|
| < 50 | frozen features + logistic regression; consider few-shot / CLIP |
| 50–200 | freeze backbone, train the head |
| 200–1000 | freeze up to `layer3`, fine-tune `layer4` + head |
| 1000–5000 | full fine-tune, discriminative rates, heavy augmentation |
| > 10,000 | full fine-tune; from-scratch becomes arguable but rarely wins |

These are the numbers for a domain close to natural photographs. Multiply them
by three to five for a far one.

---

## The domain gap

ImageNet is 8-bit RGB photographs of objects, taken by people, at roughly 224
pixels. Everything that is not that is a gap:

| Target | Gap | Does transfer help? |
|---|---|---|
| Product photos, pets, food | none | enormously |
| Document scans, screenshots | moderate | yes |
| Chest X-ray, histopathology | large — grayscale, 16-bit, texture | yes, less |
| Satellite multi-spectral | large — 13 bands, top-down, no canonical up | early layers only |
| Radar, ultrasound, depth maps | very large | marginal |

The early layers transfer across all of these, because edges are edges. The deep
layers transfer only as far as the objects resemble ImageNet's.

For a wide gap, freeze less and train longer — or pretrain on unlabelled data
from your own domain, which is usually plentiful even when labels are not.

---

## When transfer fails

- **Channel count mismatch.** Thirteen satellite bands do not fit a 3-channel
  stem. Keep three bands, or expand the stem weights by copying and rescaling —
  do not randomly initialise the whole first layer.
- **Resolution mismatch.** A stride-2 stem on a 32-pixel CIFAR image throws most
  of it away before layer 2.
- **Wrong preprocessing.** The single most common failure, below.
- **Genuinely novel input.** Spectrograms, point clouds, raw sensor grids — a
  small CNN from scratch can beat a mis-matched backbone.

Always run the from-scratch baseline. Without it you cannot say what transfer
was worth, only that the number is what it is.

---

## torchvision

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
preprocess = weights.transforms()       # the exact eval transform used in training
```

`weights.transforms()` is the important line. It returns the resize, crop and
normalization that checkpoint was trained under, so you never hard-code the
ImageNet six and never guess whether it was 224 or 232.

`pretrained=True` is deprecated and ambiguous: V1 and V2 weights for the same
architecture differ by nearly 3 points of top-1. Name the enum.

---

## timm

```python
import timm

model = timm.create_model("convnext_tiny", pretrained=True, num_classes=10)
cfg = timm.data.resolve_model_data_config(model)
train_tf = timm.data.create_transform(**cfg, is_training=True)
```

`timm` carries well over a thousand checkpoints — ConvNeXt, EfficientNetV2, ViT,
Swin, DINOv2 — behind one interface. `num_classes=10` replaces the head for you
and `resolve_model_data_config` gives you that checkpoint's preprocessing rather
than a generic one.

`timm.list_models(pretrained=True)` lists what is available;
`model.forward_features(x)` returns the backbone output instead of logits.

---

## The default recipe

1. Load a pretrained `resnet50` or `convnext_tiny` **and its own transforms**.
2. Replace the head. Freeze the backbone, `freeze_bn`, train the head for 3
   epochs at `lr=1e-3`.
3. Unfreeze. Continue for 10–15 epochs with `1e-4` on the backbone and `1e-3` on
   the head, cosine schedule, `RandomResizedCrop` + flip.
4. Keep the best-validation checkpoint. Touch the test set once, at the end.

> Report the from-scratch baseline next to the fine-tuned number, or the
> fine-tuned number means nothing.

That last line is the one the lab grades.

---

## Check yourself

1. Backbone frozen with `requires_grad = False`, model left in `model.train()`,
   and validation accuracy is far below training from the first epoch with no
   other sign of overfitting. What is drifting, and what is the one-line fix?

   **Answer.** BatchNorm's `running_mean` and `running_var`. They are buffers,
   not parameters, so `requires_grad` never touched them and they keep updating
   on every forward pass in training mode. Call `freeze_bn(model)` after every
   `model.train()`.

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   bn = nn.BatchNorm2d(3)
   for p in bn.parameters():
       p.requires_grad = False
   before = bn.running_mean.clone()
   bn(torch.randn(8, 3, 4, 4))                    # training mode, "frozen"
   print(torch.equal(bn.running_mean, before))    # -> False
   bn.eval()
   after = bn.running_mean.clone()
   bn(torch.randn(8, 3, 4, 4))
   print(torch.equal(bn.running_mean, after))     # -> True
   ```

3. You have 120 photographs per class of a product catalogue. What does the
   data table tell you to do — and what would you do instead with 40?

   **Answer.** At 50–200 per class: freeze the backbone and train the head only.
   Below 50: run the frozen backbone once, cache the feature vectors, and fit a
   `LogisticRegression` on them.
