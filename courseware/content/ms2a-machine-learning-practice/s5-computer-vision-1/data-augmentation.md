# Data Augmentation

With a few thousand labelled pictures, augmentation moves the validation number
more than any architecture or optimizer choice. It generates new training images
from the ones you have, by applying transformations that do not change the
label.

<!-- notes: 30 minutes. Show a batch of augmented images on screen before
explaining any of it — half the room will spot an augmentation that destroys
their own label. The train/eval transform split is the slide that shows up in
the lab grading. -->

---

## The idea

A convolution is equivariant to translation, and pooling makes the network
roughly invariant to small shifts. Nothing in the architecture makes it
invariant to a flip, a rotation, a change of scale or of lighting: the network
has to learn those from examples, and a small dataset does not contain enough of
them.

Augmentation supplies them. Choose a family $\mathcal{T}$ of transformations
that preserve the label, and train on the expected loss over it:

In practice each epoch draws a fresh $t$ for every image, so the network never
sees exactly the same picture twice.

Every augmentation is a claim: *this transformation does not change the label.*
That claim is domain knowledge, and it is the cheapest domain knowledge you will
ever inject into a model.


![data-augmentation-image-augment.webp](assets/cv/data-augmentation-image-augment.webp)


---

## Geometric transforms

A geometric transform moves pixels without changing their values. Every one
below sends a pixel position $(x, y)$ to $(x', y')$ through eight parameters —
a single $3 \times 3$ matrix in homogeneous coordinates:

$$
x' = \frac{a x + b y + c}{g x + h y + 1}, \qquad
y' = \frac{d x + e y + f}{g x + h y + 1}
$$

$(a, b, d, e)$ carry rotation, scale, shear and flip, $(c, f)$ the translation,
and $(g, h)$ the perspective. With $g = h = 0$ the transform is affine: parallel
lines stay parallel.

![Horizontal flip, vertical flip and transpose](assets/cv/flip.png)

```python
T.RandomHorizontalFlip(p=0.5)
```

Horizontal flip is the most effective augmentation on natural images. Vertical
flip is right for satellite and microscopy, wrong for upright photographs.

---

## Rotation

![Rotations from −90 to +90 degrees](assets/cv/rotate.png)

```python
T.RandomRotation(degrees=15)
```

Rotation by $\alpha$ sets $a = e = \cos\alpha$, $b = -d = -\sin\alpha$. Small
angles model camera tilt: keep them to ±10–15 degrees for photographs. Full
rotation is correct only where there is no canonical orientation — satellite
tiles, cell images, astronomical plates.

Rotation exposes corners, and fills them. Black corners are a feature the
network will happily learn.

---

## Scale and crop

![Zoom at scale 0.9, 0.75 and 0.6](assets/cv/scale.png)

```python
T.RandomResizedCrop(224, scale=(0.7, 1.0))
```

The workhorse: a random sub-region covering 70–100% of the area, resized to
224. It varies scale, translation and framing in one operation. Push `scale` too
low and you crop the object out of the picture while keeping its label.

---

## Perspective

![A perspective warp of the same scene](assets/cv/perspective.png)

```python
T.RandomPerspective(distortion_scale=0.3, p=0.5)
```

Perspective ($g, h \neq 0$) models a change of viewpoint — documents, signage,
anything photographed at an angle by a handheld camera.

---

## Photometric transforms

Photometric transforms change pixel values, not positions: lighting, exposure,
white balance.

![A low-contrast image, then contrast stretching, histogram equalisation and adaptive equalisation (CLAHE)](assets/cv/ContrastEnhancement.png)

Histogram equalisation maps each level $r_k$ through the image's cumulative
histogram, spreading the output over $[0, L-1]$:

$$
s_k = (L - 1) \sum_{j=0}^{k} p_r(r_j)
$$

```python
T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
```

As augmentation the change is sampled, not fixed. Keep `hue` small: 0.05 is a
different bulb, 0.5 is a different object.

---

## When an augmentation is wrong

| Augmentation | Breaks |
|---|---|
| Horizontal flip | text and digits (`b`/`d`, `2`/`5`), medical laterality |
| Vertical flip | any photograph with a horizon in it |
| Large rotation | scene photographs, road signs, handwriting |
| Colour jitter | anything where colour *is* the label: ripeness, rust, skin lesions |
| Random crop | small objects, or any task where context is the signal |

> An augmentation that changes the correct label is not regularization. It is
> label noise you paid compute to generate.

Look at fifty augmented images from your own pipeline before you train on them.
Two sections to append after "When an augmentation is wrong":

---

## Augmentation in a training pipeline

Transforms are applied **on the fly** by the `Dataset`, each time an image is
loaded. Nothing is stored on disk: every epoch draws new random parameters.

```python
import torch
import torchvision.transforms.v2 as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),   # geometric, random
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),            # photometric, random
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),          # uint8 [0,255] → float [0,1]
    T.Normalize(MEAN, STD),
])

eval_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),                             # deterministic
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(MEAN, STD),                        # same stats as train
])

train_ds = ImageFolder("data/train", transform=train_tf)
val_ds   = ImageFolder("data/val",   transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)
```

| | Train | Validation / test / prediction |
|---|---|---|
| Random transforms | yes | **no** |
| Resize / crop to model size | random crop | fixed resize + center crop |
| `ToDtype` + `Normalize` | yes | yes, **identical** |

Augmentation belongs to training only. Evaluating on augmented images measures
the model on a noisier distribution than the one it will face, and makes the
score change from run to run.

---

## Train vs eval: the split pitfall

`random_split` returns subsets of **one** dataset, which has **one** transform.
Split this way and the validation images are augmented too.

```python
# Wrong: val_ds inherits train_tf
full = ImageFolder("data/all", transform=train_tf)
train_ds, val_ds = torch.utils.data.random_split(full, [0.8, 0.2])
```

Build two datasets on the same files, each with its own transform, and split
the **indices**:

```python
from torch.utils.data import Subset

train_full = ImageFolder("data/all", transform=train_tf)
eval_full  = ImageFolder("data/all", transform=eval_tf)

idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(0))
n_train = int(0.8 * len(idx))

train_ds = Subset(train_full, idx[:n_train])
val_ds   = Subset(eval_full,  idx[n_train:])
```

The training loop itself does not change: augmentation happens inside the
`DataLoader`, before the batch reaches the model.

```python
for epoch in range(n_epochs):
    model.train()
    for x, y in train_loader:          # new random augmentations every epoch
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:        # clean, deterministic images
            ...
```
