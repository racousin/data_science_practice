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

$$
\min_\theta \; \mathbb{E}_{(x,y)} \; \mathbb{E}_{t \sim \mathcal{T}}
\big[\, \ell\big(f_\theta(t(x)),\, y\big) \big]
$$

In practice each epoch draws a fresh $t$ for every image, so the network never
sees exactly the same picture twice.

Every augmentation is a claim: *this transformation does not change the label.*
That claim is domain knowledge, and it is the cheapest domain knowledge you will
ever inject into a model.

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

---

## Train and eval transforms

```python
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                      T.RandomHorizontalFlip(), T.ToTensor(), normalize])
eval_tf  = T.Compose([T.Resize(256), T.CenterCrop(224),
                      T.ToTensor(), normalize])
```

Augmentation is a **training-time** operation. Evaluation must be deterministic:
the same image must give the same prediction on every run. Two dataset objects,
two transforms, and nothing `Random` in the eval one. Reusing `train_tf` for
validation does not crash; it makes validation noisy and pessimistic from the
first epoch.

---

## mixup and cutmix

$$
\tilde{x} = \lambda x_a + (1 - \lambda) x_b, \qquad
\tilde{y} = \lambda y_a + (1 - \lambda) y_b, \qquad \lambda \sim \mathrm{Beta}(\alpha, \alpha)
$$

**mixup** blends two images and their labels with the same $\lambda$.
**cutmix** pastes a rectangle of one image into another and sets $\lambda$ to
the pixel fraction.

```python
lam = np.random.beta(0.2, 0.2)
idx = torch.randperm(x.size(0))
out = model(lam * x + (1 - lam) * x[idx])
loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[idx])
```

Both reduce overconfidence and need long schedules to pay off; on a 20-epoch
fine-tune they usually make things worse. Reach for them after the basics.
