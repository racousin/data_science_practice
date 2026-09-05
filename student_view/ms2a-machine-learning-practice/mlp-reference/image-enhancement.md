# Image Enhancement

Reference only — not covered in class. Classical preprocessing has largely been
superseded by learned features: a CNN with enough data discovers its own
contrast normalisation. Read it when data is scarce or genuinely degraded —
low-light photography, microscopy, medical scans, scanned documents — where the
classical operators still earn their place. Session 2 covers the tabular half
of preprocessing, Session 5 images as tensors.

<!-- notes: Self-study, linked from Sessions 2 and 5. Never lectured. -->

---

## Enhancement is not augmentation

The distinction students get wrong, so it goes first.

| | Enhancement | Augmentation |
|---|---|---|
| Applied to | every image, train and test | training images only |
| Determinism | fixed, same result every call | random, different each epoch |
| Purpose | make the signal visible | make the model invariant |
| Belongs in | the dataset / preprocessing step | the training transform |

```python
train_tf = T.Compose([clahe, T.RandomResizedCrop(224),
                      T.RandomHorizontalFlip(), T.ToTensor(), norm])
eval_tf  = T.Compose([clahe, T.Resize(256),
                      T.CenterCrop(224), T.ToTensor(), norm])
```

`clahe` appears in both; the random operations appear in only one.

**Failure mode.** `RandomRotation` in the evaluation transform gives a
validation score with variance you cannot explain, biased low. The mirror
failure — histogram equalisation at training time only — gives a model that has
never seen the distribution it is tested on.

---

## Point operations

A point operation maps each pixel intensity through a function, ignoring its
neighbours. **Histogram equalisation** uses the cumulative distribution of
intensities as that function, spreading the used values over the full range:

$$
s_k = (L-1) \sum_{j=0}^{k} p_r(r_j)
$$

**Gamma correction** is the parametric version, $I_{out} = I_{in}^{\gamma}$:
below 1 it brightens shadows, above 1 it darkens them.

![Contrast enhancement](/api/academic_courses/assets/lessons/128/ContrastEnhancement.png)

---

## CLAHE

Global equalisation fails when an image has regions at different exposures — it
blows out the bright half to fix the dark half. CLAHE equalises on a grid of
tiles and interpolates between them, clipping each histogram so flat regions do
not amplify into noise.

```python
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

Equalise the **L** channel in LAB, not the three RGB channels separately.
Per-channel equalisation changes the ratios between channels, so it changes the
hue: skin turns green, and every downstream colour feature is now a lie.

---

## Spatial filtering

These replace each pixel with a function of its neighbourhood.

| Filter | Form | Removes / adds | Cost |
|---|---|---|---|
| Gaussian blur | $e^{-(x^2+y^2)/2\sigma^2}$, normalised | Gaussian noise | blurs edges |
| Median | order statistic, no kernel | salt-and-pepper | slow, preserves edges |
| Unsharp mask | $I + \lambda (I - G * I)$ | apparent sharpness | amplifies noise |
| Sobel / Laplacian | fixed derivative stencils | edge maps | discards intensity |

```python
denoised = cv2.medianBlur(img, ksize=3)
sharp = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0)
```

The pairing that matters: **Gaussian blur cannot remove salt-and-pepper noise**
— averaging an outlier spreads it across the window instead of deleting it. A
median filter discards it, because an order statistic ignores magnitude. Edge
kernels are worth knowing because a CNN's first layer learns approximations of
them; feeding a network a precomputed Sobel map is almost always a downgrade.

---

## Geometric transforms and interpolation

Rotation, scaling, translation, shear and perspective are affine or projective
maps of pixel coordinates. The mapped coordinates are not integers, so every
one of them forces an interpolation choice.

| Mode | Use for |
|---|---|
| `nearest` | segmentation masks, class-index maps, any categorical raster |
| `bilinear` | the default for photographic images |
| `bicubic` | upscaling, when smoothness matters more than speed |
| `area` | downscaling by a large factor — averages instead of sampling |

**Rule.** Never resize a label mask with anything but nearest neighbour.
Bilinear interpolation between class 3 and class 5 produces class 4, which
exists in your tensor and not in your problem. The model learns to predict a
class that never occurs, and the metric quietly loses points at every boundary.

---

## When to bother

Reach for enhancement when the degradation is systematic and the dataset is
small: consistent under-exposure, a scanner with known noise characteristics,
microscopy where contrast varies with the sample rather than the label. There
the operator injects prior knowledge a few thousand images cannot supply.

Skip it on tens of thousands of natural images. The network learns a better
normalisation than you can specify, and a fixed enhancement step is one more
thing to keep identical between training and production. Whatever you choose,
pin it in the dataset code — not in a notebook cell that ran once.
