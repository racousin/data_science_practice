# Images as Tensors

An image is a grid of numbers with a shape convention attached. Almost every
bug in a vision pipeline is a shape, a dtype, or a normalization that disagreed
with the model, and all three are visible in the first thirty seconds if you
look.

<!-- notes: 25 minutes. Open by loading one JPEG live and printing shape, dtype
and min/max. The (H, W, C) vs (B, C, H, W) transpose is the single point they
must leave with. -->

---

## A pixel is a number

![Grayscale image as a grid of intensities](/api/academic_courses/assets/lessons/85/gray.png)

One channel, one number per position: intensity, 0 for black up to the maximum
for white. Nothing about a digital image is continuous — it is a sampled grid,
and the grid spacing is the resolution.

Everything downstream is arithmetic on that grid.

---

## Bit depth

![Bit depth: 2-bit, 8-bit and 16-bit intensity ranges](/api/academic_courses/assets/lessons/85/bit-depth-representation.png)

| Depth | Levels | Where you meet it |
|---|---|---|
| 1-bit | 2 | masks, scanned documents |
| 8-bit | 256 | JPEG, PNG, everything consumer |
| 16-bit | 65,536 | medical (DICOM), RAW photography, satellite |

8-bit unsigned is the default assumption of every library you will use. A 16-bit
DICOM loaded as if it were 8-bit does not error — it wraps or clips, and you get
a black image with three bright pixels.

---

## Colour channels

![RGB image decomposed into red, green and blue channels](/api/academic_courses/assets/lessons/85/rgb.png)

An RGB image is three co-registered grids stacked: the red, green and blue
intensity at each position, each usually 8-bit. Colour is not extra semantics,
it is two extra grids.

Grayscale conversion is a weighted sum, not an average — `0.299R + 0.587G +
0.114B` — because the eye is most sensitive to green.

---

## More than three channels

![Multi-spectral satellite bands](/api/academic_courses/assets/lessons/85/satelite.png)

Nothing in a convolution cares that `C = 3`. Satellite imagery carries 4 to 13
bands including near-infrared and thermal; medical data stacks modalities as
channels or adds a depth axis (Reference module, *3D CNNs*); video adds time.

Only the first layer changes. `nn.Conv2d(13, 64, 3)` is as valid as
`nn.Conv2d(3, 64, 3)`.

---

## The shape convention

An image is a rank-3 tensor. Which axis comes first depends on who wrote the
library.

| Library | Layout | Dtype | Range |
|---|---|---|---|
| PIL, OpenCV, matplotlib, NumPy | `(H, W, C)` | `uint8` | 0–255 |
| PyTorch | `(C, H, W)`, batched `(B, C, H, W)` | `float32` | 0.0–1.0 |
| TensorFlow | `(B, H, W, C)` | `float32` | 0.0–1.0 |

PyTorch is channels-first because cuDNN convolutions were written that way.
OpenCV additionally stores channels as **BGR**, not RGB — a swap that shows up
as an image with blue skin and orange sky.

---

## The rule

> Convert once, at the boundary. `(H, W, C) uint8` goes in, `(C, H, W) float32`
> comes out, and nothing after that point ever sees a NumPy image again.

`transforms.ToTensor()` does the permute, the divide by 255 and the dtype cast
in one call. Doing any of the three by hand is how you end up dividing twice.

---

## Loading

```python
from PIL import Image
from torchvision import transforms

img = Image.open("cat.jpg").convert("RGB")   # PIL, (W, H), uint8
x = transforms.ToTensor()(img)                # (3, H, W), float32, [0, 1]
```

`.convert("RGB")` is not optional. Real datasets contain grayscale JPEGs, PNGs
with an alpha channel, and CMYK scans; without the convert you get a `(1, H, W)`
or `(4, H, W)` tensor and a channel-mismatch crash two hundred images into
training.

PIL reports size as `(width, height)`; the tensor is `(channels, height, width)`.
Both orders are correct — they are different orders.

---

## Normalization

```python
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

Per channel, subtract the mean and divide by the standard deviation:

$$
x' = \frac{x - \mu}{\sigma}
$$

Those six numbers are the channel statistics of the ImageNet training set. They
are not magic constants — they are the statistics the pretrained weights you are
about to download were fitted under.

---

## Why those numbers, and when to change them

A pretrained backbone expects its inputs centred where they were during
pretraining. Feed it `[0, 1]` instead of normalized input and accuracy drops by
several points with no error message anywhere.

| Situation | Statistics |
|---|---|
| Any ImageNet-pretrained backbone | the ImageNet six, always |
| Training from scratch on your own data | your own train-split mean/std |
| Non-RGB (satellite, medical) | your own, per band, computed on train only |

Computing the statistics on train + test is a leak. Session 2 made this point
for tabular features; it is the same point.

---

## Resizing

```python
transforms.Resize(256)          # shorter side -> 256, aspect ratio kept
transforms.CenterCrop(224)      # then take the middle 224 x 224
```

Networks with a global pooling head accept any input size, but batching does
not: a batch is a single tensor, so every image in it must share a shape.

`Resize((224, 224))` with a tuple forces both sides and **distorts** the aspect
ratio. `Resize(256)` with an integer scales the shorter side and preserves it.
Resize-then-crop is the standard because it keeps geometry honest at the cost of
the border.

---

## Interpolation

| Mode | Use |
|---|---|
| Nearest | segmentation masks and label maps — never averages classes |
| Bilinear | the default for photographs |
| Bicubic | upsampling, slightly sharper |
| Lanczos | offline downsampling of high-resolution originals |

Resizing a segmentation mask bilinearly invents class id 3.5 between a road and
a car. Masks are nearest-neighbour, always.

Downsampling 4000×3000 to 224×224 at load time, every epoch, on the CPU, is an
invisible way to make the GPU idle 80% of the time — resize once, offline, cache.

---

## What a dataset looks like on disk

```text
data/
  train/cat/0001.jpg   train/dog/0001.jpg
  val/cat/0002.jpg     val/dog/0002.jpg
  test/cat/0003.jpg    test/dog/0003.jpg
```

```python
from torchvision.datasets import ImageFolder
ds = ImageFolder("data/train", transform=train_tf)
print(ds.classes, ds.class_to_idx)
```

One directory per class, splits separated at the top level. `ImageFolder` sorts
class names alphabetically to assign indices, so renaming a directory silently
changes what your saved checkpoint means.

For metadata, per-image labels or multi-label tasks, write a manifest CSV of
`path, label, split` and a small `Dataset` that reads it. Same advice as
Session 1: media on disk, a table beside it.

---

## Verify at the boundary

```python
x, y = next(iter(train_loader))
assert x.shape == (32, 3, 224, 224), x.shape
assert x.dtype == torch.float32
assert -3.0 < x.min() < x.max() < 3.0     # normalized, not [0, 255]
```

Three assertions on the first batch. The third catches the most common failure
in this session: a transform pipeline in which `ToTensor` was forgotten or
`Normalize` was applied twice.

The image on the screen is a debugging tool, not a check. Print the numbers.
