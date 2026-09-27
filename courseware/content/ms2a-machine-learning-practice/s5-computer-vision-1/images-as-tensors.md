# Images as Tensors

At its core, a digital image is a structured grid of numerical values. Each point in this grid, called a pixel, contains information about color intensity.

<!-- notes: 25 minutes. Open by loading one JPEG live and printing shape, dtype
and min/max. The (H, W, C) vs (B, C, H, W) transpose is the single point they
must leave with. -->

---

## A pixel is a number

![Grayscale image as a grid of intensities](assets/cv/gray.png)

One channel, one number per position: intensity, 0 for black up to the maximum
for white. Nothing about a digital image is continuous — it is a sampled grid,
and the grid spacing is the resolution.


---

## Bit depth

![Bit depth: 2-bit, 8-bit and 16-bit intensity ranges](assets/cv/bit-depth-representation.png)

| Depth | Levels | Where you meet it |
|---|---|---|
| 1-bit | 2 | masks, scanned documents |
| 8-bit | 256 | JPEG, PNG, everything consumer |
| 16-bit | 65,536 | medical (DICOM), RAW photography, satellite |

---

## Colour channels

![RGB image decomposed into red, green and blue channels](assets/cv/rgb.png)

An RGB image is three co-registered grids stacked: the red, green and blue
intensity at each position.

Grayscale conversion is a weighted sum, not an average — `0.299R + 0.587G +
0.114B` — because the eye is most sensitive to green.

---

## More than three channels

![Multi-spectral satellite bands](assets/cv/satelite.png)

Satellite imagery carries 4 to 13
bands including near-infrared and thermal; medical data stacks modalities as
channels or adds a depth axis (Reference module, *3D CNNs*); video adds time.


---

## The shape convention



An image is a rank-3 tensor. Which axis comes first depends on who wrote the
library.

| Library | Layout | Dtype | Range |
|---|---|---|---|
| PIL, OpenCV, matplotlib, NumPy | `(H, W, C)` | `uint8` | 0–255 |
| PyTorch | `(C, H, W)`, batched `(B, C, H, W)` | `float32` | 0.0–1.0 |

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
