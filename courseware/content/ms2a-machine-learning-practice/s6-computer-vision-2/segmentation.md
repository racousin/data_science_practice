# Segmentation

A box says roughly where. A mask says exactly which pixels. For a tumour, a
road surface or a field boundary the difference is the whole point.

<!-- notes: 35 minutes. The two things to land are the encoder-decoder shape
(why you must upsample, and why skip connections are needed to do it well) and
Dice loss (why cross-entropy alone fails on 2%-foreground data). Show a
prediction on an imbalanced medical image trained with plain CE: it predicts
all background and reports 98% pixel accuracy. -->

---

## The task

![Per-pixel labels](assets/cv/segmentation-intro.png)

Segmentation is classification run once per pixel. The output has the spatial
dimensions of the input and one channel per class.

```python
logits = model(x)            # x: (B, 3, H, W)
logits.shape                 # (B, K, H, W)
pred = logits.argmax(1)      # (B, H, W), values in [0, K-1]
```

Nothing about the loss is new — it is cross-entropy over `H × W` positions
instead of one. What is new is the architecture needed to produce a
full-resolution output, and the metrics that judge it.

---

## Three tasks, not one

![Semantic, instance, panoptic](assets/cv/segmentation-types-comparison.jpg)

| Task | Output per pixel | Two touching cars |
|---|---|---|
| **Semantic** | class label | one "car" region |
| **Instance** | instance id, for countable objects only | two separate cars |
| **Panoptic** | class label *and* instance id, everywhere | two cars, plus road and sky |

Semantic segmentation cannot count. If the deliverable is "how many cells are
in this image", a semantic model is the wrong tool whatever its mIoU — two
adjacent cells merge into one blob. Panoptic distinguishes *things*
(countable: car, person) from *stuff* (uncountable: road, sky, grass).

---

## Ground truth is an index map

A mask is a single-channel integer image, not an RGB picture.

```python
mask = np.array(Image.open("mask.png"))
assert mask.ndim == 2 and mask.dtype == np.uint8
assert set(np.unique(mask)) <= set(range(K))
```

Two traps. Palette PNGs: `Image.open` already hands you mode `P`, and
`np.array` on it gives class indices — so the assert above passes as written.
The bug is the `.convert("RGB")` you copied out of your *image* loader into your
*mask* loader: it turns class 3 into a colour triplet and the assert fires on
`ndim`. Masks are never converted. And resizing — a mask must be resized with
**nearest-neighbour** interpolation, since bilinear averages neighbouring class
ids: an edge between class 3 and class 7 comes back containing 4 and 6, labels
that were never annotated.

```python
img  = TF.resize(img,  (512, 512))                       # bilinear, fine
mask = TF.resize(mask, (512, 512), InterpolationMode.NEAREST)
```

---

## The resolution problem

A classification backbone downsamples aggressively: a 512×512 input leaves a
16×16 feature map after five stride-2 stages — the right trade for a single
label, useless for a dense one. Two ways out, combined in every architecture:

1. **Downsample then upsample** — an encoder–decoder. Cheap, but the decoder
   has to invent the detail the encoder threw away.
2. **Do not downsample** — keep resolution and enlarge the receptive field
   another way. Expensive in memory, but no detail is lost.

---

## Upsampling

Three mechanisms, in increasing order of how much you should trust them.

```python
nn.Upsample(scale_factor=2, mode="bilinear")           # no parameters
nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # learned
nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1))
```

Transposed convolution — insert zeros between input pixels, then convolve — is
learned, and produces checkerboard artefacts whenever `kernel_size` is not
divisible by `stride`. Use `4, stride=2` or `2, stride=2`, never `3, stride=2`.
The third form avoids the artefact entirely at the same cost, and is the safer
default.

---

## U-Net

![U-Net](assets/cv/unet-architecture.png)

A symmetric encoder–decoder with **skip connections**: each decoder stage
concatenates the encoder feature map of the same resolution before convolving.

```python
d3 = up(bottleneck)
d3 = conv_block(torch.cat([d3, e3], dim=1))   # skip from the encoder
d2 = conv_block(torch.cat([up(d3), e2], dim=1))
out = final_conv(d2)                          # (B, K, H, W)
```

The encoder knows *what* is in the image but has lost *where*; the skip
connection hands the decoder back the high-resolution edges. Remove the skips
and the boundaries blur immediately — a one-line ablation worth running once so
you believe it. U-Net trains on a few hundred annotated images, which is why it
has owned medical imaging since 2015.

---

## Atrous convolution

Dilate the kernel instead of shrinking the image: insert `r - 1` zeros between
kernel taps. A 3×3 kernel at rate 6 sees a 13×13 region with nine weights and
no loss of resolution.

```python
nn.Conv2d(256, 256, 3, padding=6, dilation=6)
```

This decouples receptive field from stride, exactly the constraint that forces
encoder–decoders to exist.

Failure mode: a stack of layers all at the same dilation rate samples the same
lattice of pixels and ignores the rest — the *gridding* artefact. Vary the
rates.

---

## DeepLab and ASPP

![Atrous spatial pyramid pooling](assets/cv/deeplab-aspp.png)

ASPP applies parallel atrous convolutions at several rates plus a global
average pool, then concatenates. One module, several receptive-field sizes, so
objects at different scales are all covered.

```python
feats = torch.cat([
    conv1x1(x), atrous(x, 6), atrous(x, 12), atrous(x, 18),
    global_pool_and_upsample(x),
], dim=1)
```

DeepLabv3+ adds a light decoder to recover boundary detail. It and U-Net are
the two semantic baselines worth trying first; SegFormer is the transformer
alternative, often better on large-scale natural imagery.

---

## Mask R-CNN

![Mask R-CNN](assets/cv/mask-rcnn-architecture.png)

Instance segmentation is detection with a third head. Take Faster R-CNN, and
alongside the class and box heads add a small FCN that predicts a binary mask
inside each proposal.

```python
roi = roi_align(features, proposals)        # not roi_pool
classes, boxes = detection_head(roi)
masks = mask_head(roi)                      # (N, K, 28, 28)
```

The mask is predicted per class, low-resolution, then resized into the box. The
detail that matters is `roi_align`: RoI *pooling* quantises proposal
coordinates to integer feature-map cells, shifting masks by a few pixels —
tolerable for a box, fatal for a mask.

Everything that detection costs — annotation, NMS, mAP — instance segmentation
costs too, plus polygon annotation instead of boxes.

---

## SAM

Segment Anything (Meta, 2023) is a promptable segmentation model trained on a
billion masks. You give it a point, a box or a rough mask; it returns a
high-quality mask. It has **no class labels** — it segments, it does not name.

```python
predictor.set_image(image)
masks, scores, _ = predictor.predict(box=np.array([x1, y1, x2, y2]))
```

The practical use is as an annotation accelerator: a detector proposes boxes,
SAM turns each into a mask, a human corrects. That converts a polygon budget
into a box budget — roughly a five-fold saving.

---

## Metrics

![Dice versus IoU](assets/cv/dicevsiou.png)

Pixel accuracy is worthless on imbalanced data — 98% on a dataset that is 98%
background, from a model that predicts background everywhere. Use IoU per class
and average it:

$$
IoU_k = \frac{|P_k \cap G_k|}{|P_k \cup G_k|} \qquad mIoU = \frac{1}{K} \sum_{k=1}^K IoU_k
$$

Dice, the medical-imaging convention, weights the intersection twice:

$$
Dice_k = \frac{2 |P_k \cap G_k|}{|P_k| + |G_k|}
$$

They rank models identically — Dice is monotone in IoU — but Dice is always
the larger number. Report which one you used.

---

## Losses

Cross-entropy over pixels is the starting point:

$$
L_{CE} = -\frac{1}{HW} \sum_{i,j} \sum_{k} y_{ijk} \log \hat{y}_{ijk}
$$

It optimises *per-pixel* correctness, so when the foreground is 2% of the image
the fastest way to reduce it is to predict background everywhere. Dice loss
optimises *overlap*, which is scale-free in the foreground size:

$$
L_{Dice} = 1 - \frac{2 \sum_{i,j} p_{ij} g_{ij}}{\sum_{i,j} p_{ij} + \sum_{i,j} g_{ij}}
$$

The sums run over soft probabilities, not the argmax, so it is differentiable.

> Default to `CE + Dice`. Cross-entropy gives clean early gradients, Dice fixes
> the imbalance. A pure Dice loss is unstable in the first epochs.

Focal loss and class weights handle imbalance *between* classes rather than
against the background.

---

## What to actually use

| Situation | Choice |
|---|---|
| Medical, few hundred images | U-Net, `CE + Dice`, heavy augmentation |
| Natural scenes, many classes | DeepLabv3+ or SegFormer, pretrained |
| Need to count instances | Mask R-CNN or Mask2Former |
| Everything at once, panoptic | Mask2Former |
| Annotating a new dataset | SAM in the loop |

```python
import segmentation_models_pytorch as smp
model = smp.Unet("resnet34", encoder_weights="imagenet", classes=K)
```

A pretrained encoder in a U-Net is the highest-value default here: one
argument, and typically 5–10 mIoU points on a small dataset.

---

## Check yourself

1. Your model reports 98% pixel accuracy and 0.49 mIoU on data whose foreground
   is 2% of every image. What did it predict, and which loss do you reach for?

   **Answer.** Background everywhere — which is exactly 98% of the pixels, and
   the fastest way for per-pixel cross-entropy to fall. The mIoU gives it away:
   0.98 on the background class, 0.00 on the foreground, averaging to 0.49 over
   the two while accuracy still reads 98%. Switch to `CE + Dice`:
   Dice scores overlap in the foreground, which a background-only prediction
   cannot fake.

2. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from PIL import Image
   mask = Image.fromarray(np.array([[3, 3, 7, 7]], dtype=np.uint8))
   print(np.unique(np.array(mask.resize((8, 1), Image.BILINEAR))))   # -> [3 4 6 7]
   print(np.unique(np.array(mask.resize((8, 1), Image.NEAREST))))    # -> [3 7]
   ```

   **Answer.** Classes 4 and 6 were never annotated — bilinear resizing of a
   label map manufactured them at the boundary. Nearest-neighbour cannot invent
   a class, which is why masks are resized with it and images are not.

3. The deliverable is "how many cells are in this image". Why is semantic
   segmentation the wrong tool however high its mIoU?

   **Answer.** It labels pixels, not objects: two touching cells become one
   region and cannot be counted. Counting needs instance segmentation — Mask
   R-CNN or Mask2Former.
