# YOLO — One-Stage Detection End to End

The previous lesson surveyed how detectors evolved and how they are scored.
This one builds one detector completely, YOLO, from the input tensor to the
final boxes, then covers what changed in modern versions. IoU, NMS and mAP
are used here as defined in *Object Detection*.

<!-- notes: 45 minutes. The core is the YOLOv1 block (six slides): input,
output tensor, target assignment, loss, decoding. Write the 7x7x30 tensor on
the board and keep it there. Modern YOLO is explained as changes to v1, so do
not start it until the v1 loss is understood. End with the exercise, done by
hand. -->

---

## YOLO: one pass, one tensor

![YOLOv1: CNN, S×S grid, boxes and class map, final detections](assets/cv/yolo-grid.png)

YOLOv1 (Redmon et al., 2016) handles detection as a single regression. The
image is divided into an $S \times S$ grid. Each cell predicts $B$ boxes, a
confidence for each box, and one class distribution. The output is a fixed
tensor:

$$
f_\theta : \mathbb{R}^{3 \times 448 \times 448} \longrightarrow
\mathbb{R}^{S \times S \times (5B + C)}
$$

On Pascal VOC, $S = 7$, $B = 2$, $C = 20$, so the output is
$7 \times 7 \times 30$: 1470 numbers that encode $7 \cdot 7 \cdot 2 = 98$
candidate boxes.

| | What | Value (VOC) |
|---|---|---|
| **Learnt** | $\theta$: 24 conv layers + 2 FC layers | the first 20 convs pretrained on ImageNet |
| **Fixed** | grid $S$, boxes per cell $B$, classes $C$ | 7, 2, 20 |
| **Fixed** | loss weights $\lambda_{coord}, \lambda_{noobj}$ | 5, 0.5 |

<!-- notes: The figure is Fig. 2/3 of the paper glued together: backbone on
the left, the 98 boxes (top) and the per-cell class map (bottom), combined
into the final detections. Point at each part as it appears in the next
slides. -->

---

## YOLOv1: the network

| # | Layer | Output shape | Learnt parameters |
|---|---|---|---|
| 0 | input image | $3 \times 448 \times 448$ | — |
| 1 | 24 conv layers, leaky ReLU, 4 max-pools + 2 stride-2 convs | $1024 \times 7 \times 7$ | conv kernels and biases |
| 2 | flatten | $50176$ | — |
| 3 | FC + leaky ReLU (+ dropout 0.5) | $4096$ | $50176 \times 4096 \approx 205.5$M |
| 4 | FC, linear | $1470$ | $4096 \times 1470 \approx 6.0$M |
| 5 | reshape | $7 \times 7 \times 30$ | — |

The downsampling factor is $448 / 7 = 64$: each cell covers a
$64 \times 64$ pixel patch. Because the head is fully connected, every cell's
prediction can use the whole image, not only its patch.

The last layer is linear. All the values of the output are produced by the
same regression; what gives them their meaning is the loss.

---

## YOLOv1: the 30 numbers of a cell

Each cell outputs:

$$
\big[\, (x, y, w, h, \hat{C})_1,\; (x, y, w, h, \hat{C})_2,\;
\Pr(c_1 \mid \mathrm{obj}), \dots, \Pr(c_{20} \mid \mathrm{obj}) \,\big]
$$

$B \cdot 5 = 10$ box numbers, then $C = 20$ class probabilities.

| Output | Meaning | Range |
|---|---|---|
| $x, y$ | centre of the box, offset **inside the cell** | $[0, 1]$ |
| $w, h$ | size of the box, relative to the **whole image** | $[0, 1]$ |
| $\hat{C}$ | confidence $= \Pr(\text{obj}) \cdot \mathrm{IoU}^{\text{truth}}_{\text{pred}}$ | $[0, 1]$ |
| $\Pr(c \mid \text{obj})$ | class distribution, **one per cell**, shared by the $B$ boxes | sums to 1 |

The confidence has two roles. It is 0 when the cell contains no object. When
it does, it should equal how well this box fits the object. Class
probabilities are conditional on an object being present, and because there
is one distribution per cell, the two boxes of a cell always have the same
class.

---

## YOLOv1: target assignment

Given a ground-truth box with centre $(c_x, c_y)$ and size $(w, h)$ in pixels,
on an image $W \times H$:

1. **Responsible cell.** The cell that contains the centre:
   $\text{col} = \lfloor c_x S / W \rfloor$, $\text{row} = \lfloor c_y S / H \rfloor$.
2. **Regression target** in that cell:
   $$
   x^* = \frac{c_x S}{W} - \text{col}, \quad y^* = \frac{c_y S}{H} - \text{row}, \quad
   w^* = \frac{w}{W}, \quad h^* = \frac{h}{H}
   $$
3. **Responsible predictor.** Among the $B$ boxes of the cell, the one whose
   current prediction has the highest IoU with the ground truth.

This defines the indicators used in the loss:

- $\mathbb{1}_{ij}^{obj} = 1$ if predictor $j$ of cell $i$ is responsible
  for an object, and 0 otherwise;
- $\mathbb{1}_{ij}^{noobj} = 1 - \mathbb{1}_{ij}^{obj}$;
- $\mathbb{1}_{i}^{obj} = 1$ if an object's centre falls in cell $i$.

As in the paper, $i \in \{0, \dots, S^2 - 1\}$ indexes the cells (flattened) and $j \in \{0, \dots, B-1\}$ the predictors of a cell. The
assignment changes during training: the predictor that fits best becomes
responsible, so the $B$ predictors of a cell specialise in different sizes
and aspect ratios.

---

## YOLOv1: the loss

$$
\mathcal{L} = \mathcal{L}_{xy} + \mathcal{L}_{wh} + \mathcal{L}_{obj} + \mathcal{L}_{noobj} + \mathcal{L}_{cls}
$$

$$
\mathcal{L}_{xy} = \lambda_{coord} \sum_{i=0}^{S^2-1} \sum_{j=0}^{B-1} \mathbb{1}_{ij}^{obj}
\left[(x_i - x_i^*)^2 + (y_i - y_i^*)^2\right]
$$

$$
\mathcal{L}_{wh} = \lambda_{coord} \sum_{i} \sum_{j} \mathbb{1}_{ij}^{obj}
\left[\left(\sqrt{w_i} - \sqrt{w_i^*}\right)^2 + \left(\sqrt{h_i} - \sqrt{h_i^*}\right)^2\right]
$$

$$
\mathcal{L}_{obj} = \sum_{i} \sum_{j} \mathbb{1}_{ij}^{obj} \left(\hat{C}_i - C_i^*\right)^2,
\qquad
\mathcal{L}_{noobj} = \lambda_{noobj} \sum_{i} \sum_{j} \mathbb{1}_{ij}^{noobj} \hat{C}_i^{\,2}
$$

$$
\mathcal{L}_{cls} = \sum_{i} \mathbb{1}_{i}^{obj} \sum_{c=1}^{C} \left(p_i(c) - p_i^*(c)\right)^2
$$

---

## YOLOv1: the five terms

| Term | Supervises | Why this form |
|---|---|---|
| $\mathcal{L}_{xy}$, centre | responsible box | $\lambda_{coord} = 5$: box accuracy is weighted more than classification |
| $\mathcal{L}_{wh}$, size | responsible box | $\sqrt{\cdot}$: an error of 10 px matters more on a small box than on a large one |
| $\mathcal{L}_{obj}$ | responsible box | target $C^* = \mathrm{IoU}(\text{pred}, \text{truth})$ |
| $\mathcal{L}_{noobj}$ | all other predictors (97 of 98 with one object) | $\lambda_{noobj} = 0.5$: prevents the many empty cells from pushing all confidences to 0 |
| $\mathcal{L}_{cls}$ | cells that contain an object centre | target $p^*$ is one-hot |

Everything is squared error, which keeps the loss simple. Later versions
replace each term with a better-suited loss, but keep this five-part
structure.

---

## YOLOv1: decoding at inference

**1. Class-specific score** for each of the 98 boxes and each class:

$$
s_{c} = \Pr(c \mid \text{obj}) \cdot \Pr(\text{obj}) \cdot \mathrm{IoU}
= \Pr(c \mid \text{obj}) \cdot \hat{C} \;\in [0, 1]
$$

**2. Cell coordinates to pixels**, for box $(x, y, w, h)$ in cell
$(\text{row}, \text{col})$:

$$
c_x = \frac{(\text{col} + x)\, W}{S}, \quad c_y = \frac{(\text{row} + y)\, H}{S}, \quad
w_{px} = w\, W, \quad h_{px} = h\, H
$$

then to corners, $x_1 = c_x - w_{px}/2$ and so on (*Object Detection*).

**3. Threshold**: keep the pairs (box, class) with $s_c > \tau$.

**4. NMS**, per class, as in *Object Detection*: `torchvision.ops.batched_nms(boxes_xyxy, scores, classes, iou_threshold=0.5)`.

| Step | Tensor | Shape |
|---|---|---|
| input | image | $3 \times 448 \times 448$ |
| backbone | feature map | $1024 \times 7 \times 7$ |
| head | raw output | $7 \times 7 \times 30$ |
| split | boxes, confidences, class probabilities | $98 \times 4$, $98$, $49 \times 20$ |
| scores | $s_c$ for every box and class | $98 \times 20$ |
| threshold $\tau$ | candidates | $K \times 6$: box, class, score |
| NMS | detections | $N \times 6$, $N \le K$ |

---

## Limits of YOLOv1

The grid gives YOLOv1 its speed: 45 images per second, 63.4 mAP on VOC 2007.
It also gives it three structural limits.

| Limit | Cause |
|---|---|
| at most $B = 2$ objects per cell | one cell, $B$ predictors |
| at most one class per cell | one $\Pr(c \mid \text{obj})$ per cell |
| small and grouped objects (birds, crowds) are missed | a $64 \times 64$ patch and a single $7 \times 7$ output scale |
| imprecise boxes for unusual shapes | $w, h$ are regressed from nothing, with no shape prior |

Each later version addresses one of these rows. The changes are **anchors**
(v2 and v3), **several output scales** (v3), and then an **anchor-free head**
(v8).

---

## Anchors (v2)

![Anchor boxes assigned to ground-truth boxes by IoU](assets/cv/d2l-anchor-label.png)
*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

**Anchors.** Each cell has $A$ fixed prior boxes of size $(p_w, p_h)$, found
by k-means on the training boxes. The network predicts offsets from them,
each with its own class vector:

$$
b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y, \quad
b_w = p_w\, e^{t_w}, \quad b_h = p_h\, e^{t_h}, \quad
\Pr(\text{obj}) = \sigma(t_o)
$$

Here $(c_x, c_y)$ is the cell index and the result is in grid units. The
sigmoid keeps the centre inside the cell, and the exponential keeps the size
positive, close to the anchor when $t = 0$. **Learnt:** $t$. **Fixed:** anchors
$(p_w, p_h)$.

---

## Multi-scale heads (v3)

v3 predicts at three strides, from three depths of the backbone. For a $640 \times 640$ input:

| Stride | Grid | Locations | Detects |
|---|---|---|---|
| 8 | $80 \times 80$ | 6400 | small objects |
| 16 | $40 \times 40$ | 1600 | medium |
| 32 | $20 \times 20$ | 400 | large |

That is 8400 locations. With 3 anchors per location it gives 25,200 boxes, each
with $4 + 1 + C$ outputs, against 98 for v1.

---

## Anchor-free, decoupled head (v8 to v11)

<!-- placeholder: image to add (YOLOv8 architecture: backbone, PAN-FPN neck, decoupled detection head with separate box/DFL and class branches at strides 8/16/32; e.g. the Ultralytics YOLOv8 architecture diagram by RangeKing, GitHub issue ultralytics/ultralytics#189) -->

Modern YOLO removes anchors and objectness. Each of the 8400 locations, with
centre $(a_x, a_y)$ and stride $s$, predicts **distances to the four edges**:

$$
\hat{b} = \big(a_x - l,\; a_y - t,\; a_x + r,\; a_y + b\big) \cdot s,
\qquad l, t, r, b \ge 0
$$

and $C$ independent sigmoids $\sigma(z_c)$, without softmax and without an
objectness term. The detection score is simply $\sigma(z_c)$. Box and class
come from two separate conv branches (a **decoupled** head).

| | Box branch | Class branch |
|---|---|---|
| output per location | $4 \times 16$ (a distribution per edge) | $C$ |
| output for 640, COCO | $64 \times 8400$ | $80 \times 8400$ |
| decoded | $4 \times 8400$ | $80 \times 8400$ |

$$
\mathcal{L} = \lambda_{box}\, \mathcal{L}_{CIoU} + \lambda_{cls}\, \mathcal{L}_{BCE}
+ \lambda_{dfl}\, \mathcal{L}_{DFL}
$$

- **BCE**: per-class binary cross-entropy on the sigmoids.
- **CIoU**: $1 - \mathrm{IoU}$, plus a penalty on the distance between centres
  and on the difference in aspect ratio.
- **DFL** (distribution focal loss): each edge distance is predicted as a
  softmax over 16 bins, $l = \sum_k k\, p_k$, trained to put its mass on the
  two bins around the target.

Targets are assigned from the current predictions: each ground truth takes
the 10 locations with the best $s^{\alpha}\,\mathrm{IoU}^{\beta}$.

---

## In practice: ultralytics

One `.txt` file per image, one line per object: the class index, then
`cxcywh` normalised by the image size.

```text
# labels/img_001.txt      class  cx     cy     w      h
2  0.421875  0.281250  0.234375  0.312500
```

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")                    # COCO-pretrained
model.train(data="data.yaml", epochs=50, imgsz=640)
r = model("street.jpg", conf=0.25, iou=0.7)[0]
r.boxes.xyxy, r.boxes.cls, r.boxes.conf      # (N, 4), (N,), (N,)
```

`conf` is the score threshold and `iou` the NMS threshold: these are the two
fixed hyperparameters of the decoding step. `data.yaml` gives the image
folders and the class names. Always draw about twenty labelled images before
training. A `cxcywh` / `xyxy` mix-up, or a missing normalisation, produces
boxes that look plausible on small objects and are completely wrong on large
ones.

---

## Exercise: YOLOv1 by hand

YOLOv1, image $448 \times 448$, $S = 7$, $B = 2$, $C = 20$.

**A. Encoding.** A dog (class 11) has a ground-truth box
$(x_1, y_1, x_2, y_2) = (50, 100, 274, 380)$ in pixels.

1. Size of the output tensor? Number of numbers? Number of candidate boxes?
2. Centre and size of the box in pixels? Which cell $(\text{row}, \text{col})$ is responsible?
3. Regression target $(x^*, y^*, w^*, h^*)$, and the values the loss actually
   compares for the size?
4. The two predictors of that cell currently have IoU 0.31 and 0.58 with the
   dog. Which is responsible? What is its confidence target? What are the
   confidence targets of the other 97 predictors?

**B. Decoding.** Cell $(\text{row}, \text{col}) = (1, 5)$ outputs, for its first box,
$(x, y, w, h) = (0.5, 0.25, 0.25, 0.25)$ with $\hat{C} = 0.8$, and
$\Pr(\text{car} \mid \text{obj}) = 0.9$.

5. Class-specific score for "car"? Box in `xyxy` pixels?

<!-- notes: 15 minutes. A2 is where they get stuck: remind them of the floor
and that the row comes from y. A4 is the conceptual one: the target for
confidence is an IoU, not 1. -->

---

## Solution

**A1.** $7 \times 7 \times 30$, i.e. $1470$ numbers, $7 \cdot 7 \cdot 2 = 98$
boxes. A cell covers $448 / 7 = 64$ pixels.

**A2.** $c_x = 162$, $c_y = 240$, $w = 224$, $h = 280$.
$\text{col} = \lfloor 162 / 64 \rfloor = \lfloor 2.53 \rfloor = 2$,
$\text{row} = \lfloor 240 / 64 \rfloor = \lfloor 3.75 \rfloor = 3$: cell $(3, 2)$.

**A3.**

| | Value |
|---|---|
| $x^* = 2.53125 - 2$ | $0.531$ |
| $y^* = 3.75 - 3$ | $0.75$ |
| $w^* = 224 / 448$ | $0.5$, loss on $\sqrt{w^*} = 0.707$ |
| $h^* = 280 / 448$ | $0.625$, loss on $\sqrt{h^*} = 0.791$ |

The class target of cell $(3, 2)$ is the one-hot vector at index 11.

**A4.** The second predictor (IoU 0.58 > 0.31): $\mathbb{1}^{obj}_{ij} = 1$
for it, and its confidence target is $C^* = 0.58$, the current IoU. The other 97
predictors, including the first one of the same cell, have target 0 with
weight $\lambda_{noobj} = 0.5$ and receive no coordinate loss.

**B5.** $s_{car} = 0.9 \times 0.8 = 0.72$.
$c_x = (5 + 0.5) \cdot 64 = 352$, $c_y = (1 + 0.25) \cdot 64 = 80$,
$w = h = 0.25 \cdot 448 = 112$.
Box: $(296, 24, 408, 136)$. It is kept if $0.72 > \tau$, then passes through
NMS with the other car boxes.
