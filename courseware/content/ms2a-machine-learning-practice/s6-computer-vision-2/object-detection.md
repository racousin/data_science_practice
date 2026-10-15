# Object Detection

Classification answers *what is in this image*. Detection answers *what, and
where, and how many* — and that change of output type changes the loss, the
metric, the annotation cost and the architecture.

<!-- notes: 40 minutes. Longest lesson of the session. Spend the time on IoU,
NMS and mAP — students who skip those cannot read a detection paper or debug
their own model. The architecture lineage is fast: it is context, not exam
material. Live demo at the end with a pretrained YOLO on a webcam frame. -->

---

## The task

![Detections with boxes and labels](assets/cv/object-detection-examples.jpeg)

One image in, a **variable-length set** out: an empty road yields zero
detections, a street scene forty. That is the whole difficulty. A classifier
has a fixed-size output and a one-to-one target; a detector has neither, so it
needs a matching step before it can even compute a loss.

---

## Problem Formulation

Object detection maps an input image to a variable-length set of detections:

$$
f : \mathbb{R}^{H \times W \times C} \rightarrow \{(b_i, c_i, p_i)\}_{i=1}^{N}
$$

Each detection consists of:

- $b_i \in \mathbb{R}^4$: Bounding box coordinates $(x, y, \text{width}, \text{height})$ or $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$
- $c_i \in \{1, \dots, K\}$: Class label from $K$ object categories
- $p_i \in [0, 1]$: Confidence score

The number of detections $N$ varies per image and is determined by the model.

---

## The output format

Every detector, whatever its internals, emits three parallel arrays.

```python
out = model([image_tensor])[0]
out["boxes"]    # (N, 4) float, [x_min, y_min, x_max, y_max] in pixels
out["labels"]   # (N,)  int64, class index
out["scores"]   # (N,)  float in [0, 1], descending
```

Two box conventions exist and they are silently incompatible: corner form
`(x1, y1, x2, y2)` and centre form `(cx, cy, w, h)`, the latter often
normalized to `[0, 1]`.

---

## IoU

![Intersection over union](assets/cv/iou-visualization.png)

The only sensible way to say "these two boxes are the same box":

$$
IoU(B_p, B_g) = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}
$$

It is scale-invariant, bounded in `[0, 1]`, and 0 for any two disjoint boxes.
A prediction counts as a true positive when its IoU with an unmatched
ground-truth box of the same class exceeds a threshold — 0.5 by convention,
0.75 when you care about localization.

---

## Precision and recall, per class


![1_GbILYTKlwEKAUEhupVKQTg.png](assets/cv/1_GbILYTKlwEKAUEhupVKQTg.png)


Sort all predictions of one class by score, walk down the list, and mark each
as a true or false positive by the IoU rule. Each prefix of that sorted list
gives a `(recall, precision)` point, and together they trace the precision–
recall curve for the class. Average precision is the area under it:

$$
AP = \sum_n (R_n - R_{n-1}) \cdot P_n
$$

---

## mAP

Mean average precision is `AP` averaged over classes. The IoU threshold is part
of the metric's name, and reporting it is not optional.

| Name | Meaning |
|---|---|
| mAP@0.5 | one threshold, IoU ≥ 0.5. The Pascal VOC convention |
| mAP@0.75 | strict localization |
| mAP@[0.5:0.95] | averaged over ten thresholds. The COCO number |

A model at 55 mAP@0.5 and 32 mAP@[0.5:0.95] is finding the objects and boxing
them loosely — the gap is a localization diagnostic, not a rounding difference.

---

## Non-maximum suppression (NMS)

Detectors fire several overlapping boxes on the same object; NMS keeps one.
Per class: sort by score, keep the top box, remove the others with $\mathrm{IoU} > \tau$ (typically 0.5), repeat.
Threshold trade-off: too low merges neighbours in crowds, too high leaves duplicates.


![non-max-suppression.webp](assets/cv/non-max-suppression.webp)

---

## Two-stage detectors

![Faster R-CNN](assets/cv/faster-rcnn-architecture.jpg)


| Model | Idea | Cost per image |
|---|---|---|
| R-CNN (2014) | selective search → CNN on each of 2000 crops | ~50 s |
| Fast R-CNN | one CNN pass, RoI pooling on the feature map | ~2 s |
| Faster R-CNN | proposals from a learned network (RPN) | ~0.2 s |

Stage 1: Region Proposal Network (RPN) generates candidate boxes
Stage 2: Each proposal is classified and its box refined

---

---

## Limits of the R-CNN family

Faster R-CNN is accurate, but its design carries costs that motivated
everything that followed.

| Issue | Where it comes from | Consequence |
|---|---|---|
| **Speed** | Two sequential stages: RPN, then a head run on every proposal | ~5 fps; head cost scales with the number of RoIs; not real-time |
| **Anchors** | Hand-chosen scales and aspect ratios ($k = 9$ per location) | Prior on object shape: long, thin objects (cracks, cables, text lines) get poor recall; more hyperparameters to tune |
| **Heuristic assignment** | IoU thresholds (0.7 / 0.3 in the RPN, 0.5 in the head) decide positives | Brittle, dataset-dependent choices; not learned |
| **Class imbalance** | Most of the ~20k anchors are background | Requires sampled mini-batches (e.g., 1:3 positive:negative) |


<!-- notes: 4 minutes. Practical takeaway: before accepting default anchors,
look at the aspect-ratio histogram of your training boxes. -->

---

## YOLO: one pass, one tensor


![yolo_v1_diagram.png](assets/cv/yolo_v1_diagram.png)


YOLOv1 (Redmon et al., 2016) handles detection as a single regression. The
image is divided into an $S \times S$ grid. Each cell predicts $B$ boxes, a
confidence for each box, and one class distribution. The output is a fixed
tensor:

$$
f_\theta : \mathbb{R}^{3 \times 448 \times 448} \longrightarrow
\mathbb{R}^{S \times S \times (5B + C)}
$$


$S = 7$, $B = 2$, $C = 20$, so the output is
$7 \times 7 \times 30$: 1470 numbers that encode $7 \cdot 7 \cdot 2 = 98$
candidate boxes.


---

## YOLOv1: the network



![50821f93a75d0121345b4116c90b848f67348df5-1236x638.png](assets/cv/50821f93a75d0121345b4116c90b848f67348df5-1236x638.png)


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
3. **Responsible box.** Among the $B$ predicted boxes of the cell, the one with
   the highest IoU with the ground truth.

This defines the indicators used in the loss:

- $\mathbb{1}_{ij}^{obj} = 1$ if box $j$ of cell $i$ is responsible
  for an object, and 0 otherwise;
- $\mathbb{1}_{ij}^{noobj} = 1 - \mathbb{1}_{ij}^{obj}$;
- $\mathbb{1}_{i}^{obj} = 1$ if an object's centre falls in cell $i$.

where $i \in \{0, \dots, S^2-1\}$ indexes the cells ($i = \text{row} \cdot S + \text{col}$)
and $j \in \{1, \dots, B\}$ indexes the boxes within a cell.

![images4.jpeg](assets/cv/images4.jpeg)


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

**4. NMS**, per class


![final_result.jpg](assets/cv/final_result.jpg)


---

## Limits of YOLOv1 - Questions

1. A photo shows a flock of 30 birds in the sky, or a crowd seen from far
   away. How many objects can YOLOv1 detect at most, and how many of them
   can lie in the same $64 \times 64$ region? What happens if two objects
   of different classes have their centres in the same cell?


<!-- notes: 10 minutes in pairs. Q1 is the grid constraint, Q2 is
localization. Both answers point to what YOLOv2/v3 change. -->

---

## Limits of YOLOv1 - Answers

**Q1. The grid caps the output.**
- At most $S^2 \cdot B = 98$ boxes, and **at most 2 objects per cell**:
  a cell is responsible only for the objects whose centre falls in it.
- **One class distribution per cell**: two objects of different classes
  in the same cell cannot both be detected correctly.
- Small objects in groups (birds, people in a crowd, cars in a parking lot)
  share cells, so most of them are lost.

---

## DETR — Detection as Set Prediction

A detector outputs a set of boxes. DETR (Carion et al., 2020) predicts that
set directly: a fixed number of slots, each a class and a box, trained with a
loss that first matches slots to ground-truth objects one-to-one. Anchors and
NMS disappear, and the model is the encoder–decoder transformer of Session 6.

<!-- notes: 35 minutes. Students know IoU, NMS and mAP from Object Detection,
the grid and anchors from YOLO, and ViT from earlier in the session. The whole lesson is "the
Session 6 encoder-decoder, with three changes" plus the Hungarian matching.
Spend the time on the matching slide and the exercise: the matching is the
single idea that makes everything else possible. Do the dimensional-flow table
on the board with 800x1066 before showing it. -->


---

## The idea: a set, not a grid

A YOLO-style detector makes **dense** predictions — one or several boxes per
grid cell and anchor — and the same object is typically predicted by several
neighbouring cells. Nothing in the loss forbids it, so a post-processing step,
NMS, removes the duplicates.

DETR outputs exactly $N$ predictions $\hat{y} = \{\hat y_j\}_{j=1}^N$, with
$N$ larger than the number of objects in any image, and matches them
**one-to-one** with the ground truth $y$, padded with "no object" $\varnothing$
to size $N$:

$$
\hat y_j = (\hat p_j, \hat b_j), \qquad
\hat p_j \in \Delta^{K}\; (K \text{ classes} + \varnothing), \qquad
\hat b_j \in [0,1]^4
$$

Each ground-truth object is matched to **one** prediction; every other
prediction is trained to say $\varnothing$. A duplicate is therefore not
harmless: it is an unmatched slot predicting an object, and it is penalised.
The model learns to not produce duplicates, so NMS is not needed. The grid and
the anchors go too: a slot is not attached to a location.

---

## Architecture


![2-8.png](assets/cv/2-8.png)
