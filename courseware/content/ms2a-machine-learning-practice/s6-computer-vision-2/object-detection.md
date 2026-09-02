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
normalized to `[0, 1]`. Half the detection bugs in student code are a
conversion that was never done. Assert the convention at the boundary:

```python
assert (boxes[:, 2] > boxes[:, 0]).all(), "not corner-form xyxy"
```

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

## Computing IoU

```python
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda z: (z[2] - z[0]) * (z[3] - z[1])
    return inter / (area(a) + area(b) - inter)
```

The `max(0, ...)` clamps are the whole function. Without them a disjoint pair
produces a *positive* intersection from two negative widths and your metric
quietly reports excellent performance. Test it on identical boxes (1.0) and
disjoint boxes (0.0); that two-line test catches the bug in seconds.

---

## Non-maximum suppression

A detector fires many times on the same object — neighbouring anchors, adjacent
grid cells. NMS keeps one.

```python
from torchvision.ops import nms
keep = nms(boxes, scores, iou_threshold=0.5)
boxes, scores = boxes[keep], scores[keep]
```

Sort by score, take the top box, drop every remaining box whose IoU with it
exceeds the threshold, repeat. Run it **per class** — otherwise a dog box
suppresses the overlapping person box behind it.

A low threshold merges distinct adjacent objects; a high one leaves
duplicates. 0.5–0.7 is the usual range, and it is a knob worth sweeping.

---

## Precision and recall, per class

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

> Never compare mAP figures across datasets or thresholds. It is the most
> common misreading of a detection benchmark.

---

## Two-stage detectors

![Faster R-CNN](assets/cv/faster-rcnn-architecture.jpg)

The lineage, each step removing the previous bottleneck:

| Model | Idea | Cost per image |
|---|---|---|
| R-CNN (2014) | selective search → CNN on each of 2000 crops | ~50 s |
| Fast R-CNN | one CNN pass, RoI pooling on the feature map | ~2 s |
| Faster R-CNN | proposals from a learned network (RPN) | ~0.2 s |

Fast R-CNN's insight is that the 2000 crops overlap enormously, so compute the
features once and *index* into them. Faster R-CNN's is that selective search,
the last non-learned component, can be a small convolutional head.

---

## The RPN and anchors

At each position of the feature map, place `k` reference boxes of fixed scales
and aspect ratios: the **anchors**. The RPN predicts, per anchor, an
objectness score and four offsets that deform it toward a real object.

```python
# 3 scales x 3 aspect ratios = 9 anchors per location
objectness = conv_obj(features)   # (B, 9, H, W)
deltas     = conv_box(features)   # (B, 36, H, W)
```

Anchors are a prior on object shape. If your objects are long and thin —
cracks, cables, text lines — the default square-ish anchor set will not match
them and recall collapses. Look at the aspect-ratio histogram of your training
boxes before you accept a default configuration.

---

## One-stage detectors

![The YOLO grid](assets/cv/yolo-grid.png)

YOLO drops the proposal stage. The image is divided into an `S × S` grid; each
cell predicts a fixed number of boxes with an objectness score and a class
distribution, in one forward pass.

```python
# (B, anchors, H, W, 5 + K)  ->  5 = x, y, w, h, objectness
pred = detection_head(backbone(image))
```

SSD adds the other half of the recipe: predict from **several feature maps at
different depths**, so shallow high-resolution maps catch small objects and
deep coarse ones catch large objects. Every modern detector does some of this.

---

## The one-stage problem, and focal loss

A one-stage detector scores ~100k candidate locations per image, of which a
handful contain objects. The background term dominates the loss and the model
converges to predicting nothing.

$$
L_{focal} = -(1 - \hat{p})^\gamma \log \hat{p}
$$

The factor $(1-\hat{p})^\gamma$ is near zero for confidently-classified easy
background and near one for hard examples, so the gradient concentrates where
learning is still possible. With $\gamma = 2$, RetinaNet was the first
one-stage detector to match two-stage accuracy.

Anchor-free detectors — FCOS, CenterNet — regress the four distances from each
pixel to the box edges instead, deleting the anchor hyperparameters entirely.

---

## DETR and set prediction

![DETR](assets/cv/detr-architecture.png)

DETR reformulates detection as predicting a *set* of exactly `N` slots (100 is
typical), most of which are the class "no object".

```python
memory = transformer_encoder(cnn_backbone(image))
pred   = transformer_decoder(object_queries, memory)   # (100, d)
```

Training uses a **Hungarian matching** between the 100 predictions and the
ground-truth boxes, then a loss on the matched pairs only. Because the matching
is one-to-one, duplicates are penalised directly — so **there is no NMS and
there are no anchors**. The whole postprocessing stack disappears.

The price is slow convergence — the original needed 500 epochs. Deformable
DETR, DINO and RT-DETR fixed that; RT-DETR is now a real alternative to YOLO at
comparable latency.

---

## Annotation formats

Two you will meet. COCO JSON, one file for the whole dataset:

```json
{"images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}],
 "annotations": [{"image_id": 1, "category_id": 3, "bbox": [50, 30, 150, 150]}]}
```

COCO `bbox` is `[x_min, y_min, width, height]` in pixels. YOLO uses one `.txt`
per image, one line per object, class index then centre form normalized to
`[0, 1]`:

```text
2 0.421875 0.281250 0.234375 0.312500
```

Neither is more correct. Converting between them wrong — forgetting the
normalization, or reading `w, h` as `x2, y2` — produces boxes that look
plausible on small objects and are catastrophically wrong on large ones. Render
twenty annotated images before training. Always.

---

## Labelling costs money

A box takes 5–15 seconds of human time and you need thousands per class.

- **Start pretrained.** COCO's 80 classes cover more of your problem than you
  expect. Check before you annotate anything.
- **Annotate the hard cases first**, not a random sample — occlusions, small
  objects, unusual lighting.
- **Measure inter-annotator agreement** on 100 images. If two humans disagree
  at IoU 0.7, no model will do better: your metric ceiling is already set.
- Semi-automatic labelling — a weak model proposes, a human corrects — is
  roughly three times faster and is standard practice.

---

## What to actually use

| Situation | Choice |
|---|---|
| Common objects, no training budget | pretrained YOLO or Faster R-CNN, zero-shot |
| Custom classes, a few thousand boxes | fine-tune YOLO (`ultralytics`) |
| Accuracy matters more than latency | Faster R-CNN or Cascade R-CNN with an FPN |
| Real time on an edge device | YOLO-nano/small, exported to ONNX or TensorRT |
| Open vocabulary, describe by text | Grounding DINO, OWL-ViT |

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```

Default to fine-tuning a small pretrained one-stage model; reach for two-stage
or transformer detectors when the mAP gap actually costs you something.
