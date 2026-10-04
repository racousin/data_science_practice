# DETR — Detection as Set Prediction

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

![DETR: CNN backbone, transformer encoder, transformer decoder with object queries, shared FFN prediction heads](assets/cv/detr-architecture.png)

The encoder–decoder of Session 6, with three changes:

| | Session 6 (translation) | DETR |
|---|---|---|
| Encoder input | word embeddings $E[\mathrm{src}] + PE$, $n \times d$ | CNN feature pixels, $HW \times d$, 2D positions |
| Decoder input | target tokens, shifted, causal mask | $N$ learned **object queries**, no mask, one pass |
| Output head | $Y_N E^T$, softmax over the vocabulary | per slot: $K{+}1$ class logits and a box, set loss |

Everything in between — multi-head attention, FFN, residuals, layer norm,
cross-attention with $Q$ from the decoder and $K, V$ from the encoder — is
unchanged.

<!-- placeholder: image to add (detailed DETR transformer: encoder and decoder blocks with positional encodings added to Q and K at every layer, object queries added in the decoder — Carion et al. 2020, "End-to-End Object Detection with Transformers", Fig. 10) -->

---

## Backbone: image to tokens

**Input.** An image $x \in \mathbb{R}^{3 \times H_0 \times W_0}$.

**Output.** A ResNet-50 without its pooling and classifier, stride 32:

$$
f = \mathrm{CNN}(x) \in \mathbb{R}^{2048 \times H \times W}, \qquad
H = \lceil H_0 / 32 \rceil,\; W = \lceil W_0 / 32 \rceil
$$

A $1\times 1$ convolution reduces the channels to $d = 256$, and the spatial
grid is flattened into a sequence — each feature pixel becomes a token, as a
patch does in ViT:

$$
z = W_{proj} * f \in \mathbb{R}^{d \times H \times W}, \qquad
X_0 = \mathrm{flatten}(z) \in \mathbb{R}^{HW \times d}
$$

**Learnt:** the ResNet weights (ImageNet-pretrained, fine-tuned with a
learning rate 10× smaller) and $W_{proj} \in \mathbb{R}^{d \times 2048}$ plus
bias. **Fixed:** $d = 256$, stride 32.

**Positions.** Flattening loses the 2D layout, and attention is
permutation-equivariant. Each token at row $u$, column $v$ gets a fixed
encoding: the Session 6 sinusoid with $d/2 = 128$ dimensions applied to each
coordinate (normalised to $[0, 2\pi]$), concatenated:

$$
P_{(u,v)} = \big[\,PE_{128}(u)\ ;\; PE_{128}(v)\,\big] \in \mathbb{R}^{d},
\qquad P \in \mathbb{R}^{HW \times d}
$$

Unlike Session 6, $P$ is not added once at the input: it is added to the
queries and keys of **every** attention layer, never to the values.

---

## Encoder

$X_b \in \mathbb{R}^{HW \times d}$ enters encoder block $b$, $X_0$ from the
backbone. The block is the Session 6 block; only the positions enter
differently:

$$
H_b = X_b + \mathrm{MHA}\big(Q = X_b + P,\; K = X_b + P,\; V = X_b\big)
$$

$$
X_{b+1} = H_b + \mathrm{FFN}(H_b)
$$

(layer norms omitted; DETR uses the post-norm placement $LN(x + f(x))$.)

**Input / output.** $HW \times d \to HW \times d$, six blocks:
$\mathrm{enc} = X_6 \in \mathbb{R}^{HW \times d}$, the **memory**.

**Learnt, per block:** $W^Q, W^K, W^V, W^O$ ($8$ heads of $d/h = 32$), the FFN
$d \to 2048 \to d$, two layer norms. **Fixed:** 6 blocks, 8 heads,
$d_{ff} = 2048$, dropout 0.1.

**Cost.** Each head builds an $HW \times HW$ attention matrix: quadratic in the
number of pixels, as it was quadratic in sentence length in Session 6. At
stride 32 the sequence is short enough; at stride 8 it would be 16 times
longer and the attention matrix 256 times larger. Every feature pixel can
attend to every other: the encoder reasons about the whole image, which is
what separates adjacent instances.

<!-- placeholder: image to add (encoder self-attention maps: for a few reference points on different cows, the attention of that pixel over the image, each covering its own instance — Carion et al. 2020, Fig. 3) -->

---

## Decoder: object queries

**Object queries.** $Q_{obj} \in \mathbb{R}^{N \times d}$, $N = 100$, a
learned embedding table — one row per slot. They play the role of positional
encodings for the decoder: the decoder state starts at $Y_0 = 0 \in
\mathbb{R}^{N \times d}$ and $Q_{obj}$ is added to the queries (and keys, in
self-attention) at every layer.

Decoder block $b$ takes $Y_b \in \mathbb{R}^{N \times d}$ through the three
Session 6 steps:

1. Slots read each other:

$$
S_b = Y_b + \mathrm{MHA}\big(Q = Y_b + Q_{obj},\; K = Y_b + Q_{obj},\; V = Y_b\big)
$$

2. Slots read the image:

$$
C_b = S_b + \mathrm{MHA}\big(Q = S_b + Q_{obj},\; K = \mathrm{enc} + P,\; V = \mathrm{enc}\big)
$$

3. Each slot on its own:

$$
Y_{b+1} = C_b + \mathrm{FFN}(C_b)
$$

| Attention | $Q$ from | $K, V$ from | Matrix | Mask |
|---|---|---|---|---|
| encoder self | pixels | pixels | $HW \times HW$ | padding |
| decoder self | slots | slots | $N \times N$ | none |
| cross | slots | encoder memory | $N \times HW$ | padding |

---

## Decoder: why a set works

**No causal mask, not autoregressive.** The $N$ outputs form a set with no
order, so all slots are decoded in one parallel pass. **Self-attention between
slots** is what lets them coordinate: a slot can see that another is already
describing an object and move away from it — the mechanism that replaces NMS.

**Learnt:** $Q_{obj}$ ($N \cdot d$ parameters), and per block two MHA, one
FFN, three layer norms. **Fixed:** $N = 100$, 6 blocks.

<!-- placeholder: image to add (decoder cross-attention for each predicted object, concentrated on the object extremities — heads, legs, edges — Carion et al. 2020, Fig. 6) -->

---

## Prediction heads

The same heads are applied to each of the $N$ output rows
$Y_6 \in \mathbb{R}^{N \times d}$ independently:

$$
\hat{\ell} = Y_6 W_{cls} + b_{cls} \in \mathbb{R}^{N \times (K+1)}, \qquad
\hat p_j = \mathrm{softmax}(\hat{\ell}_j)
$$

$$
\hat{b} = \sigma\big(\mathrm{MLP}(Y_6)\big) \in [0, 1]^{N \times 4},
\qquad \hat b_j = (c_x, c_y, w, h)
$$

The extra class $\varnothing$ ("no object") is how a slot says it is empty.
Boxes are centre, width and height **normalised** by the image size: the
sigmoid keeps them in $[0,1]$, and the loss does not depend on image
resolution.

**Learnt:** $W_{cls} \in \mathbb{R}^{d \times (K+1)}$, and a 3-layer MLP
$d \to d \to d \to 4$ with ReLU. **Fixed:** $K$, set by the dataset.

---

## Dimensional flow

COCO image resized to $800 \times 1066$, $d = 256$, $N = 100$, $K = 91$
(COCO category ids, some unused).

| # | Step | Tensor | Shape |
|---|---|---|---|
| 1 | image | $x$ | $3 \times 800 \times 1066$ |
| 2 | ResNet-50, stride 32 | $f$ | $2048 \times 25 \times 34$ |
| 3 | $1\times1$ conv | $z$ | $256 \times 25 \times 34$ |
| 4 | flatten | $X_0$, $P$ | $850 \times 256$ |
| 5 | 6 encoder blocks | $\mathrm{enc}$ | $850 \times 256$ |
| | — attention per head | | $850 \times 850 = 722{,}500$ |
| 6 | object queries | $Q_{obj}$, $Y_0 = 0$ | $100 \times 256$ |
| 7 | 6 decoder blocks | $Y_6$ | $100 \times 256$ |
| | — self / cross attention per head | | $100 \times 100$ / $100 \times 850$ |
| 8 | class head | $\hat{\ell}$ | $100 \times 92$ |
| 9 | box head | $\hat{b}$ | $100 \times 4$ |

---

## Parameters

| Parameters | Count |
|---|---|
| ResNet-50 backbone (no classifier) | 23.5M |
| Encoder, 6 blocks × 1.32M | 7.9M |
| Decoder, 6 blocks × 1.58M | 9.5M |
| $1\times1$ projection, $Q_{obj}$, class and box heads | 0.7M |
| **Total** | **≈ 41.5M** (the paper reports 41M) |

A decoder block has one more MHA ($4d^2 + 4d = 263$k) than an encoder block;
the FFN ($2 \cdot d \cdot 2048$ + biases $= 1.05$M) dominates both.

---

## Hungarian matching

Before any loss, find the assignment of slots to ground truth. Pad the $M$
ground-truth objects with $\varnothing$ to $N$ entries, and search over the
permutations $\mathcal{S}_N$:

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathcal{S}_N}
\sum_{i=1}^{N} \mathcal{L}_{match}\big(y_i, \hat y_{\sigma(i)}\big)
$$

$$
\mathcal{L}_{match}(y_i, \hat y_j) =
-\mathbb{1}_{c_i \neq \varnothing}\, \hat p_j(c_i)
+ \mathbb{1}_{c_i \neq \varnothing}\, \mathcal{L}_{box}(b_i, \hat b_j)
$$

Padding entries cost 0 whatever they are matched to, so only an
$M \times N$ cost matrix matters. A good match has high probability on the
right class and a close box. The Hungarian algorithm solves the assignment
exactly in $O(N^3)$ — on the CPU, without gradients: $\hat{\sigma}$ is a fixed
target for this step.

```python
from scipy.optimize import linear_sum_assignment
C = -P[:, gt_cls].T + 5 * L1 + 2 * (-GIoU)   # (M, N): one row per GT object
gt_idx, query_idx = linear_sum_assignment(C) # M pairs; the N-M other slots -> ∅
```

The optimum is global: a greedy choice, object by object, is not.

<!-- placeholder: image to add (bipartite matching between N predicted boxes and M ground-truth boxes on one image, matched pairs linked, unmatched predictions labelled "no object" — e.g. adapted from Carion et al. 2020, Fig. 1/2 style) -->

---

## Loss

With $\hat{\sigma}$ fixed, the **Hungarian loss** sums over all $N$ slots:

$$
\mathcal{L}_{Hung} = \sum_{i=1}^{N} \Big[
-w_{c_i} \log \hat p_{\hat{\sigma}(i)}(c_i)
+ \mathbb{1}_{c_i \neq \varnothing}\, \mathcal{L}_{box}\big(b_i, \hat b_{\hat{\sigma}(i)}\big)
\Big]
$$

$$
\mathcal{L}_{box}(b, \hat{b}) = \lambda_{L1} \| b - \hat{b} \|_1
+ \lambda_{iou}\, \mathcal{L}_{GIoU}(b, \hat{b}),
\qquad \lambda_{L1} = 5,\; \lambda_{iou} = 2
$$

Classification covers every slot; the $\varnothing$ term is down-weighted,
$w_\varnothing = 0.1$ ($w_c = 1$ otherwise), because about 95 of the 100 slots
are empty. Boxes are penalised only for matched slots.

The L1 term alone grows with box size. **Generalised IoU** is scale-invariant
and, unlike IoU, still has a gradient when the boxes do not overlap. With $C$
the smallest box enclosing $b$ and $\hat{b}$:

$$
\mathrm{GIoU} = \mathrm{IoU}(b, \hat{b}) - \frac{|C \setminus (b \cup \hat{b})|}{|C|}
\in (-1, 1], \qquad
\mathcal{L}_{GIoU} = 1 - \mathrm{GIoU}
$$

**Auxiliary losses.** The shared heads are also applied after each of the 6
decoder blocks, each output matched and penalised with the same loss; this
helps each block produce the right number of objects.

---

## Inference

One forward pass, then a threshold. No NMS, no anchors to decode:

$$
\text{keep } j \Leftrightarrow \arg\max_c \hat p_j(c) \neq \varnothing
\; \text{ and } \; \max_{c \neq \varnothing} \hat p_j(c) > \tau
$$

Kept boxes are converted from normalised $(c_x, c_y, w, h)$ to pixel corners,
$x_1 = (c_x - w/2)\,W_0$, and so on.

```python
from transformers import DetrImageProcessor, DetrForObjectDetection  # needs timm
proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
# logits (1,100,92), pred_boxes (1,100,4)
out = model(**proc(images=img, return_tensors="pt"))
det = proc.post_process_object_detection(
    out, threshold=0.9, target_sizes=[img.size[::-1]])[0]
```

`det` holds `scores`, `labels` and `boxes` in pixels — the three arrays of the
YOLO lesson.

---

## DETR against YOLO

| | YOLO (dense) | DETR (set) |
|---|---|---|
| Output | many boxes per cell and anchor | $N$ slots, one per object at most |
| Duplicates | removed by NMS | prevented by one-to-one matching |
| Hand-designed parts | grid, anchors, NMS threshold | $N$, cost weights |
| Convergence | fast: dense supervision per cell | slow: 500 epochs on COCO |
| Small objects | multi-scale features | weaker: one stride-32 map, $(HW)^2$ forbids finer |
| COCO, ResNet-50 | — | 42.0 AP, 41M parameters, 28 FPS (V100) |

DETR matches Faster R-CNN in AP (42.0), with better large-object AP (61.1 vs
53.4) and worse small-object AP (20.5 vs 26.6). Both weaknesses come from
dense global attention. **Deformable DETR** lets each query attend to a few
learned sampling points on multi-scale feature maps, which makes finer maps
affordable and cuts training about tenfold. **DINO** improves the matching and
query initialisation; **RT-DETR** reaches real-time speed and competes with
YOLO — without NMS.

---

## Exercise

A DETR with ResNet-50 (stride 32), $d = 256$, 8 heads, $N = 100$, trained on
$K = 3$ classes (cat, dog, bird).

**A. Shapes.** Input image $640 \times 640$.

1. Shape of $f$, number of encoder tokens, shape of $X_0$.
2. Size of one encoder attention matrix, one decoder self-attention matrix,
   one cross-attention matrix.
3. Shapes of the class logits and of the boxes.
4. The image becomes $1280 \times 1280$. By what factor does the encoder
   attention matrix grow? The decoder self-attention?

**B. Matching.** Two ground-truth objects (a cat, a dog) and three slots.
Use $\mathcal{L}_{match} = -\hat p_j(c_i) + 5\,\| b_i - \hat b_j \|_1$
(GIoU omitted).

| | $\hat{p}(\text{cat})$ | $\hat{p}(\text{dog})$ | $\| b_{cat} - \hat{b} \|_1$ | $\| b_{dog} - \hat{b} \|_1$ |
|---|---|---|---|---|
| slot 1 | 0.7 | 0.2 | 0.10 | 0.40 |
| slot 2 | 0.6 | 0.3 | 0.04 | 0.04 |
| slot 3 | 0.1 | 0.8 | 0.50 | 0.20 |

1. Write the $2 \times 3$ cost matrix.
2. Assign greedily: the cat takes its cheapest slot, then the dog. Total cost?
3. Find the optimal assignment. Total cost?
4. Which slot is matched to $\varnothing$? Which loss terms does it receive?

<!-- notes: 15 minutes. A4 is the argument for Deformable DETR. In B, slot 2
is the best candidate for both objects: the point is that the greedy choice
is not the optimum. -->

---

## Solution

**A1.** $640 / 32 = 20$: $f \in \mathbb{R}^{2048 \times 20 \times 20}$,
$400$ tokens, $X_0 \in \mathbb{R}^{400 \times 256}$.

**A2.** Encoder $400 \times 400 = 160{,}000$; decoder self $100 \times 100$;
cross $100 \times 400$. Per head, 8 heads per layer.

**A3.** Logits $100 \times 4$ ($K + 1$ with $\varnothing$), boxes $100 \times 4$.

**A4.** $40 \times 40 = 1600$ tokens, $4\times$ more: the encoder attention
grows $16\times$ ($1600^2 = 2.56$M). The decoder self-attention does not
change: it depends on $N$, not on the image.

**B1.** $C_{ij} = -\hat p_j(c_i) + 5\,\mathrm{L1}_{ij}$:

| | slot 1 | slot 2 | slot 3 |
|---|---|---|---|
| cat | $-0.7 + 0.5 = -0.2$ | $-0.6 + 0.2 = -0.4$ | $-0.1 + 2.5 = 2.4$ |
| dog | $-0.2 + 2.0 = 1.8$ | $-0.3 + 0.2 = -0.1$ | $-0.8 + 1.0 = 0.2$ |

**B2.** Cat → slot 2 ($-0.4$), dog → slot 3 ($0.2$): total $-0.2$.

**B3.** Cat → slot 1, dog → slot 2: $-0.2 - 0.1 = -0.3$, the minimum of the
six assignments (`linear_sum_assignment` returns the same).

**B4.** Slot 3, although it is confident about "dog". It receives only the
classification term $-0.1 \log \hat p_3(\varnothing)$: it is pushed to
predict "no object", and gets no box loss.
