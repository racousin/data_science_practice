# Segmentation

A box says roughly where. A mask says exactly which pixels. Segmentation is
classification run once per pixel: the output has the spatial size of the
input, and the whole difficulty is producing it at full resolution.

<!-- notes: 50 minutes. Three things to land: (1) the output is a K×H×W
tensor and the loss is cross-entropy per pixel, plus Dice for imbalance;
(2) the encoder-decoder shape, with U-Net as the worked case — do the
dimensional-flow table on the board; (3) instance segmentation is detection
plus a mask head, and Mask2Former is DETR plus masks. The exercise takes
15 minutes; part A is the one that convinces them pixel accuracy is useless. -->

---

## Three tasks

![Semantic, instance and panoptic segmentation of the same image](assets/cv/segmentation-types-comparison.jpg)

For an image $x \in \mathbb{R}^{3 \times H \times W}$ and $K$ classes:

| Task | Output | Two touching people |
|---|---|---|
| **Semantic** | $\hat{y} \in \{1, \dots, K\}^{H \times W}$, one class per pixel | one "person" region |
| **Instance** | a set $\{(m_i, c_i)\}_{i=1}^{n}$, $m_i \in \{0,1\}^{H \times W}$, $c_i \in \{1,\dots,K\}$, countable objects only | two masks |
| **Panoptic** | every pixel gets a class, and pixels of *things* also get an instance id | two masks, plus sky, sea and sand |

Semantic segmentation cannot count: adjacent objects of one class merge into a
single region. Panoptic separates *things* (countable: person, car) from
*stuff* (amorphous: sky, road), and is the union of the two other tasks.

---

## Output tensor and loss

The network returns one logit per class per pixel,
$z \in \mathbb{R}^{K \times H \times W}$, and a softmax over the $K$ axis at
each pixel $u \in \Omega$, $|\Omega| = HW$. The loss is cross-entropy,
averaged over pixels:

$$
\mathcal{L}_{CE} = -\frac{1}{HW} \sum_{u \in \Omega} \log \frac{e^{z_{y_u,u}}}{\sum_{k=1}^{K} e^{z_{k,u}}}
$$

When the foreground is 2% of the pixels, predicting background everywhere
already gives a low $\mathcal{L}_{CE}$. The **soft Dice loss** scores overlap
per class, which does not depend on how large the class is. With
$p_{k,u}$ the softmax probability and $g_{k,u}$ the one-hot target:

$$
\mathcal{L}_{Dice} = 1 - \frac{1}{K} \sum_{k=1}^{K}
\frac{2 \sum_u p_{k,u}\, g_{k,u} + \epsilon}{\sum_u p_{k,u} + \sum_u g_{k,u} + \epsilon}
$$

It uses probabilities, not the argmax, so it is differentiable. Standard
choice: $\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$.

The ground truth is an integer map $y \in \{0,\dots,K-1\}^{H \times W}$, never
an RGB image, and it is resized with **nearest-neighbour** interpolation:
bilinear averages class ids, and a boundary between class 3 and class 7 comes
back containing 4 and 6.

---

## Metrics

![IoU and Dice](assets/cv/dicevsiou.png)

With $TP_k, FP_k, FN_k$ counted over pixels for class $k$:

$$
\mathrm{PA} = \frac{1}{HW} \sum_{u} \mathbb{1}[\hat y_u = y_u], \qquad
\mathrm{IoU}_k = \frac{TP_k}{TP_k + FP_k + FN_k}, \qquad
\mathrm{mIoU} = \frac{1}{K} \sum_{k=1}^{K} \mathrm{IoU}_k
$$

Pixel accuracy (PA) is dominated by the largest class: on data that is 98%
background, a model that predicts background everywhere scores 98%, with
$\mathrm{IoU}_{fg} = 0$. mIoU gives each class equal weight, and is the
standard benchmark metric (counts accumulated over the whole test set).

Dice, the medical-imaging convention, counts the intersection twice:

$$
\mathrm{Dice}_k = \frac{2\,TP_k}{2\,TP_k + FP_k + FN_k} = \frac{2\,\mathrm{IoU}_k}{1 + \mathrm{IoU}_k}
$$

Dice is a monotone function of IoU: the two rank models identically, Dice is
always the larger number (IoU 0.5 is Dice 0.67). State which one you report.

---

## From classifier to dense prediction: FCN

![Fully convolutional network](assets/cv/d2l-fcn.png)
*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

A classification backbone divides the resolution by 32. The Fully
Convolutional Network (Long et al., 2015) drops the pooling and the linear
layer, and upsamples back. Pascal VOC, $K = 21$, input $3 \times 320 \times 480$:

| Component | Input | Output | Learnt |
|---|---|---|---|
| ResNet-18 without GAP + FC | $3 \times 320 \times 480$ | $512 \times 10 \times 15$ | yes (pretrained) |
| $1 \times 1$ conv | $512 \times 10 \times 15$ | $21 \times 10 \times 15$ | yes, $21 \cdot 513$ |
| transposed conv, $k=64, s=32, p=16$ | $21 \times 10 \times 15$ | $21 \times 320 \times 480$ | yes (initialised bilinear) |

Each $10 \times 15$ cell has to produce a $32 \times 32$ block of pixels on its
own: boundaries come out blurred. This is the problem skip connections solve.

---

## Transposed convolution

![Transposed convolution, stride 2](assets/cv/d2l-transposed-conv-stride2.png)
*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Each input value multiplies the whole kernel, the result is written into the
output at a position shifted by the stride, and overlaps are summed. It is
the transpose of the matrix of a convolution: it maps a small grid to a large one.

$$
H_{out} = (H_{in} - 1)\, s - 2p + k
$$

| Layer | $H_{in}$ | $H_{out}$ |
|---|---|---|
| figure above, $k=2, s=2, p=0$ | 2 | $1 \cdot 2 + 2 = 4$ |
| U-Net up-conv, $k=2, s=2, p=0$ | $H$ | $2H$ |
| FCN, $k=64, s=32, p=16$ | 10 | $9 \cdot 32 - 32 + 64 = 320$ |

Learnt: a kernel of shape $C_{in} \times C_{out} \times k \times k$, same count
as a convolution. When $k$ is not a multiple of $s$ the overlaps are uneven
and produce checkerboard artefacts; bilinear upsampling followed by a
$3 \times 3$ conv avoids them.

---

## U-Net

![U-Net architecture](assets/cv/unet-architecture.png)

U-Net (Ronneberger et al., 2015) is a symmetric encoder–decoder. The encoder
applies (two $3 \times 3$ conv + ReLU, then $2 \times 2$ max-pool) four times;
the decoder applies (up-conv $\times 2$, **concatenate** the encoder map of
the same resolution, two $3 \times 3$ conv) four times.

$$
d_\ell = \mathrm{DoubleConv}\big(\,[\,\mathrm{UpConv}(d_{\ell+1})\ ;\; e_\ell\,]\,\big)
$$

The encoder knows *what* is in the image but has lost *where*; the skip
$e_\ell$ gives the decoder back the high-resolution edges. Remove the skips
and boundaries blur. U-Net trains from a few hundred annotated images, which
is why it is the default in medical imaging.

---

## U-Net: dimensional flow

Input $3 \times 256 \times 256$, padded convolutions (the 2015 paper used
unpadded ones and a $572 \times 572$ input), $K$ classes.

| # | Operation | Output shape | Parameters |
|---|---|---|---|
| e1 | DoubleConv $3 \to 64$ | $64 \times 256 \times 256$ | 38.7k |
| e2 | pool, DoubleConv $64 \to 128$ | $128 \times 128 \times 128$ | 221k |
| e3 | pool, DoubleConv $128 \to 256$ | $256 \times 64 \times 64$ | 885k |
| e4 | pool, DoubleConv $256 \to 512$ | $512 \times 32 \times 32$ | 3.54M |
| b | pool, DoubleConv $512 \to 1024$ | $1024 \times 16 \times 16$ | 14.16M |
| d4 | up-conv $1024 \to 512$, concat e4, DoubleConv $1024 \to 512$ | $512 \times 32 \times 32$ | 2.10M + 7.08M |
| d3 | up-conv, concat e3, DoubleConv $512 \to 256$ | $256 \times 64 \times 64$ | 0.52M + 1.77M |
| d2 | up-conv, concat e2, DoubleConv $256 \to 128$ | $128 \times 128 \times 128$ | 131k + 443k |
| d1 | up-conv, concat e1, DoubleConv $128 \to 64$ | $64 \times 256 \times 256$ | 33k + 111k |
| out | $1 \times 1$ conv $64 \to K$ | $K \times 256 \times 256$ | $65K$ |

Total $\approx 31.0$M ($31{,}031{,}810$ for $K = 2$), with conv biases and no
batch norm; the bottleneck alone is 46%. Max-pool and concatenation have no
parameters. Four poolings: $H$ and $W$ must be divisible by $2^4 = 16$.

---

## DeepLab: atrous convolution and ASPP

![Dilated convolution at rates 1, 2, 3](assets/cv/deeplab-aspp.png)

Instead of downsampling then upsampling, keep the resolution and enlarge the
receptive field by spacing the kernel taps $r$ pixels apart:

$$
y[u] = \sum_{t \in \{-1,0,1\}^2} w[t]\; x[u + r\,t], \qquad k_{\text{eff}} = k + (k-1)(r-1)
$$

A $3 \times 3$ kernel at $r = 6$ covers $13 \times 13$ pixels with 9 weights.
DeepLabv3 replaces the strides of the last ResNet stages by dilation (output
stride 16 instead of 32), then applies **ASPP**: five parallel branches on the
same map — $1 \times 1$ conv, $3 \times 3$ at $r = 6, 12, 18$, global average
pooling — each 256 channels, concatenated to 1280, projected to 256.

For $3 \times 512 \times 512$: backbone $\to 2048 \times 32 \times 32$,
ASPP $\to 256 \times 32 \times 32$, $1 \times 1$ conv $\to K \times 32 \times 32$,
bilinear $\times 16 \to K \times 512 \times 512$. Learnt: all conv weights;
fixed: the rates and the upsampling.

<!-- placeholder: image to add (ASPP module diagram: parallel 1x1, 3x3 rate 6/12/18 and image pooling branches, concatenated — Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017, Fig. 5) -->

---

## Instance segmentation: Mask R-CNN

![Mask R-CNN](assets/cv/d2l-mask-rcnn.png)
*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Mask R-CNN (He et al., 2017) = Faster R-CNN + a small FCN that predicts a
mask inside each proposal. ResNet-50-FPN, COCO $K = 80$:

| Component | Input | Output | Learnt |
|---|---|---|---|
| backbone + FPN | $3 \times H \times W$ | $256 \times \frac{H}{s} \times \frac{W}{s}$, $s = 4, \dots, 32$ | yes |
| RPN | feature maps | ~1000 proposal boxes | yes |
| RoIAlign, box branch | features + box | $256 \times 7 \times 7$ per RoI | no |
| class + box head | $256 \times 7 \times 7$ | $K{+}1$ scores, $4K$ offsets | yes |
| RoIAlign, mask branch | features + box | $256 \times 14 \times 14$ per RoI | no |
| mask head: 4 conv, up-conv, $1 \times 1$ | $256 \times 14 \times 14$ | $K \times 28 \times 28$ logits | yes |

**RoIAlign** samples the feature map by bilinear interpolation at real-valued
positions; RoI pooling rounded the box to integer cells, a shift of up to 16
pixels at stride 16 — acceptable for a box, not for a mask.

The mask loss is a per-pixel binary cross-entropy on the channel of the
ground-truth class $k^*$ only, against the ground-truth mask cropped to the
RoI and resized to $28 \times 28$:

$$
\mathcal{L}_{mask} = -\frac{1}{28^2} \sum_{u} \Big[ y_u \log \sigma(z_{k^*,u}) + (1 - y_u) \log\big(1 - \sigma(z_{k^*,u})\big) \Big]
$$

Classes do not compete for pixels; the class head decides which. Total loss
$\mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{mask}$. At inference the
$28 \times 28$ mask of the predicted class is resized to the box and
thresholded at 0.5.

---

## Transformers: ViT with a linear decoder

A ViT gives one token per $16 \times 16$ patch. Put them back on the grid and
classify each one. Input $3 \times 512 \times 512$, $d = 768$:

| Component | Input | Output | Learnt |
|---|---|---|---|
| patch embedding | $3 \times 512 \times 512$ | $1024 \times 768$ | yes, 590k |
| $L$ transformer blocks | $1024 \times 768$ | $Z \in \mathbb{R}^{1024 \times 768}$ | yes |
| linear head $W \in \mathbb{R}^{768 \times K}$ | $1024 \times 768$ | $1024 \times K$ | yes, $769K$ |
| reshape | $1024 \times K$ | $K \times 32 \times 32$ | no |
| bilinear upsampling $\times 16$ | $K \times 32 \times 32$ | $K \times 512 \times 512$ | no |

Every token attends to every other from the first layer, so the receptive
field is global without dilation or pooling. The cost is a coarse
$32 \times 32$ grid. SegFormer (Xie et al., 2021) fixes it with a
hierarchical transformer encoder (maps at $\frac{1}{4}$ to $\frac{1}{32}$)
and a decoder made of MLPs that fuses the four scales.

---

## Mask2Former and SAM

<!-- placeholder: image to add (Mask2Former architecture: backbone, pixel decoder, transformer decoder with masked attention, N queries producing class + mask — Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation", CVPR 2022, Fig. 2) -->

Mask2Former (Cheng et al., 2022) is **DETR with masks**. A pixel decoder
produces per-pixel embeddings
$\mathcal{E}_{pixel} \in \mathbb{R}^{\frac{HW}{16} \times C}$ ($C = 256$).
A transformer decoder turns $N = 100$ learnt queries into
$q_1, \dots, q_N \in \mathbb{R}^{C}$. Each query yields a class and a mask:

$$
p_i = \mathrm{softmax}(W_{cls}\, q_i) \in \Delta^{K+1}, \qquad
m_i = \sigma\big(\mathcal{E}_{pixel}\; \mathrm{MLP}(q_i)\big) \in [0,1]^{\frac{H}{4} \times \frac{W}{4}}
$$

The extra class is "no object". Training uses Hungarian matching exactly as in
DETR, with a mask cost (BCE + Dice) added to the class cost. The same $N$
pairs $(p_i, m_i)$ answer all three tasks; for semantic segmentation,
$\hat y_u = \arg\max_k \sum_i p_i(k)\, m_i[u]$.

**SAM** (Kirillov et al., 2023) is promptable: a ViT image encoder run once, a
prompt (point, box or mask) and a light decoder return a mask in about
50 ms. Trained on 1.1 billion masks, it has no class labels: it segments, it
does not name.

<!-- placeholder: image to add (SAM: one image with point / box prompts and the returned masks — Kirillov et al., "Segment Anything", ICCV 2023, Fig. 1 or the SAM demo) -->

---

## In practice

`segmentation_models_pytorch` builds U-Net, FPN or DeepLabv3+ around any
ImageNet-pretrained encoder:

```python
import segmentation_models_pytorch as smp
model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=K)
dice = smp.losses.DiceLoss(mode="multiclass")
logits = model(x)                                   # (B, K, H, W)
loss = F.cross_entropy(logits, y) + dice(logits, y)  # y: (B, H, W), long
```

The ResNet encoder downsamples five times, so $H$ and $W$ must be divisible
by 32. A pretrained encoder is the most effective single choice on a small
dataset. For instance or panoptic outputs, use a pretrained Mask R-CNN
(torchvision) or Mask2Former (Hugging Face `transformers`).

---

## Exercise

**A. Metrics.** Ground truth $G$ and prediction $P$, foreground = 1:

```text
G          P
0 0 0 0    0 0 0 0
0 1 1 0    0 1 1 1
0 1 1 0    0 1 0 1
0 0 0 0    0 0 0 0
```

1. $TP$, $FP$, $FN$, $TN$ for the foreground. Pixel accuracy?
2. $\mathrm{IoU}_{fg}$, $\mathrm{IoU}_{bg}$, mIoU. $\mathrm{Dice}_{fg}$, and
   check $\mathrm{Dice} = 2\,\mathrm{IoU}/(1 + \mathrm{IoU})$.
3. Same metrics for a model that predicts background everywhere. What does
   each metric say about the two models?

**B. Shapes.** The padded U-Net of the table, $K = 5$.

1. Input $3 \times 320 \times 320$: shapes of e4, of the bottleneck, after the
   first up-conv, after the first concatenation, and of the output.
2. Why does a $3 \times 300 \times 300$ input crash?
3. `ConvTranspose2d(k=3, s=2, p=1)` on a $20 \times 20$ map: output size?
   Give $(k, s, p)$ that doubles it exactly.

<!-- notes: 15 minutes. In A3 most students expect accuracy to collapse;
it barely moves (0.81 → 0.75) while mIoU drops from 0.63 to 0.38. B2 is the
bug they will hit in the lab. -->

---

## Solution

**A1.** $TP = 3$, $FP = 2$ (right column), $FN = 1$ (position (3,3)),
$TN = 10$. $\mathrm{PA} = 13/16 = 0.81$.

**A2.** $\mathrm{IoU}_{fg} = 3/6 = 0.50$.
Background: $TP = 10$, $FP = 1$, $FN = 2$, $\mathrm{IoU}_{bg} = 10/13 = 0.77$.
$\mathrm{mIoU} = 0.63$.
$\mathrm{Dice}_{fg} = 6/9 = 0.67 = 2 \cdot 0.5 / 1.5$.

**A3.**

| Model | PA | $\mathrm{IoU}_{fg}$ | $\mathrm{IoU}_{bg}$ | mIoU | $\mathrm{Dice}_{fg}$ |
|---|---|---|---|---|---|
| $P$ | 0.81 | 0.50 | 0.77 | 0.63 | 0.67 |
| all background | 0.75 | 0 | 0.75 | 0.38 | 0 |

Pixel accuracy barely separates a model that finds the object from one that
ignores it; mIoU and Dice do.

**B1.** e4 $512 \times 40 \times 40$, bottleneck $1024 \times 20 \times 20$,
up-conv $512 \times 40 \times 40$, concat $1024 \times 40 \times 40$,
output $5 \times 320 \times 320$.

**B2.** $300 \to 150 \to 75 \to 37 \to 18$ (floor). The up-conv gives
$18 \to 36$, but e4 is $37 \times 37$: the concatenation fails. $H$ and $W$
must be multiples of 16.

**B3.** $(20 - 1) \cdot 2 - 2 + 3 = 39$, one pixel short (PyTorch adds
`output_padding=1` for this). $k = 2, s = 2, p = 0$ or $k = 4, s = 2, p = 1$:
$(H - 1) \cdot 2 - 2 + 4 = 2H$.
