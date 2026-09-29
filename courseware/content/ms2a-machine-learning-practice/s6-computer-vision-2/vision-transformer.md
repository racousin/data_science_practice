# Vision Transformer

The Vision Transformer (ViT, Dosovitskiy et al., 2021) is the encoder from
Session 7 applied to an image, with one new component: a way to turn pixels into
tokens. Cut the image into square patches, flatten each patch, project it to the
model width — from there on nothing is specific to vision. What changes is not
the architecture but what the model knows before training, and therefore how
much data it needs.

<!-- notes: 35 minutes. Students know the transformer block from Session 7:
do not re-derive attention, point back to it. Spend the time on three things:
the patch embedding (and that it is a strided convolution), the parameter count
(98% of the weights are in the blocks, against a fifth in BERT's embedding
table), and the inductive-bias argument for why ViT needs so much data. The
position-embedding interpolation slide is the one they will need in practice. -->

---

## From pixels to tokens

An image $x \in \mathbb{R}^{H \times W \times C}$ is cut into non-overlapping
$P \times P$ patches, each flattened to a vector:

$$
N = \frac{HW}{P^2}, \qquad
x_p \in \mathbb{R}^{N \times (P^2 C)}
$$

Each patch is then projected to the model width $D$ by one learned matrix, the
**patch embedding**:

$$
E \in \mathbb{R}^{(P^2 C) \times D}, \qquad x_p^i E \in \mathbb{R}^{D}
$$

| | Input | Output | Learnt |
|---|---|---|---|
| Patchify | $H \times W \times C$ | $N \times P^2C$ | nothing — $P$ is a hyperparameter |
| Patch embedding | $N \times P^2C$ | $N \times D$ | $E$ and a bias: $P^2 C D + D$ |

The same $E$ is applied to every patch. A linear map on non-overlapping
$P \times P$ windows, shared across positions, is a convolution with
kernel $= $ stride $= P$ and $D$ output channels — which is how every
implementation writes it:

```python
patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)  # (B,3,224,224) -> (B,768,14,14)
tokens = patch_embed(x).flatten(2).transpose(1, 2)          # (B,196,768)
```

A patch plays the role of a word; $E$ plays the role of the embedding table of
Session 7, except that its input is a continuous vector, not an index.

---

## The [CLS] token and position embeddings

![A [CLS] token prepended to the sequence](assets/nlp/cls.png)

Two more learned tensors, as in BERT:

- a **[CLS] token** $x_{cls} \in \mathbb{R}^{D}$, prepended to the patches; its
  final state will summarise the image
- **position embeddings** $E_{pos} \in \mathbb{R}^{(N+1) \times D}$, one row per
  position, learned from scratch — not sinusoidal

$$
X_0 = \big[\,x_{cls};\; x_p^1 E;\; x_p^2 E;\; \dots;\; x_p^N E\,\big] + E_{pos}
\;\in\; \mathbb{R}^{(N+1) \times D}
$$

$X_0$ is exactly the $X_0$ of Session 7, with $n = N + 1$ tokens and width
$d = D$ (the ViT paper writes it $z_0$). Attention is permutation-equivariant,
so without $E_{pos}$ the model would see a bag of patches.

$E_{pos}$ is a 1D table: position $i$ is only an index, the 2D grid is not
given. The paper reports that 2D-aware variants do no better — the learned
embeddings recover the row and column structure on their own.

<!-- placeholder: image to add (cosine similarity of the learned position embeddings of ViT-L/32, one small map per patch position showing row/column structure; Dosovitskiy et al. 2021, Fig. 7 centre) -->

---

## The encoder and the head

![The Vision Transformer architecture](assets/cv/d2l-vit.png)
*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

$L$ copies of the pre-norm block from Session 7, bidirectional, no mask — every
patch attends to every patch:

$$
H_b = X_b + \mathrm{MHA}\big(LN_1(X_b)\big), \qquad
X_{b+1} = H_b + \mathrm{FFN}\big(LN_2(H_b)\big), \qquad b = 0, \dots, L-1
$$

The FFN has hidden width $4D$ and a GELU activation. The output
$X_L \in \mathbb{R}^{(N+1) \times D}$ has one row per token; classification
reads only row $0$, the [CLS] token, through a final layer norm and a linear
head:

$$
y = LN\big(X_L^{0}\big)\, W_{head} + b_{head}, \qquad
W_{head} \in \mathbb{R}^{D \times K}, \quad y \in \mathbb{R}^{K}
$$

The $N$ patch outputs are discarded. Averaging them instead of using [CLS]
(global average pooling, as in a CNN) works as well and is a common variant.

---

## ViT-B/16: dimensional flow

ViT-B/16: $224 \times 224$ RGB input, $P = 16$, $D = 768$, $L = 12$ blocks,
$h = 12$ heads, FFN width $3072$, $K = 1000$ classes.

| # | Step | Tensor | Shape |
|---|---|---|---|
| 1 | input image | $x$ | $224 \times 224 \times 3$ |
| 2 | patchify, $N = 224^2/16^2$ | $x_p$ | $196 \times 768$ ($16 \cdot 16 \cdot 3$) |
| 3 | patch embedding | $x_p E$ | $196 \times 768$ ($D$) |
| 4 | prepend [CLS], add $E_{pos}$ | $X_0$ | $197 \times 768$ |
| 5 | per head, in each block | $Q_i, K_i, V_i$ | $197 \times 64$ |
| 6 | attention matrix, per head | $A_i$ | $197 \times 197$ (12 of them) |
| 7 | attention sublayer + residual | $H_b$ | $197 \times 768$ |
| 8 | FFN hidden layer | $\phi(\cdot\, W_1 + b_1)$ | $197 \times 3072$ |
| 9 | block output, after 12 blocks | $X_{12}$ | $197 \times 768$ |
| 10 | take the [CLS] row, $LN$ | $LN(X_{12}^0)$ | $768$ |
| 11 | head | $y$ | $1000$ |

Rows 2 and 3 both read 768 by coincidence: $P^2 C = 16 \cdot 16 \cdot 3 = 768 = D$.
In general they differ — $E$ is square only for this configuration.

---

## ViT-B/16: parameters

| Component | Count | Parameters |
|---|---|---|
| Patch embedding $E$ + bias | $P^2 C \cdot D + D = 768 \cdot 768 + 768$ | 590,592 |
| [CLS] token | $D$ | 768 |
| Position embeddings | $(N+1) \cdot D = 197 \cdot 768$ | 151,296 |
| One block: MHA | $4D^2 + 4D$ | 2,362,368 |
| One block: FFN | $D \cdot 4D + 4D + 4D \cdot D + D = 8D^2 + 5D$ | 4,722,432 |
| One block: 2 layer norms | $4D$ | 3,072 |
| **One block** | $12D^2 + 13D$ | **7,087,872** |
| 12 blocks | | 85,054,464 |
| Final layer norm | $2D$ | 1,536 |
| Head | $D \cdot K + K = 768 \cdot 1000 + 1000$ | 769,000 |
| **Total** | | **86,567,656** |

The blocks hold 98.3% of the weights. Everything that is specific to images —
$E$, [CLS], $E_{pos}$ — is 0.86%. Compare BERT-base, where the vocabulary table
alone is about a fifth of the model: a patch embedding replaces a 30,000-row
lookup by one $768 \times 768$ matrix.

As in Session 7, no count depends on $N$ except $E_{pos}$. The same weights can
read an image of another size — once $E_{pos}$ is resized (later slide).

---

## Inductive bias

A CNN is told, by its architecture, what images are like (Session 5,
*Pooling and CNN Architectures*):

| Assumption | CNN | ViT |
|---|---|---|
| Locality — nearby pixels matter first | $3 \times 3$ kernels | only inside a patch; attention is global from block 1 |
| Translation equivariance | weight sharing across positions | only in $E$; attention depends on learned $E_{pos}$ |
| 2D neighbourhood | built into the kernel | not given; learned into $E_{pos}$ |
| Hierarchy, growing receptive field | pooling, strided convolutions | none; $N$ tokens at every depth |

A ViT is a general sequence model that happens to receive patches. Everything a
CNN assumes, it must learn from data. That is a disadvantage when data is
scarce and an advantage when it is not: an assumption that is built in cannot
be relaxed where it is wrong.

---

## ViT needs data

<!-- placeholder: image to add (ImageNet top-1 of ViT variants vs BiT ResNets as a function of the pre-training dataset: ImageNet-1k, ImageNet-21k, JFT-300M; Dosovitskiy et al. 2021, Fig. 3) -->

The central experiment of the ViT paper: pre-train on datasets of increasing
size, fine-tune on ImageNet.

| Pre-training data | Images | ViT vs ResNet (BiT) |
|---|---|---|
| ImageNet-1k | 1.3M | ResNets better; larger ViTs worse than smaller ones |
| ImageNet-21k | 14M | comparable |
| JFT-300M | 300M | ViT better, and at lower pre-training compute |

On small data a CNN wins, because its assumptions are close to correct for
natural images. Given enough data, learning the structure beats assuming it.

**DeiT** (Touvron et al., 2021) closed most of the gap on ImageNet-1k alone,
with strong augmentation and regularisation (RandAugment, Mixup, CutMix) and
distillation from a CNN teacher through an extra token. For your own small
dataset, the lesson is the same as in Session 5: do not train a ViT from
scratch — start from pre-trained weights.

---

## What the heads attend to

<!-- placeholder: image to add (mean attention distance in pixels per head vs network depth for ViT-L/16; Dosovitskiy et al. 2021, Fig. 7 right) -->

For each head, the **mean attention distance** is the average pixel distance
between a query patch and the patches it attends to, weighted by the attention
weights — the analogue of a receptive field.

- In the first blocks, some heads are local (a few patches) and others already
  global: the whole image is one hop away, which no CNN layer allows.
- With depth, the distance grows and all heads become global.

The local heads in early blocks are learned, not imposed: the model rediscovers
part of the CNN prior because the data rewards it.

<!-- placeholder: image to add (attention from the [CLS] token to the input patches, rolled out over all layers, overlaid on a few input images, e.g. a dog; Dosovitskiy et al. 2021, Fig. 6) -->

The [CLS] attention, accumulated over layers, concentrates on the object that
determines the label. As in Session 7: evidence about the computation, not an
explanation of the prediction.

---

## Fine-tuning with timm

```python
import timm
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
cfg = timm.data.resolve_model_data_config(model)
tf = timm.data.create_transform(**cfg, is_training=True)
```

`num_classes=10` replaces $W_{head}$ by a new $768 \times 10$ layer (7,690
parameters); every other weight is pre-trained. `resolve_model_data_config`
matters more than for a ResNet: many ViT checkpoints normalise with mean and
standard deviation $0.5$, not the ImageNet statistics.

The recipe is the one from Session 5, *Transfer Learning*: head first with the
backbone frozen, then all blocks at a low learning rate, with a rate that
decays towards the input (layer-wise decay, one group per block). The
768-dimensional $LN(X_L^0)$ is also a strong frozen feature for a logistic
regression.

---

## Fine-tuning at a higher resolution

Fine-tuning at a higher resolution than pre-training usually improves accuracy.
Keep $P$; the image gives more patches:

$$
224 \to 384: \qquad N = 14^2 = 196 \;\to\; N' = 24^2 = 576
$$

$E$, the blocks and the head do not depend on $N$ and are reused unchanged.
$E_{pos} \in \mathbb{R}^{197 \times D}$ has no row for positions 198 to 577,
so it is resized as an image:

| Step | Shape |
|---|---|
| split off the [CLS] row | $1 \times D$ and $196 \times D$ |
| reshape the patch rows to their grid | $14 \times 14 \times D$ |
| 2D bicubic interpolation | $24 \times 24 \times D$ |
| flatten, put [CLS] back | $577 \times D$ |

This is the one place where the 2D structure of the image is injected by hand.
`timm.create_model(..., pretrained=True, img_size=384)` does it when loading
the checkpoint. The cost: each attention matrix grows from $197^2$ to $577^2$
entries, about 8.6 times more.

---

## Beyond ViT

- **Swin Transformer** (Liu et al., 2021): attention inside local windows,
  shifted between blocks, and patch merging that halves the grid between
  stages. It puts locality and hierarchy back — a CNN's silhouette with
  attention inside — and its cost becomes linear in the number of patches.
- **DINOv2** (self-supervised on images) and **CLIP** (trained on image–text
  pairs) are ViTs pre-trained without class labels. Used frozen, their [CLS]
  or pooled features with a linear head are among the strongest starting
  points for classification, and both are available in timm.

The block is unchanged in all of them. They differ in the patch arrangement and
in the pre-training data and objective.

---

## Exercise: ViT on CIFAR-10

**A. A small ViT from scratch.** CIFAR-10: $32 \times 32$ RGB images,
10 classes. Patch size $P = 4$, $D = 192$, $L = 6$ blocks, $h = 3$ heads,
FFN width $4D$.

1. $N$? Length of one flattened patch? Shape of $E$?
2. Shape of $X_0$? Of $Q_i$ and of $A_i$ in one head?
3. Parameters of: $E$ (with bias), $E_{pos}$, one block, the head. Total?
4. Why not keep $P = 16$ as in ViT-B/16?

**B. ViT-B/16 at 384.** You fine-tune the ImageNet ViT-B/16 on
$384 \times 384$ images, with 10 classes.

1. New $N$, and shape of $X_0$?
2. Which parameters change shape, and which are reused unchanged?

<!-- notes: 15 minutes in pairs. Give them the block formula 12D^2 + 13D only if
they are stuck. A4 is the discussion question: P=16 gives 4 tokens. -->

---

## Solution

**A1.** $N = 32^2 / 4^2 = 64$; a patch is $4 \cdot 4 \cdot 3 = 48$ values;
$E \in \mathbb{R}^{48 \times 192}$.

**A2.** $X_0 \in \mathbb{R}^{65 \times 192}$ (64 patches + [CLS]).
$Q_i \in \mathbb{R}^{65 \times 64}$ ($D/h = 64$), $A_i \in \mathbb{R}^{65 \times 65}$.

**A3.**

| Component | Computation | Parameters |
|---|---|---|
| $E$ + bias | $48 \cdot 192 + 192$ | 9,408 |
| [CLS] | $192$ | 192 |
| $E_{pos}$ | $65 \cdot 192$ | 12,480 |
| One block | $12 \cdot 192^2 + 13 \cdot 192$ | 444,864 |
| 6 blocks | | 2,669,184 |
| Final layer norm | $2 \cdot 192$ | 384 |
| Head | $192 \cdot 10 + 10$ | 1,930 |
| **Total** | | **2,693,578** |

**A4.** $P = 16$ gives $N = 4$ tokens: attention over four patches, each
already a quarter of the image. The patch size must scale with the image; the
price of $P = 4$ is $N^2 = 4096$ attention entries per head instead of 16.

**B1.** $N = (384/16)^2 = 576$, $X_0 \in \mathbb{R}^{577 \times 768}$.

**B2.** $E_{pos}$ is interpolated from $197 \times 768$ to $577 \times 768$
(443,136 parameters). The head is replaced: $768 \cdot 10 + 10 = 7{,}690$.
$E$, [CLS], the 12 blocks and the final layer norm are reused unchanged — none
of them depends on $N$.
