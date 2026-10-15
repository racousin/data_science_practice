# Vision Transformer

The Vision Transformer (ViT, Dosovitskiy et al., 2021) is the encoder from
Session 6 applied to an image, with one new component: a way to turn pixels into
tokens. Cut the image into square patches, flatten each patch, project it to the
model width — from there on nothing is specific to vision. What changes is not
the architecture but what the model knows before training, and therefore how
much data it needs.

<!-- notes: 35 minutes. Students know the transformer block from Session 6:
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


![focus-your-attention-towards-data-intuitive-lightweight-vision-transformers-1.png](assets/cv/focus-your-attention-towards-data-intuitive-lightweight-vision-transformers-1.png)

 $P$ is a hyperparameter.

---

## The [CLS] token and position embeddings

Two more learned tensors, as in BERT:

- a **[CLS] token** $x_{cls} \in \mathbb{R}^{D}$, prepended to the patches; its
  final state will summarise the image
- **position embeddings** $E_{pos} \in \mathbb{R}^{(N+1) \times D}$, one row per
  position, learned from scratch — not sinusoidal

$$
X_0 = \big[\,x_{cls};\; x_p^1 E;\; x_p^2 E;\; \dots;\; x_p^N E\,\big] + E_{pos}
\;\in\; \mathbb{R}^{(N+1) \times D}
$$

$X_0$ is exactly the $X_0$ of Session 6, with $n = N + 1$ tokens and width
$d = D$ (the ViT paper writes it $z_0$). Attention is permutation-equivariant,
so without $E_{pos}$ the model would see a bag of patches.


![fig04_cls_prepend.png](assets/cv/fig04_cls_prepend.png)


<!-- placeholder: image to add (cosine similarity of the learned position embeddings of ViT-L/32, one small map per patch position showing row/column structure; Dosovitskiy et al. 2021, Fig. 7 centre) -->

---

## The encoder and the head

![The Vision Transformer architecture](assets/cv/d2l-vit.png)
*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

$L$ copies of the pre-norm block from Session 6, bidirectional, no mask — every
patch attends to every patch.

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
decays towards the input (layer-wise decay, one group per block).


---

## Exercise: ViT on CIFAR-10

**A. A small ViT from scratch example.** CIFAR-10: $32 \times 32$ RGB images,
10 classes. Patch size $P = 4$, $D = 192$, $L = 6$ blocks, $h = 3$ heads,
FFN width $4D$.

1. $N$? Length of one flattened patch? Shape of $E$?
2. Why not keep $P = 16$ as in ViT-B/16?
3. What other issue do you see with Vit? How to solve them?


<!-- notes: 15 minutes in pairs. Give them the block formula 12D^2 + 13D only if
they are stuck. A4 is the discussion question: P=16 gives 4 tokens. -->

---

## Solution

**A1.** $N = 32^2 / 4^2 = 64$; a patch is $4 \cdot 4 \cdot 3 = 48$ values;
$E \in \mathbb{R}^{48 \times 192}$.

**A2.** $P = 16$ gives $N = 4$ tokens: attention over four patches, each
already a quarter of the image. The patch size must scale with the image; the
price of $P = 4$ is $N^2 = 4096$ attention entries per head instead of 16.

**A3.** Patch structural issues
- Hard, non-overlapping boundaries cut edges and objects arbitrarily.
- Intra-patch structure is reduced to a single linear map $E$, with no local
  hierarchy.
- The $N^2$ cost pushes toward large $P$, which loses fine detail.

-> data augmentation is needed



![qwe.png](assets/cv/qwe.png)
