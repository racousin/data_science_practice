# Self-Supervised Learning and DINOv2

ImageNet's 1.3 million labels took years of human work, and they describe
only one thing per image. The web holds billions of images with no labels at
all. Self-supervised learning trains a network on images alone: the
supervision comes from the data itself — two views of one photo should agree,
a hidden patch should be predictable from the others. DINOv2 is where that
line of work arrived: a ViT whose frozen features beat ImageNet-supervised
ones on almost every task, without seeing a single label.

<!-- notes: 45 minutes. Students know the ViT (previous lesson), BERT's masked
language model and next-token prediction (Session 6), and transfer learning
(Session 5). The arc: why labels are the bottleneck → contrastive (SimCLR) →
collapse and how to avoid it without negatives (BYOL, DINO) → masked image
modelling (MAE) → DINOv2 as the combination that works, and how to use it.
Spend the time on the two loss formulas and on collapse; the measured slide is
the one they will remember. -->

---

## Supervised, unsupervised, self-supervised

![Supervised, unsupervised and self-supervised learning](assets/cv/ssl-paradigms.png)
*Figure: X. Liu et al., [Self-supervised Learning: Generative or Contrastive](https://arxiv.org/abs/2006.08218), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Supervised,_Unsupervised,_and_Self-Supervised_Learning.jpg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

A self-supervised method invents a **pretext task** whose target is computed
from the input, trains on it, then throws the task away and keeps the encoder.
You have already met the most successful pretext tasks — in text:

| Domain | Pretext task | Model |
|---|---|---|
| text | predict a masked word | BERT (Session 6) |
| text | predict the next token | GPT (Session 8) |
| images | make two augmented views agree | SimCLR, BYOL, DINO |
| images | predict masked patches | MAE, iBOT |

The pre-trained encoder is then used as in Session 5: frozen with a linear
head, or fine-tuned.

---

## How a representation is judged

The pretext task's own loss says nothing useful. A representation is judged by
what a simple model can do with it, the encoder frozen:

| Protocol | What is trained on the labelled set | Measures |
|---|---|---|
| **linear probe** | one linear layer on the frozen features | linear separability of the classes |
| **k-NN** | nothing: vote of the $k$ nearest training features (cosine) | the geometry of the space itself |
| **fine-tuning** | every weight | how good an initialisation it is |
| **few-shot** | a linear probe on 1–100 labels per class | label efficiency |

ImageNet linear-probe top-1 is the field's scoreboard. Keep the distinction
in mind: a method can be excellent at fine-tuning and mediocre at linear
probing (MAE, later in this lesson).

---

## Contrastive learning: two views of one image

![Positive and negative pairs](assets/cv/ssl-contrastive-pairs.png)
*Figure: A potato hater, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Positive_and_negative_pairs_for_contrastive_learning.png), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Pull together the embeddings of a **positive pair**, push apart the
**negatives**. Without labels, "same class" is not available, so SimCLR
(Chen et al., 2020) uses the closest substitute: two random augmentations of
the **same image** are a positive pair; every other image in the batch is a
negative.

The augmentations define what the representation must ignore: random crop
and resize (position, scale), colour jitter and grayscale (colour), blur.
SimCLR's ablation: crop plus colour distortion is the combination that
matters. Crop alone lets the network match two views by their colour
histogram.

---

## SimCLR: architecture

| Component | Input | Output | Learnt |
|---|---|---|---|
| augment $t, t' \sim \mathcal{T}$ | image $x$ | views $\tilde x_i, \tilde x_j$ | no |
| encoder $f$ (ResNet-50) | $3 \times 224 \times 224$ | $h \in \mathbb{R}^{2048}$ | yes |
| projection head $g$ (2-layer MLP) | $h$ | $z \in \mathbb{R}^{128}$ | yes, discarded after |
| cosine similarity | $z_i, z_k$ | $s_{ik} = \frac{z_i^\top z_k}{\lVert z_i \rVert \lVert z_k \rVert}$ | no |

A batch of $N$ images gives $2N$ views. Each view has exactly one positive
(its twin) and $2N - 2$ negatives.

The projection head is thrown away after training: linear probes on $h$ beat
probes on $z$ by more than 10 points. The contrastive loss forces $z$ to
forget whatever the augmentations change; $h$ keeps it.

---

## The InfoNCE loss

For a positive pair $(i, j)$, with temperature $\tau$:

$$
\ell_{i,j} = -\log \frac{\exp(s_{ij}/\tau)}{\sum_{k=1, k \neq i}^{2N} \exp(s_{ik}/\tau)}
$$

averaged over all $2N$ anchors. Read it as a softmax classification with
$2N - 1$ classes: which of the other views is my twin? It is exactly a
cross-entropy, with the batch playing the role of the label set.

- **More negatives, harder task**: SimCLR needs batches of 4,096 (8,190
  negatives per anchor) — 128 TPU cores. **MoCo** (He et al., 2020) keeps a
  queue of 65,536 past embeddings from a slowly-moving copy of the encoder
  instead, and runs on 8 GPUs.
- **Temperature**: a small $\tau$ (0.1) sharpens the softmax and focuses the
  gradient on the hardest negatives.

ResNet-50, ImageNet linear probe: SimCLR 69.3%, against 76.5% for the same
network trained with labels.

---

## Collapse: the failure every method must avoid

Drop the negatives and keep only "two views must agree": the encoder that
outputs the **same constant vector** for every image is a perfect solution.
The loss is zero; the representation is worthless. This is **collapse**.

| Method | What prevents collapse |
|---|---|
| SimCLR, MoCo | negatives: the denominator of InfoNCE |
| BYOL (Grill et al., 2020) | asymmetry: a predictor on one branch, a target network that is an EMA of the online one, stop-gradient |
| DINO (Caron et al., 2021) | EMA teacher + **centering** and **sharpening** of its output |
| MAE | none needed: reconstructing pixels has no constant solution |

BYOL showed negatives are not necessary (74.3% linear probe on ResNet-50),
which removed the large-batch requirement. That opened the path to DINO.

---

## DINO: self-distillation with no labels

![DINO self-distillation](assets/cv/dino-self-distillation.png)
*Figure: M. Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Two copies of the same ViT. The **student** $g_{\theta_s}$ is trained by
gradient descent; the **teacher** $g_{\theta_t}$ is never trained — its
weights are an exponential moving average of the student's:

$$
\theta_t \leftarrow \lambda\, \theta_t + (1 - \lambda)\, \theta_s, \qquad
\lambda: 0.996 \to 1 \text{ (cosine schedule)}
$$

Each network ends in a head with $K = 65{,}536$ outputs, turned into a
distribution by a softmax with temperature. The student learns to predict
the teacher's distribution on a *different view* of the same image. There
are no classes: the $K$ dimensions are free "prototypes" the two networks
agree on.

---

## DINO: the loss, centering and sharpening

$$
P_t(x) = \mathrm{softmax}\!\Big(\frac{g_{\theta_t}(x) - c}{\tau_t}\Big), \qquad
P_s(x') = \mathrm{softmax}\!\Big(\frac{g_{\theta_s}(x')}{\tau_s}\Big)
$$

$$
\mathcal{L} = - \sum_{x \in \{x^g_1, x^g_2\}} \; \sum_{x' \neq x} \;
P_t(x)^\top \log P_s(x')
$$

| Ingredient | Effect alone |
|---|---|
| **sharpening**: $\tau_t = 0.04 < \tau_s = 0.1$ | the teacher's output becomes one-hot on one dimension → collapse to one prototype |
| **centering**: subtract $c \leftarrow 0.9\,c + 0.1\,\mu$, $\mu$ the batch mean of the teacher's outputs | no dimension can dominate → collapse to the uniform distribution |
| both | they cancel: a non-trivial, confident target |

The gradient flows only through the student (stop-gradient on the teacher).

---

## Collapse, measured

![Centering and sharpening ablation](assets/cv/dino-collapse.png)
*Figure: M. Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Left, the entropy of the teacher's output. With sharpening only it drops to
0 (one prototype for every image); with centering only it stays at the
maximum, $\log K$ (the uniform distribution). Right, the KL divergence
between teacher and student: in both collapsed runs it is 0 — the student
copies the teacher perfectly and learns nothing. Only the combination keeps
a target that is confident *and* informative.

---

## Multi-crop: local to global

![Multi-crop views of one photo](assets/cv/ssl-multicrop.png)
*Photo: [Lucasbosch](https://commons.wikimedia.org/wiki/File:Dalmatian_fetching_a_stick.jpg), Wikimedia Commons, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); views drawn with this lesson's augmentations.*

Each image gives two **global** views ($224^2$, covering more than half the
image) and several **local** views ($96^2$, under half). All views go through
the student; only the two global ones go through the teacher.

The student must therefore predict, from a $96 \times 96$ patch of fur, what
the teacher says about the whole dog: **local-to-global correspondence**. A
$96^2$ crop costs $(96/224)^2 \approx 18\%$ of a global one in patches, so
ten local views add less than two global ones in compute.

---

## What emerges: attention that segments

![DINO attention maps](assets/cv/dino-attention-banner.png)
*Figure: [facebookresearch/dino](https://github.com/facebookresearch/dino), Apache-2.0.*

The [CLS] token's attention in the last block of a DINO ViT, with no
segmentation label ever seen: it falls on the objects and ignores the
background.

---

## Different heads, different objects

![Multi-head attention of DINO](assets/cv/dino-heads.png)
*Figure: M. Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Each colour is one head of the last block. Different heads lock onto
different objects or parts — the vegetables and the knife, the sandwich and
the bowl, the sign and the water.

---

## Supervised ViT against DINO

![Supervised vs DINO attention](assets/cv/dino-vs-supervised.png)
*Figure: M. Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The same ViT-S/8, the best head, thresholded to keep 60% of the attention
mass. Trained with labels (top), the attention is scattered: classifying the
image needs only a few discriminative patches, so that is all it learns to
look at. Trained with DINO (bottom), it covers the object.

DINO ViT-S/16, ImageNet: 77.0% linear probe, 74.5% with a plain k-NN — a
nearest-neighbour vote in the frozen feature space is nearly as good as a
trained classifier. The same ViT-S trained with ImageNet's labels still wins
on ImageNet (79.8%): on the very task the labels describe, labels help. The
gap closes with data — next slides.

---

## Masked image modelling: MAE

![MAE architecture](assets/cv/mae-architecture.png)
*Figure: K. He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

BERT for images (He et al., 2022): hide 75% of the patches at random,
reconstruct their pixels.

| Component | Input | Output | Learnt |
|---|---|---|---|
| random mask, 75% | 196 patches | 49 visible patches | no |
| encoder (ViT) | the 49 visible tokens (+ [CLS]) | 49 × $D$ | yes |
| add mask tokens, positions | 49 × $D$ | 196 × $D_{dec}$ | one shared mask token |
| decoder (8 blocks, 512 wide) | 196 tokens | 196 × $P^2 C$ pixels | yes, discarded after |

The loss is the MSE on the **masked patches only**, against per-patch
normalised pixels. The encoder never sees a mask token, so it processes a
quarter of the sequence: with attention quadratic in length, pre-training is
three times faster or more.

---

## MAE: what it reconstructs, what it learns

![MAE reconstructions](assets/cv/mae-samples.png)
*Figure: K. He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Each triplet: masked input, reconstruction, original (ImageNet validation).*

Why 75% and not BERT's 15%: images are redundant. With few patches masked,
a hole is filled by interpolating its neighbours — no understanding needed.

---

## MAE: the masking ratio

![Masking ratio](assets/cv/mae-masking-ratio.png)
*Figure: K. He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). ViT-L, ImageNet top-1 against the masking ratio (%): fine-tuning (top), linear probe (bottom).*

ViT-L at 75%: 84.9% fine-tuned, but 73.5% linear probe. MAE features are an
excellent initialisation and poor frozen features — pixels force the encoder
to keep low-level detail that a linear classifier cannot use. Joint-embedding
methods (DINO) are the opposite.

---

## DINOv2: the recipe

DINOv2 (Oquab et al., 2023) combines both families and scales the data:

| Ingredient | What it does |
|---|---|
| image-level DINO loss on [CLS] | global semantics, as above |
| patch-level **iBOT** loss: masked patches of the student predict the teacher's patch tokens | dense features: masked modelling in feature space, not pixels |
| **KoLeo** regulariser | spreads the features uniformly in the batch |
| **LVD-142M** data | 142M images selected from 1.2B web images by de-duplication and retrieval of images close to curated datasets |
| ViT-g/14 (1.1B parameters), then **distillation** | the giant is the teacher of ViT-S/B/L students |
| last epochs at $518^2$ | sharper dense features |

Patch size 14: a $224^2$ image gives $16 \times 16 = 256$ patch tokens.
Data curation mattered as much as the losses: the same recipe on 142M
*uncurated* images is clearly worse.

---

## DINOv2 features: no labels, real parts

![PCA of DINOv2 patch features](assets/cv/dinov2-pca.png)
*Figure computed with DINOv2 ViT-B/14 (registers) by this course; photos: [Lucasbosch](https://commons.wikimedia.org/wiki/File:Dalmatian_fetching_a_stick.jpg), [Dietmar Rabich](https://commons.wikimedia.org/wiki/File:D%C3%BClmen,_Hausd%C3%BClmen,_Golden_Retriever_--_2022_--_5945.jpg), [Terragio67](https://commons.wikimedia.org/wiki/File:Calico_cat,_-_Assisi,_Italy.jpg), [Rhododendrites](https://commons.wikimedia.org/wiki/File:Cedar_waxwing_in_pokeweed_(10132).jpg), [JJ Harrison](https://commons.wikimedia.org/wiki/File:Black-naped_Monarch_0A2A8267.jpg), Wikimedia Commons, [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).*

The $32 \times 32$ patch tokens of five photos at $448^2$. The foreground is
where the [CLS] attends, grown to the whole object by two-means on the patch
features; one PCA fitted on the foreground patches of all five maps its first
three components to RGB. No mask and no label were ever given, yet the
objects come out with clean outlines. Matching colours mean matching
features: the two birds, two different species, take the same colours part
by part — head, body, tail. The model has learned **parts** from images
alone.

---

## Correspondence across images

![DINOv2 patch matching](assets/cv/dinov2-matching.png)
*Figure computed with DINOv2 ViT-B/14 (registers) by this course; photos: [Lucasbosch](https://commons.wikimedia.org/wiki/File:Dalmatian_fetching_a_stick.jpg), [AngMoKio](https://commons.wikimedia.org/wiki/File:Greyhound_Racing_2_amk.jpg), Wikimedia Commons, [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).*

Each line joins a patch of the left dog to its most similar patch (cosine) on
the right dog, kept only when the match is mutual and on the foreground.
Head goes to head, back to back, legs to legs — across two breeds, two
coats, two backgrounds. Nothing in training said what a leg is.

---

## Measured: frozen features with few labels

![DINOv2 vs supervised ResNet-50 with few labels](assets/cv/dinov2-few-labels.png)

CIFAR-10, frozen features + logistic regression, $k$ labelled images per
class (three random draws averaged), 2,000 test images, $224^2$ inputs
(`tools/figures/ms2a_s6_dinov2.py`). Parameters are the encoders' without heads:

| Labels per class | 1 | 10 | 100 | 500 |
|---|---|---|---|---|
| ResNet-50, ImageNet labels (23.5M) | 37.7% | 79.1% | 88.4% | 90.9% |
| DINOv2 ViT-S/14, no labels (22.1M) | 65.9% | 92.9% | 96.6% | 96.8% |

Same size, no labels in pre-training, and the DINOv2 features are better at
every budget. **Ten labelled images per class with DINOv2 beat five hundred
with the supervised ResNet** (92.9% against 90.9%). One label per class already
gives 65.9%. Pre-training on 142M curated images bought what labels used to
buy.

---

## Using DINOv2

```python
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").eval()
x = torch.randn(1, 3, 224, 224)            # sides: multiples of 14
cls = model(x)                             # (1, 384)  image embedding
out = model.forward_features(x)
patches = out["x_norm_patchtokens"]        # (1, 256, 384)  one per patch
```

| Variant | Parameters | Width | ImageNet linear | k-NN |
|---|---|---|---|---|
| ViT-S/14 | 21M | 384 | 81.1% | 79.0% |
| ViT-B/14 | 86M | 768 | 84.5% | 82.1% |
| ViT-L/14 | 300M | 1024 | 86.3% | 83.5% |
| ViT-g/14 | 1.1B | 1536 | 86.5% | 83.5% |

`_reg` models add 4 **register** tokens (Darcet et al., 2024): extra tokens
that absorb the high-norm "artefact" patches a large ViT otherwise writes into
the background, giving cleaner patch features and attention maps. In timm:
`vit_small_patch14_reg4_dinov2`. Use the [CLS] for classification and
retrieval, the patch tokens for segmentation, depth or matching.

---

## What to use when

| Your situation | Start from |
|---|---|
| a classification task, few labels | DINOv2 [CLS] + logistic regression; k-NN as the zero-training baseline |
| retrieval, de-duplication, clustering | DINOv2 [CLS], cosine similarity |
| segmentation or depth with little data | DINOv2 patch tokens + a linear or small head |
| text queries on images, zero-shot labels | CLIP (image–text contrastive) |
| plenty of labels, compute to fine-tune | fine-tune DINOv2 or an MAE-pre-trained ViT |
| unlabeled data from your own domain (medical, satellite) | continue self-supervised pre-training on it |

The current frontier, **DINOv3** (Siméoni et al., 2025), scales the same
recipe to a 7B-parameter teacher on 1.7B images, with a "Gram anchoring" loss
that keeps the patch features from degrading over long training.

---

## Exercise

**A. SimCLR batch.** Batch of $N = 512$ images, $\tau = 0.1$.

1. How many views, positives per anchor, negatives per anchor?
2. At initialisation all similarities are about equal. What is the InfoNCE
   loss, approximately?

**B. DINO multi-crop.** ViT-S/16; two global $224^2$ views, ten local
$96^2$ views.

1. Tokens (with [CLS]) of a global view? Of a local view?
2. How many (teacher, student) pairs enter the loss for one image?
3. Student tokens per image, against two global views only?

**C. MAE.** ViT-B/16 at $224^2$, 75% masking.

1. Tokens the encoder processes? Ratio of attention-matrix entries against
   the unmasked image?

**D. DINOv2 at 448.** ViT-S/14 with registers on a $448^2$ image: shape of
`x_norm_patchtokens`?

<!-- notes: 15 minutes in pairs. A2 is the one to discuss: log(2N-1) is the
loss of a classifier that knows nothing, the number to compare the training
curve against. -->

---

## Solution

**A1.** $2N = 1024$ views; 1 positive; $2N - 2 = 1022$ negatives.

**A2.** All $2N - 1 = 1023$ terms of the softmax equal: the probability of
the positive is $1/1023$, so $\ell \approx \log 1023 \approx 6.93$. A loss
stuck there means nothing is being learned; $\tau$ does not change it.

**B1.** Global: $(224/16)^2 + 1 = 197$. Local: $(96/16)^2 + 1 = 37$.

**B2.** Each of the 2 teacher views is paired with every *other* student
view: $2 \times (12 - 1) = 22$ pairs.

**B3.** $2 \cdot 197 + 10 \cdot 37 = 764$ tokens against $394$: 1.9×.

**C1.** $196 \times 0.25 = 49$ patch tokens, plus the [CLS]: 50 against
197. Attention entries: $(50/197)^2 \approx 1/15.5$ of the full image.

**D.** $(448/14)^2 = 32^2 = 1024$ patches: shape $(1, 1024, 384)$. The [CLS]
and the 4 registers are returned separately.

---

## Check yourself

1. A colleague trains SimCLR with crop augmentation only and gets features
   that cluster images by their dominant colour. What went wrong?

   **Answer.** Two crops of one image share its colour histogram, so the
   network can match positives by colour alone and has no reason to learn
   anything else. Colour jitter and grayscale remove that shortcut: the
   augmentations define what the features must be invariant to.

2. In DINO, what happens if you remove the centering? The sharpening?

   **Answer.** Without centering the sharpened teacher puts all its mass on
   one dimension for every image — collapse to a constant. Without
   sharpening, centering pushes the output to uniform — the other collapse.
   In both cases the student matches the teacher perfectly (KL 0) and the
   features are useless.

3. MAE gives 73.5% linear probe but 84.9% fine-tuned on ViT-L. Which would you
   use frozen for a k-NN retrieval system, MAE or DINOv2, and why?

   **Answer.** DINOv2. Joint-embedding training shapes the feature space
   itself so that similar images are close; pixel reconstruction keeps
   low-level detail and needs fine-tuning to become linearly separable.
   k-NN uses the raw geometry, which is exactly what MAE does not optimise.
