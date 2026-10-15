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

## Contrastive learning: two views of one image


![asd.png](assets/cv/asd.png)


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

## Contrastive loss

A batch of $N$ images gives $2N$ views. For a positive pair $(i, j)$:

$$
\ell_{i,j} = -\log
\frac{\exp\big(\mathrm{sim}(z_i, z_j)/\tau\big)}
{\sum_{k \neq i} \exp\big(\mathrm{sim}(z_i, z_k)/\tau\big)},
\qquad
\mathrm{sim}(u, v) = \frac{u^\top v}{\lVert u \rVert\, \lVert v \rVert}
$$

The total loss averages $\ell_{i,j}$ over all $2N$ positive pairs.

- It is a softmax cross-entropy: among $2N - 1$ candidates, **classify which
  one is the other view** of the same image.
- Numerator pulls the positive together; denominator pushes the
  $2N - 2$ negatives apart.
- Temperature $\tau$: small $\tau$ sharpens the softmax and focuses on hard
  negatives.
- More negatives give a harder task, hence SimCLR's large batches.

---
## MAE

![1_w66BoSlqp8LTOQzndbPFqQ.png](assets/cv/1_w66BoSlqp8LTOQzndbPFqQ.png)

---

## MAE loss: reconstruct what was hidden

Split the image into patches, mask a random subset $M$ (75%), and encode
only the visible ones. A light decoder predicts the pixels of every patch,
and the loss counts **only the masked patches**:

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{p \in M}
\big\lVert \hat{x}_p - x_p \big\rVert_2^2
$$

- $x_p$ = pixels of patch $p$, normalised per patch (mean / std).
  This works better than raw pixels.
- No negatives, no augmentation pairs: the supervision is the image itself.
- Loss on visible patches would be trivial copying, which is why they are
  excluded.
- A high mask ratio prevents filling holes by interpolating neighbours. The
  model must infer the content.

---
## DINO: self-distillation with no labels


![dino_ben.png](assets/cv/dino_ben.png)


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


## Supervised ViT against DINO

![Supervised vs DINO attention](assets/cv/dino-vs-supervised.png)
*Figure: M. Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The same ViT-S/8, the best head, thresholded to keep 60% of the attention
mass. Trained with labels (top), the attention is scattered: classifying the
image needs only a few discriminative patches, so that is all it learns to
look at. Trained with DINO (bottom), it covers the object.

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
every budget.

---

## Using DINOv2

```python
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").eval()
x = torch.randn(1, 3, 224, 224)            # sides: multiples of 14
cls = model(x)                             # (1, 384)  image embedding
out = model.forward_features(x)
patches = out["x_norm_patchtokens"]        # (1, 256, 384)  one per patch
```
**DINOv3** is the last version 2026.
