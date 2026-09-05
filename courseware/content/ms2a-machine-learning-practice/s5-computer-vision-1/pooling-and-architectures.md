# Pooling and CNN Architectures

Convolution gives you features. Getting from a feature map to a prediction needs
two more ideas: a way to shrink the spatial grid, and a way to stack blocks deep
enough to be useful. Thirty years of architecture research is mostly the second
one.

<!-- notes: 30 minutes. Pooling in 8, the lineage in 20. Do not lecture every
architecture — LeNet to VGG is context, ResNet is the one they must understand,
and the last two slides are a purchasing decision. -->

---

## Pooling

![Max pooling and average pooling over 2x2 windows](assets/cv/pooling.png)

Slide a window, reduce each window to one number, per channel independently. No
parameters, no learning.

$$
G[c,i,j] = \max_{0 \leq p, q < K} F[c,\, Si+p,\, Sj+q]
$$

A 2×2 window at stride 2 quarters the number of positions. Three of them in a
row take 224×224 down to 28×28, and the receptive field of everything downstream
grows eight times faster.

---

## Max or average

| | Max pooling | Average pooling |
|---|---|---|
| Keeps | the strongest response | the mean response |
| Effect | sharpens, discards | smooths, blurs |
| Invariance | to small translations of the peak | to noise |
| Typical use | inside the feature extractor | as a global head |

Max pooling won empirically: a filter response says "this pattern is present
somewhere in this window", and the maximum is the natural summary of that claim.

Output size uses the convolution formula with $P = 0$ — padding a pooling layer
is unusual, since the point is to reduce:

$$
O = \left\lfloor \frac{N - K}{S} \right\rfloor + 1
$$

---

## Global average pooling

```python
nn.AdaptiveAvgPool2d(1)     # (B, C, H, W) -> (B, C, 1, 1), any H, W
```

Average each channel over its entire spatial extent. One number per channel, and
the output size no longer depends on the input size.

$$
\hat{y}_c = \frac{1}{HW} \sum_{i,j} G[c,i,j]
$$

This is what killed the giant fully connected head.

---

## What it replaced

VGG-16 flattens a 7×7×512 map into 25,088 values and runs it into two 4096-unit
dense layers. The first of those alone is 103 million parameters; the whole head
is 124 million, about 90% of the network, in the part that does the least.

| Model | Head | Head parameters |
|---|---|---|
| VGG-16 | flatten → 4096 → 4096 → 1000 | ~124 M |
| ResNet-50 | GAP → 1000 | 2.05 M |

The GAP head also accepts any input resolution, and it makes each channel
directly interpretable as evidence for a class. There is no reason to build a
flatten-and-dense head in 2026.

---

## Stride is pooling that learns

A stride-2 convolution downsamples too, and its weights are trainable. Modern
architectures — ResNet's later stages, ConvNeXt, most detection backbones — use
strided convolution for downsampling and keep pooling only for the global head.

> Default: one max-pool after the stem, strided convolutions after that, global
> average pooling at the end.

Pooling is not wrong; it is simply the parameter-free special case.

---

## The canonical shape

![A LeNet-style CNN on MNIST](assets/cv/cnn-network.jpg)

Every classification CNN has the same silhouette: alternating convolution and
downsampling, spatial size falling while channel count rises, then a head.

```python
nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(2),                                 # 224 -> 112
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
)
```

Halve the resolution, double the channels. The compute per stage stays roughly
constant, which is not a coincidence.

---

## LeNet-5 (1998) and AlexNet (2012)

**LeNet-5** is the figure above: two convolutions, two poolings, two dense
layers, 60k parameters, handwritten digits for cheque processing. The
architecture was right; the data and the hardware were not.

**AlexNet** is LeNet made 1000× bigger, trained on ImageNet across two GPUs:
top-5 error 15.3% against 26.2% for the best hand-engineered entry. What
mattered was ReLU instead of tanh, dropout, aggressive augmentation, and enough
compute. The fourteen years between them were spent waiting for ImageNet and
CUDA.

---

## VGG (2014): depth from 3×3 stacks

VGG replaced AlexNet's 11×11 and 5×5 kernels with nothing but 3×3, stacked.

| | One 5×5 layer | Two 3×3 layers |
|---|---|---|
| Receptive field | 5×5 | 5×5 |
| Weights per channel pair | 25 | 18 |
| Nonlinearities | 1 | 2 |

Strictly better on both axes. Every architecture since uses small kernels
repeatedly. VGG also showed the limit: at 19 layers it stopped improving, and
plain networks past that got **worse on training error** — not a generalization
problem, an optimization one.

---

## ResNet (2015): the skip connection

$$
y = F(x) + x
$$

Instead of learning the mapping, learn the *residual* and add the input back.

Two consequences. The identity path gives the gradient a route to every earlier
layer with derivative 1, so it neither vanishes nor explodes through depth. And
a block that has nothing useful to add can drive $F$ to zero and become the
identity — so adding layers can no longer make the network worse.

152 layers trained where 20 plain layers had stalled. This is the single most
important architectural idea in deep learning, and it is three characters.

---

## The residual block

```python
def forward(self, x):
    out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
    return F.relu(out + self.shortcut(x))
```

`self.shortcut` is `nn.Identity()` when the shapes match, and a 1×1 stride-2
convolution when the block changes resolution or channel count — the addition
requires identical shapes, and that projection is the only reason the block is
ever more than two convolutions.

Skip connections appear again in U-Net (Session 6) and in every transformer
block (Session 7).

---

## EfficientNet (2019) and ConvNeXt (2022)

**EfficientNet** asked how to spend a compute budget: scale depth, width and
input resolution together by a fixed ratio rather than one at a time. Same
ImageNet accuracy at 5–10× fewer FLOPs; its B0 variant is still the right choice
for phones and embedded targets.

**ConvNeXt** took a ResNet-50 and applied transformer-era design decisions one at
a time — larger kernels, fewer activations, LayerNorm, an inverted bottleneck —
and matched a Vision Transformer of the same size. The gap was never convolution
versus attention, it was training recipes.

---

## Vision Transformer (2020), part 1

Cut the image into 16×16 patches, flatten each to a vector, project it linearly,
add a position embedding, and feed the sequence to a standard transformer
encoder. A 224×224 image becomes 196 tokens. Session 7 covers the encoder
itself.

No convolution, and therefore none of convolution's built-in assumptions: no
locality, no translation equivariance. Every patch can attend to every other
patch from layer 1.

---

## Vision Transformer, part 2: the price

Removing an inductive bias means the data has to supply it. The original ViT
**underperformed** ResNet on ImageNet-1k and only overtook it when pretrained on
300 million images.

| | CNN | ViT |
|---|---|---|
| Built-in prior | locality, equivariance | none |
| Data appetite | moderate | high, or heavy augmentation |
| Receptive field at layer 1 | 3×3 | global |
| Small-data behaviour | robust | needs a strong pretrained checkpoint |

Later recipes — DeiT, distillation, masked pretraining — closed most of the gap.
You will almost never train either from scratch, which makes the real question
"which pretrained checkpoint", not "which architecture".

---

## What to pick in 2026

| Situation | Choice |
|---|---|
| Default, a few thousand labelled images | pretrained **ResNet-50** or **ConvNeXt-T** |
| You need every point of accuracy and have a GPU | **ViT-B** or a self-supervised checkpoint |
| Phone, embedded, latency budget | **EfficientNet-B0**, **MobileNetV3** |
| Fewer than 500 images | frozen features + logistic regression |
| Genuinely novel input (13-band satellite, 3D medical) | a small CNN from scratch |

> Do not design an architecture. Pick a pretrained one, and spend your time on
> the data.

Architecture is the least valuable knob on the board for applied work. The next
two lessons cover the ones that matter.

---

## Check yourself

1. A 224×224 map with 256 channels goes through three 2×2 max-pools at stride 2,
   then `AdaptiveAvgPool2d(1)`. What is the spatial size after each pool, and
   what is the final shape?

   **Answer.** 112, then 56, then 28 — the pooling formula with `P = 0` halves
   it each time — and the head returns `(B, 256, 1, 1)`, whatever the input size
   was.

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   x = torch.randn(1, 256, 224, 224)
   pool, sizes = nn.MaxPool2d(2), []
   for _ in range(3):
       x = pool(x)
       sizes.append(x.shape[-1])
   print(sizes)                              # -> [112, 56, 28]
   print(nn.AdaptiveAvgPool2d(1)(x).shape)   # -> torch.Size([1, 256, 1, 1])
   ```

3. The original ViT lost to a ResNet on ImageNet-1k and overtook it only after
   pretraining on 300 million images. What did it give up, and who paid for it?

   **Answer.** Convolution's built-in priors: locality and translation
   equivariance. Remove an inductive bias and the data has to supply it.
