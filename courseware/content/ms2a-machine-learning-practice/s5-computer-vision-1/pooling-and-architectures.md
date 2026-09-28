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

## First Deep Learning success


![1586791179489.png](assets/cv/1586791179489.png)

LeNet-5 (Yann LeCun 1998) deployed by the US Postal Service to automatically read handwritten ZIP codes on mail.


---

## History



![4a821f3c-2e41-4718-a29e-89971228d4c1_1106x631.gif](assets/cv/4a821f3c-2e41-4718-a29e-89971228d4c1_1106x631.gif)



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

## Vision Transformer (2020)
New state of the art. It will be discussed later.

---


## A simple CNN in PyTorch

CIFAR-10: 32×32 RGB images, 10 classes. Three conv blocks, then a small
classifier.

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (64, 8, 8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (128, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (128, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 128·4·4 = 2048
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),                  # logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN()
x = torch.randn(8, 3, 32, 32)
print(model(x).shape)        # torch.Size([8, 10])
```

Pattern: **spatial size goes down, channels go up**. Each block halves
H and W and doubles C.

---

## Shapes and parameters

| Layer | Output shape | Parameters |
|---|---|---|
| Input | (3, 32, 32) | — |
| `Conv2d(3, 32, 3)` + pool | (32, 16, 16) | 32 · (3·9 + 1) = 896 |
| `Conv2d(32, 64, 3)` + pool | (64, 8, 8) | 64 · (32·9 + 1) = 18,496 |
| `Conv2d(64, 128, 3)` + pool | (128, 4, 4) | 128 · (64·9 + 1) = 73,856 |
| `Linear(2048, 128)` | (128,) | 2048 · 128 + 128 = 262,272 |
| `Linear(128, 10)` | (10,) | 128 · 10 + 10 = 1,290 |
| **Total** | | **356,810** |

```python
sum(p.numel() for p in model.parameters())   # 356810
```

The three convolutions hold 26% of the parameters; the first dense layer
alone holds 73%. The dense part is where the parameters are, the conv part
is where the computation is.

---

## Training

```python
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


train_ds = CIFAR10("data", train=True,  download=True)
test_ds  = CIFAR10("data", train=False, download=True)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(15):
    # --- train ---
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    # --- evaluate ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()

    print(f"epoch {epoch+1:2d}  "
          f"loss {total_loss / len(train_ds):.3f}  "
          f"test acc {correct / len(test_ds):.3f}")
```
