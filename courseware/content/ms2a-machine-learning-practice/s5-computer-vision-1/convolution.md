# Convolution

Convolution is not a trick for making networks smaller. It is a statement about
images: features are local, and a feature means the same thing wherever it
appears.

<!-- notes: 40 minutes, the core of the session. Do the 2x2 worked example on
the board by hand before showing any PyTorch. Budget 10 minutes for the
output-size formula and make them compute two cases out loud. -->

---

## Question: Is an MLP a good model for images?

![cnn.png](assets/cv/cnn.png)

Consider a color image of size 1000×1000 fed to a fully connected network.

1. How many parameters does **a single neuron** of the first layer need?
2. Estimate the total number of parameters of a **small** MLP on this input:
   one hidden layer of 1000 neurons, then 10 output classes.
3. Beyond the count, what does an MLP ignore about the structure of an image?

<!-- notes: let them compute before revealing. Ask for orders of magnitude,
not exact numbers. -->

---

## Answer

**1. One neuron of the first layer**

- Input size: 1000 × 1000 × 3 channels = **3 × 10⁶ values**
- Each neuron is connected to every input: 3 × 10⁶ weights + 1 bias
- → **≈ 3 million parameters per neuron**

**2. A small MLP (3M → 1000 → 10)**

| Layer | Computation | Parameters |
|---|---|---|
| Input → hidden | 3 × 10⁶ × 1000 + 1000 | ≈ 3 × 10⁹ |
| Hidden → output | 1000 × 10 + 10 | ≈ 10⁴ |
| **Total** | | **≈ 3 billion** |

In float32 (4 bytes), that is **≈ 12 GB just for the weights**, before
gradients and optimizer state. Almost all of it sits in the first layer.

---

## The other half of the argument

Flattening destroys adjacency. After `x.view(-1)`, the pixel above a given pixel
is 224 positions away and the network has no way to know it was ever a
neighbour.

Worse: a dense layer has no **translation equivariance**. Shift a digit three
pixels right and every input coordinate changes, so the MLP has to learn "the
digit 7, at each of 50,000 offsets" separately. It needs the data to cover every
offset. A convolution gets it for free — shift the input, and the output shifts
with it.


![img.png](assets/cv/img.png)


---



## Convolution is all you need

![Fully connected versus locally connected](assets/cv/mlpvscnn.png)

Share the same 10×10 patch weights across all positions and it is 100 weights
per filter.


---

## The operation

![Sliding a 2x2 kernel over a 3x3 input](assets/cv/conv-worked-example.png)

Place a small matrix of weights — the **kernel** — over a patch of the input,
multiply element-wise, sum, write one number. Slide, repeat. The grid of results
is a **feature map**. That is the entire operation: no pixel is treated
specially, and the same weights are used at every position.

---

## Question: Compute the convolution

Input image and 2×2 kernel:

```text
Input            Kernel
4 7 1            1 -1
2 6 9            0  2
8 5 3
```

1. Compute the output of this kernel on this image.
2. What is the size of the output?

<!-- notes: do it by hand on the board. Let them compute the first position,
then check together before they do the remaining three. -->

---

## Solution

**1. Slide the kernel over each 2×2 patch**

| Position | Patch | Computation | Result |
|---|---|---|---|
| Top-left | `4 7 / 2 6` | 4·1 + 7·(−1) + 2·0 + 6·2 | **9** |
| Top-right | `7 1 / 6 9` | 7·1 + 1·(−1) + 6·0 + 9·2 | **24** |
| Bottom-left | `2 6 / 8 5` | 2·1 + 6·(−1) + 8·0 + 5·2 | **6** |
| Bottom-right | `6 9 / 5 3` | 6·1 + 9·(−1) + 5·0 + 3·2 | **3** |

Output feature map:

```text
 9  24
 6   3
```

**2. Output size**

A 3×3 input and a 2×2 kernel give a **2×2 output**.

---

## A kernel is a filter

![A horizontal edge kernel applied to a step image](assets/cv/conv.png)

The kernel above is `[[-1,-1,-1], [0,0,0], [1,1,1]]` and the input is black on
top, white on the bottom. The output is near zero in the flat regions and large
exactly where the intensity changes.

The kernel does not "look for" an edge in any semantic sense. It computes a
weighted difference between the rows above and below a position, which is large
exactly when they differ.

---

## Kernels people designed by hand

| Kernel | Effect |
|---|---|
| Sobel-x `[[-1,0,1],[-2,0,2],[-1,0,1]]` | vertical edges |
| Sobel-y (its transpose) | horizontal edges |
| Box `1/9 · ones(3,3)` | blur |
| Gaussian | blur, weighted by distance |
| Sharpen `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | boost the centre against its neighbours |
| Laplacian `[[0,1,0],[1,-4,1],[0,1,0]]` | second derivative, edges in all directions |

Twenty years of computer vision was spent designing these by hand and deciding
which combination to use for which task.

![sobel.png](assets/cv/sobel.png)


---


## Kernels that are learned instead

![Feature hierarchy across CNN depth](assets/cv/cnn-learn.png)

A CNN puts the kernel weights in the parameter vector and lets gradient descent
choose them. What it converges to, reliably, on any natural-image task:

| Depth | What the filters respond to |
|---|---|
| Layer 1 | oriented edges, colour blobs — visibly Gabor-like |
| Middle | corners, textures, repeated motifs |
| Deep | object parts: wheels, eyes, text-like structure |
| Last | whole objects and scene configurations |

Layer 1 of a network trained on cats and layer 1 of a network trained on
satellite tiles look nearly identical. That fact is the reason transfer learning
works.


---
## Stride

![2x2 kernel, stride 1, no padding](assets/cv/conv_kern1.png)

Stride is how far the kernel moves between positions. A 6×6 input with a 2×2
kernel at stride 1 gives 5×5.

![2x2 kernel, stride 2, no padding](assets/cv/conv_kern2.png)

Stride 2 skips every other position and gives 3×3 — the same computation,
downsampled. Stride is the cheapest way to reduce spatial size, and modern
architectures use it in place of pooling.

---

## Padding

![2x2 kernel, stride 2, padding 1](assets/cv/conv_kern3.png)

Padding adds a border of zeros before the convolution. Without it every layer
shrinks the map, and a 20-layer 3×3 network cannot be built on a 32×32 input.
It also evens out the border: a corner pixel participates in one output and a
centre pixel in nine — padding reduces that asymmetry without removing it.

`padding=1` with a 3×3 kernel at stride 1 preserves the spatial size exactly,
which is why it is the default convolution of the last decade.

---


## Channels

A convolution kernel is not 2D. Its weight tensor has shape
`(C_out, C_in, K, K)`: each output channel has one full-depth filter that spans
**all** input channels and sums across them.

$$
G[c_{out}, i, j] = b[c_{out}] + \sum_{c=0}^{C_{in}-1} \sum_{k=0}^{K-1} \sum_{l=0}^{K-1} F[c,\, Si+k,\, Sj+l] \cdot W[c_{out}, c, k, l]
$$

So `Conv2d(3, 64, 3)` holds 64 filters, each 3×3×3. The output has 64 channels
regardless of how many the input had — channels are the layer's vocabulary size,
not a property of the image.

A 1×1 convolution has no spatial extent at all and does nothing but mix
channels. It is a per-position linear layer, and it is everywhere in modern
architectures.

---

## Complete conv — example 1

![conv_c1.gif](assets/cv/conv_c1.gif)

```python
nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0)
```

| Parameter | Value |
|---|---|
| Kernel size | 3 × 3 |
| Input size | 7 × 7 |
| Channels (in / out) | 1 / 4 |
| Stride | 1 |
| Padding | 0 |
| Output size | ⌊(7 − 3 + 0) / 1⌋ + 1 = **5 × 5** |
| Parameters | 4 · (1 · 9 + 1) = **40** |

---

## Complete conv — example 2

![conv_c2.gif](assets/cv/conv_c2.gif)

```python
nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)
```

| Parameter | Value |
|---|---|
| Kernel size | 3 × 3 |
| Input size | 7 × 7 |
| Channels (in / out) | 3 / 4 |
| Stride | 1 |
| Padding | 0 |
| Output size | ⌊(7 − 3 + 0) / 1⌋ + 1 = **5 × 5** |
| Parameters | 4 · (3 · 9 + 1) = **112** |

---

## Question: Output size and number of parameters

For each convolutional layer below (square inputs and kernels, $D = 1$, with
bias), compute:

1. the output size $O$
2. the number of learnable parameters

$C_{in}$ is the number of input channels, $C_{out}$ the number of filters.

| $N$ | $K$ | $S$ | $P$ | $C_{in}$ | $C_{out}$ | $O$ | Parameters |
|---|---|---|---|---|---|---|---|
| 6 | 2 | 1 | 0 | 1 | 1 | ? | ? |
| 6 | 2 | 2 | 0 | 3 | 8 | ? | ? |
| 6 | 2 | 2 | 1 | 8 | 16 | ? | ? |
| 32 | 3 | 1 | 1 | 3 | 64 | ? | ? |
| 224 | 7 | 2 | 3 | 3 | 64 | ? | ? |

3. Does the number of parameters depend on $N$? Compare with the MLP from
   the first question.

<!-- notes: 10 minutes. Make them compute two cases out loud (rows 2 and 5).
Hint if stuck: one filter covers all input channels, and each filter has
its own bias. -->

---

## The output-size formula

$$
O = \left\lfloor \frac{N + 2P - D(K-1) - 1}{S} \right\rfloor + 1
$$

$N$ input size, $K$ kernel size, $P$ padding, $S$ stride, $D$ dilation. With
$D = 1$ this is the familiar $(N - K + 2P)/S + 1$.

---

## Solution

**Parameter formula**

Each filter has $K \times K \times C_{in}$ weights plus one bias, and there
are $C_{out}$ filters:

$$
\text{params} = C_{out} \cdot (K^2 \cdot C_{in} + 1)
$$

**Results**

| $N$ | $K$ | $S$ | $P$ | $C_{in}$ | $C_{out}$ | $O$ | Parameters |
|---|---|---|---|---|---|---|---|
| 6 | 2 | 1 | 0 | 1 | 1 | $(6-2)/1+1 = $ **5** | $1 \cdot (4 \cdot 1 + 1) = $ **5** |
| 6 | 2 | 2 | 0 | 3 | 8 | $\lfloor 4/2 \rfloor+1 = $ **3** | $8 \cdot (4 \cdot 3 + 1) = $ **104** |
| 6 | 2 | 2 | 1 | 8 | 16 | $\lfloor 6/2 \rfloor+1 = $ **4** | $16 \cdot (4 \cdot 8 + 1) = $ **528** |
| 32 | 3 | 1 | 1 | 3 | 64 | $(32-3+2)/1+1 = $ **32** | $64 \cdot (9 \cdot 3 + 1) = $ **1 792** |
| 224 | 7 | 2 | 3 | 3 | 64 | $\lfloor 223/2 \rfloor+1 = $ **112** | $64 \cdot (49 \cdot 3 + 1) = $ **9 472** |

Notes:

- Row 4: $K = 3, P = 1, S = 1$ preserves the size ("same" padding).
- Row 5 is the first layer of ResNet: a 224×224 image becomes 112×112×64.
  ResNet actually drops the bias here (followed by BatchNorm): 9 408 parameters.
- The floor matters: in row 5, $223/2 = 111.5 \to 111$.


---

## CNN input in PyTorch

A batch of images is a 4D tensor of shape **`(N, C, H, W)`**:

| Dim | Meaning | Example |
|---|---|---|
| `N` | batch size | 8 |
| `C` | channels | 3 (RGB), 1 (grayscale) |
| `H` | height | 224 |
| `W` | width | 224 |

```python
x = torch.randn(8, 3, 224, 224)       # 8 RGB images of 224×224
```


---

## `nn.Conv2d` in PyTorch

```python
conv = nn.Conv2d(in_channels=3, out_channels=64,
                 kernel_size=3, stride=1, padding=1)

x = torch.randn(8, 3, 224, 224)
print(conv(x).shape)        # torch.Size([8, 64, 224, 224])
print(conv.weight.shape)    # torch.Size([64, 3, 3, 3])
print(conv.bias.shape)      # torch.Size([64])
```

- Input `(N, C_in, H, W)` → output `(N, C_out, H_out, W_out)`.
- `H_out`, `W_out` follow the output-size formula; here `padding=1` with a
  3×3 kernel keeps 224.
- `N` is untouched: the same filters are applied to every image of the batch.
- The weight tensor is `(C_out, C_in, K, K)`: one 3D filter per output
  channel, spanning **all** input channels.

---

## Counting parameters

Read directly from the weight and bias shapes:

$$
P_{conv} = C_{out} \left( C_{in} K^2 + 1 \right)
$$

One bias per output channel, shared across all spatial positions.

| Layer | Parameters |
|---|---|
| `Conv2d(3, 64, 3)` | 64 · (3 · 9 + 1) = **1,792** |
| `Conv2d(64, 128, 3)` | 128 · (64 · 9 + 1) = **73,856** |
| `Linear(150528, 1000)` on a flat 224×224×3 | 150,528 · 1000 + 1000 = **150,529,000** |

---


## Conv1d, Conv2d, Conv3d

Same operation, the kernel slides along 1, 2 or 3 spatial dimensions.
`N` and `C` are always the first two dims.

| Layer | Input shape | Weight shape | Typical data |
|---|---|---|---|
| `nn.Conv1d` | `(N, C, L)` | `(C_out, C_in, K)` | audio, time series, sequences |
| `nn.Conv2d` | `(N, C, H, W)` | `(C_out, C_in, K, K)` | images |
| `nn.Conv3d` | `(N, C, D, H, W)` | `(C_out, C_in, K, K, K)` | video, MRI / CT volumes |

![maxresdefault.jpg](assets/cv/maxresdefault.jpg)
