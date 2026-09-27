# Convolution

Convolution is not a trick for making networks smaller. It is a statement about
images: features are local, and a feature means the same thing wherever it
appears.

<!-- notes: 40 minutes, the core of the session. Do the 2x2 worked example on
the board by hand before showing any PyTorch. Budget 10 minutes for the
output-size formula and make them compute two cases out loud. -->

## An MLP is the wrong model for an image



![cnn.png](assets/cv/cnn.png)


question: what the number of parameters needed by neurons in the first layer of an MLP
for a color image of size 1000×1000 ?

Estimate the number of parameter of a small mlp for this input.




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

## The operation

![Sliding a 2x2 kernel over a 3x3 input](assets/cv/conv-worked-example.png)

Place a small matrix of weights — the **kernel** — over a patch of the input,
multiply element-wise, sum, write one number. Slide, repeat. The grid of results
is a **feature map**. That is the entire operation: no pixel is treated
specially, and the same weights are used at every position.

---

## The worked example

Input, and a 2×2 kernel:

```text
4 7 1        1 -1
2 6 9   *    0  2
8 5 3
```

Top-left position: `4·1 + 7·(-1) + 2·0 + 6·2 = 9`. Slide one column right:
`7·1 + 1·(-1) + 6·0 + 9·2 = 24`. Second row: `2 - 6 + 0 + 10 = 6`, then
`6 - 9 + 0 + 6 = 3`.

```text
 9  24
 6   3
```

A 3×3 input and a 2×2 kernel give a 2×2 output — the kernel needs to fit
entirely inside, so the output shrinks.

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



## An MLP is the wrong model for an image

![Fully connected versus locally connected](assets/cv/mlpvscnn.png)

A 1000×1000 image into one million fully connected hidden units is $10^{12}$
weights. Restrict each unit to a 10×10 patch and it is $10^{8}$ — four orders of
magnitude, before any weight sharing.

Share the same 10×10 patch weights across all positions and it is 100 weights
per filter. That is the whole argument, twice.

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

## The output-size formula

$$
O = \left\lfloor \frac{N + 2P - D(K-1) - 1}{S} \right\rfloor + 1
$$

$N$ input size, $K$ kernel size, $P$ padding, $S$ stride, $D$ dilation. With
$D = 1$ this is the familiar $(N - K + 2P)/S + 1$.

| $N$ | $K$ | $S$ | $P$ | $O$ |
|---|---|---|---|---|
| 6 | 2 | 1 | 0 | 5 |
| 6 | 2 | 2 | 0 | 3 |
| 6 | 2 | 2 | 1 | 4 |
| 32 | 3 | 1 | 1 | 32 |
| 224 | 7 | 2 | 3 | 112 |

Compute this on paper before you write the model. A shape mismatch at the first
linear layer is the most common error in this session, and it is arithmetic, not
a bug.

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

## Complete conv - exemple 1


![conv_c1.gif](assets/cv/conv_c1.gif)

Parameter	Value
Kernel Size	3 × 3
Input Size	7 × 7
Channels (in/out)	1/4
Stride	1
Padding	0
Output Size	5 × 5 (⌊(7 - 3 + 0)/1⌋ + 1 = 5)
Parameters	(9 weights × 4 filters) + 4 biases = 40 parameters
nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0)

---

## Complete conv - exemple 2

![conv_c2.gif](assets/cv/conv_c2.gif)
Kernel Size	3 × 3
Input Size	7 × 7
Channels (in/out)	3/4
Stride	1
Padding	0
Output Size	5 × 5 (⌊(7 - 3 + 0)/1⌋ + 1 = 5)
Parameters	(9 weights × 3 channels × 4 filters) + 4 biases = 112 parameters
nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)

---



## Counting parameters

$$
P_{conv} = C_{out} \left( C_{in} K^2 + 1 \right)
$$

One bias per output channel, added at every spatial position.

| Layer | Parameters |
|---|---|
| `Conv2d(3, 64, 3)` | 64 · (3 · 9 + 1) = **1,792** |
| `Conv2d(64, 128, 3)` | 128 · (64 · 9 + 1) = **73,856** |
| `Linear(150528, 1000)` on a flat 224×224×3 | **150,529,000** |

The convolution is independent of the input resolution. The dense layer is
proportional to it — which is why the same 224-pixel model applied to a 1024
pixel image would need a hundred times more weights in that one layer.


---

## In PyTorch

```python
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
x = torch.randn(8, 3, 224, 224)
print(conv(x).shape)          # torch.Size([8, 64, 224, 224])
print(conv.weight.shape)      # torch.Size([64, 3, 3, 3])
```

`bias=False` whenever the next layer is a `BatchNorm2d` — the norm subtracts a
learned mean, so the convolution's bias is redundant.
