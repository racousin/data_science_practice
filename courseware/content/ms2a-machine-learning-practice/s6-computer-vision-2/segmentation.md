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

![IoU and Dice](assets/cv/dicevsiou.png)

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
