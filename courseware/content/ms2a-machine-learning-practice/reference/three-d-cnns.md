# 3D CNNs

Reference only — not covered in class. Volumetric and video data are a
specialised modality with no claim on ten core sessions. Read it if your
project involves CT/MRI scans, video, or LiDAR, and you need to decide whether
a 3D convolution is worth its cost. Session 5 covers the 2D convolution this
extends.

<!-- notes: Self-study, linked from Sessions 5 and 6. Never lectured. -->

---

## The operation

![2D versus 3D convolution](assets/cv/2d-vs-3d-convolution.png)

A 3D convolution slides a kernel over three axes instead of two:

$$
z_{i,j,t,k} = b_k + \sum_{c} \sum_{p,q,\tau} w_{p,q,\tau,c,k} \, a_{i+p,\,j+q,\,t+\tau,\,c}
$$

The third axis is depth for a scan, time for a video. That is the only change;
padding, stride, pooling and backpropagation carry over from 2D unmodified.

---

## Shapes

PyTorch expects $(B, C, D, H, W)$ for `Conv3d`, where $D$ is depth or frame
count. A batch of 4 clips of 16 RGB frames at $112 \times 112$ is a
`(4, 3, 16, 112, 112)` tensor.

```python
conv = nn.Conv3d(3, 64, kernel_size=(3, 3, 3),
                 stride=(1, 2, 2), padding=(1, 1, 1))
out = conv(torch.randn(4, 3, 16, 112, 112))   # (4, 64, 16, 56, 56)
```

The asymmetric stride is the usual pattern: downsample space aggressively, keep
the depth axis intact for a few layers. Sixteen frames do not survive four
halvings.

---

## The cost

Parameters scale linearly with the kernel's third dimension:

$$
P_{3D} = K_t K_h K_w C_{in} C_{out} = K_t \cdot P_{2D}
$$

A $3 \times 3$ kernel with 64 in and 64 out is 36,864 weights; the $3 \times 3
\times 3$ version is 110,592. Three times more — annoying, not fatal.

Activations are what actually kills you, because they scale with the *data's*
depth, not the kernel's:

$$
A = B \cdot C \cdot D \cdot H \cdot W
$$

One feature map of one $256^3$ CT volume at 64 channels in float32 is 4.3 GB.
Not the model — one tensor, for one patient. This is why 3D pipelines run at
batch size 1 or 2 and lean on patch extraction, mixed precision and gradient
accumulation.

---

## The two real use cases

| | Medical volumes | Video |
|---|---|---|
| Shape | $(C, D, H, W)$, $C$ = modality | $(C, T, H, W)$, $C$ = RGB |
| Third axis | physically isotropic-ish | not a spatial axis at all |
| Typical size | $256^3$, one study | 16–64 frames sampled from minutes |
| Labels | scarce, expert-annotated | plentiful, weak |
| Metrics | 3D Dice, volumetric IoU, Hausdorff | top-1, temporal IoU, mAP |

The asymmetry matters. In a CT volume the depth axis is another spatial
dimension and a symmetric kernel is principled. In a video the time axis has
different statistics, different resolution and different semantics from height
and width — a symmetric $3 \times 3 \times 3$ kernel is a modelling assumption
that is usually wrong.

---

## The cheaper alternatives, which usually win

**2D per slice, then aggregate.** Run a pretrained 2D backbone on each slice,
then pool or run a small sequence model over the per-slice features. You
inherit ImageNet weights, which no 3D architecture gets for free, and the
memory cost is one slice at a time.

**(2+1)D factorisation.** Replace one $K_t \times K_h \times K_w$ convolution
with a spatial $1 \times K_h \times K_w$ followed by a temporal
$K_t \times 1 \times 1$:

$$
P_{(2+1)D} = K_h K_w C_{in} C_m + K_t C_m C_{out}
$$

Fewer parameters for the same receptive field, plus an extra non-linearity
between the two halves. It matches or beats full 3D on video benchmarks.

**Inflation (I3D).** Initialise a 3D kernel by copying a pretrained 2D kernel
$K_t$ times and dividing by $K_t$. The inflated network reproduces the 2D one's
behaviour on a static clip, so you start from transfer learning, not noise.

---

## Video specifically

Frame sampling is a bigger lever than architecture. A 10-second clip at 30 fps
is 300 frames; models see 16 or 32. How you choose them — uniform, random,
segment-based — changes accuracy more than swapping C3D for a (2+1)D ResNet.

**Two-stream** networks run one branch on RGB frames and one on precomputed
optical flow, then fuse; the flow branch carries motion information a 3D kernel
would otherwise have to learn. **Video transformers** (TimeSformer, ViViT,
Video Swin) tokenise space-time patches and attend over them, usually with
factorised space-then-time attention for the same reason (2+1)D works — the
current default at scale, and they need more data than a 3D CNN, not less.

---

## When 3D is genuinely required

Reach for a true 3D convolution when **the label depends on structure across
the third axis that no single slice contains**: a vessel or tumour boundary
only coherent in 3D, actions that differ only in temporal order (opening versus
closing a door), any task where the expert annotator scrolls rather than judges
one frame.

If a competent annotator can label from a single slice, a 2D model plus
aggregation will match a 3D model at a fraction of the cost. Test that first —
it is one afternoon, and it is the baseline your 3D result must beat to justify
itself.

**Failure mode.** Choosing 3D by default, then fitting a batch size of 1, then
training with batch normalisation. Batch norm on a batch of one estimates
statistics from a single volume and the run diverges or, worse, trains and then
collapses at evaluation. Use group norm or instance norm in 3D pipelines.

---

## Check yourself

1. Moving from a $3\times3$ to a $3\times3\times3$ kernel roughly triples the
   parameter count. Why is that not the number that decides whether you can
   train the model?

   **Answer.** Because activations, not parameters, are what run you out of
   memory: they scale with the *data's* depth, $A = B \cdot C \cdot D \cdot H
   \cdot W$. One 64-channel feature map of a single $256^3$ CT volume in
   float32 is 4.3 GB — one tensor, for one patient.

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   conv = nn.Conv3d(3, 64, kernel_size=(3, 3, 3),
                    stride=(1, 2, 2), padding=(1, 1, 1))
   print(tuple(conv(torch.randn(4, 3, 16, 112, 112)).shape))   # -> (4, 64, 16, 56, 56)
   print(sum(p.numel() for p in nn.Conv2d(64, 64, 3, bias=False).parameters()),
         sum(p.numel() for p in nn.Conv3d(64, 64, 3, bias=False).parameters()))
                                                               # -> 36864 110592
   ```

   **Answer.** The asymmetric stride halves height and width and leaves the 16
   frames intact — sixteen frames do not survive four halvings. And 110592 /
   36864 = 3 = $K_t$, exactly the linear scaling the lesson states.

3. Your 3D segmentation run fits only at batch size 1, and it diverges — or
   trains and then collapses at evaluation. Which layer is responsible and what
   do you replace it with?

   **Answer.** Batch normalisation: on a batch of one it estimates its
   statistics from a single volume. Use group norm or instance norm in 3D
   pipelines.
