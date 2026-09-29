# Diffusion Models

Turning an image into noise is easy: add a little Gaussian noise, many times.
A diffusion model learns to undo one small step of that process. Generation
becomes a chain of easy denoising problems, each a plain regression, with no
encoder to match and no adversary to balance.

<!-- notes: 45 minutes, the longest lesson of the three. First half: the DDPM
equations (forward, closed form, loss, the two algorithms). Second half: the
networks, where Session 7 pays off: cross-attention for text, then the
Diffusion Transformer, which is a ViT on latent patches. The two dimensional
flow tables are the slides to do slowly. -->

---

## The forward process

![Forward noising and learnt reverse process](assets/cv/diffusion-process.png)

Start from a training image $x_0 \in \mathbb{R}^{3 \times H \times W}$, pixels
in $[-1, 1]$. For $t = 1, \dots, T$, shrink the image slightly and add noise:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\big)
$$

The schedule $\beta_1, \dots, \beta_T$ is **fixed**, not learnt. DDPM uses
$T = 1000$ and $\beta_t$ linear from $10^{-4}$ to $0.02$. The factor
$\sqrt{1 - \beta_t}$ keeps the variance at 1: if $\mathrm{Var}(x_{t-1}) = 1$,
then $\mathrm{Var}(x_t) = (1 - \beta_t) + \beta_t = 1$.

Nothing in this process has parameters. It produces the training pairs, and
it defines the end point: $x_T \approx \mathcal{N}(0, I)$, a distribution we
can sample from.

---

## Jumping to any step

With $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$, a
composition of Gaussian steps is one Gaussian step:

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\big)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

For the linear schedule:

| $t$ | 1 | 100 | 250 | 500 | 750 | 1000 |
|---|---|---|---|---|---|---|
| $\bar{\alpha}_t$ | 0.9999 | 0.897 | 0.524 | 0.0786 | 0.0034 | $4.0 \times 10^{-5}$ |
| signal $\sqrt{\bar{\alpha}_t}$ | 1.000 | 0.947 | 0.724 | 0.280 | 0.058 | 0.006 |
| noise $\sqrt{1 - \bar{\alpha}_t}$ | 0.010 | 0.321 | 0.690 | 0.960 | 0.998 | 1.000 |

$x_t$ for any $t$ costs one line, with no loop over the previous steps. At
$t = 1000$ the image contributes 0.6% of the signal.

---

## The reverse process

For small $\beta_t$, the reverse step $q(x_{t-1} \mid x_t)$ is close to a
Gaussian. Learn it:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 I\big), \qquad \sigma_t^2 = \beta_t
$$

Instead of predicting the mean directly, the network predicts **the noise**
$\epsilon$ contained in $x_t$. Inverting the closed form gives an estimate of
$x_0$, and from it the mean:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)
$$

| | Input | Output | Learnt |
|---|---|---|---|
| $\epsilon_\theta$ | $x_t \in \mathbb{R}^{3 \times H \times W}$, $t \in \{1..T\}$ | $\hat{\epsilon} \in \mathbb{R}^{3 \times H \times W}$ | $\theta$ |

One network, shared across all $T$ steps: $t$ is an input, not a separate
model. Output and input have the same shape, like a segmentation network.

---

## Training

The ELBO of the VAE lesson, written for this chain of $T$ latent variables,
reduces (up to per-step weights, which DDPM drops) to a regression on the
noise:

$$
\mathcal{L}_{simple}(\theta) = \mathbb{E}_{x_0,\; t \sim \mathcal{U}\{1..T\},\; \epsilon \sim \mathcal{N}(0, I)} \Big[ \big\| \epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\; t\big) \big\|^2 \Big]
$$

**Algorithm 1 (training).** Repeat: take $x_0$ from the data, draw $t$
uniformly and $\epsilon \sim \mathcal{N}(0, I)$, build $x_t$ in closed form,
take a gradient step on $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$.

```python
t = torch.randint(1, T + 1, (B,))
eps = torch.randn_like(x0)
ab = alpha_bar[t - 1].view(B, 1, 1, 1)          # (B,) -> broadcast over (C, H, W)
xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
loss = F.mse_loss(eps_model(xt, t), eps)
```

Each example is an independent supervised regression with a fixed target. The
loss decreases and is interpretable, unlike the two losses of a GAN.

---

## Sampling

**Algorithm 2 (DDPM sampling).** Start from $x_T \sim \mathcal{N}(0, I)$. For
$t = T, \dots, 1$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right) + \sigma_t\, z, \qquad z \sim \mathcal{N}(0, I) \text{ if } t > 1, \text{ else } z = 0
$$

```python
x = torch.randn(B, 3, H, W)
for t in range(T, 0, -1):
    z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
    eps = eps_model(x, t)
    x = (x - betas[t-1] / (1 - alpha_bar[t-1]).sqrt() * eps) / alphas[t-1].sqrt() + betas[t-1].sqrt() * z
```

<!-- placeholder: image to add (denoising trajectory: one sample shown at t = 1000, 750, 500, 250, 0, coarse layout first then details; e.g. Ho, Jain, Abbeel 2020, Fig. 6) -->

The cost: $T = 1000$ network evaluations per image, against one for a GAN or a
VAE decoder. **DDIM** (Song et al., 2021) reuses the same trained
$\epsilon_\theta$ with a deterministic update on a subsequence of ~50 steps:
no retraining, 20× fewer evaluations.

---

## The denoiser: a U-Net with a time embedding

![U-Net](assets/cv/unet-architecture.png)

$\epsilon_\theta$ maps an image to an image of the same shape: the U-Net of the
segmentation lesson, with a 3-channel regression output instead of a mask. The
skip connections carry the fine detail that the noise prediction needs at full
resolution.

The timestep enters every residual block. It is encoded like a position in
Session 7, then projected per block:

$$
e(t) = \big[\sin(t\,\omega_i),\; \cos(t\,\omega_i)\big]_{i} \in \mathbb{R}^{d_t}, \qquad
h \leftarrow h + W_b\, \mathrm{MLP}\big(e(t)\big) \quad (W_b\, \mathrm{MLP}(e(t)) \in \mathbb{R}^{C_b},\; \text{broadcast over } H \times W)
$$

| Tensor | Shape |
|---|---|
| input $x_t$ | $3 \times H \times W$ |
| $t$ → $e(t)$ → MLP | scalar → $d_t$ → $d_t$ |
| feature map $h$ in block $b$ | $C_b \times H_b \times W_b$ |
| time bias added to $h$ | $C_b$, one value per channel |
| output $\hat{\epsilon}$ | $3 \times H \times W$ |

The same weights must denoise at $t = 10$ (remove faint grain) and $t = 990$
(invent a layout from noise); the per-channel time bias tells each block which
of the two tasks it is doing.

---

## Text conditioning by cross-attention

![Cross-attention: queries from one sequence, keys and values from another](assets/nlp/cross_attention.png)

To generate from a prompt $c$, the U-Net becomes $\epsilon_\theta(x_t, t, c)$.
At the lower resolutions, a transformer block is inserted after the
convolutions: flatten the feature map into tokens, then self-attention, then
**cross-attention** to the text, then the FFN. This is the decoder block of
Session 7 without the causal mask; the image plays the target, the text plays
the encoder output.

$$
Q = H W_Q,\quad K = C W_K,\quad V = C W_V,\qquad
\mathrm{CrossAttn}(H, C) = \mathrm{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

With Stable Diffusion v1 at the $64 \times 64$ level ($C_b = 320$) and a CLIP
text encoder (77 tokens, width 768):

| Tensor | Shape |
|---|---|
| feature map $h$, flattened to tokens $H$ | $320 \times 64 \times 64 \to 4096 \times 320$ |
| text tokens $C$ (frozen text encoder) | $77 \times 768$ |
| $W_Q$ / $W_K$, $W_V$ | $320 \times 320$ / $768 \times 320$ |
| $Q$ / $K$, $V$ | $4096 \times 320$ / $77 \times 320$ |
| attention matrix, per head (8 heads, $d_k = 40$) | $4096 \times 77$ |
| output, reshaped back | $4096 \times 320 \to 320 \times 64 \times 64$ |

Each spatial position chooses which words to read. "A red cube on a blue
table" can put `red` on the cube pixels and `blue` on the table pixels.

---

## Classifier-free guidance

Train one network for both tasks: replace $c$ by the empty prompt $\varnothing$
in 10% of the training examples. At sampling, evaluate it twice per step and
extrapolate from the unconditional toward the conditional prediction:

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w\,\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big)
$$

$w = 0$: unconditional. $w = 1$: plain conditional model. $w > 1$: the
direction "what the prompt adds" is amplified; Stable Diffusion's default is
$w = 7.5$. Larger $w$ gives images that match the prompt more closely and are
less diverse; too large, colours saturate and realism drops.

<!-- placeholder: image to add (same prompt and seed at guidance scales w = 1, 3, 7.5, 15; e.g. Ho & Salimans 2022, Fig. 1, or a Stable Diffusion sweep) -->

The two evaluations are batched: $2B$ inputs per step. $\tilde{\epsilon}$
replaces $\epsilon_\theta$ in Algorithm 2.

---

## Latent diffusion: Stable Diffusion

<!-- placeholder: image to add (Latent Diffusion architecture: pixel space encoder/decoder, diffusion process in latent space, denoising U-Net with cross-attention, conditioning encoder; Rombach et al. 2022, Fig. 3) -->

A U-Net on $3 \times 512 \times 512$ pixels, 50 to 1000 times per image, is
expensive. Latent diffusion runs the whole process in the latent space of the
VAE of the first lesson, and decodes once at the end.

| # | Component | Input → output | Trained |
|---|---|---|---|
| 1 | tokenizer | prompt → 77 token ids | fixed |
| 2 | CLIP text encoder | $77 \to 77 \times 768$ | pretrained, frozen |
| 3 | noise $z_T \sim \mathcal{N}(0, I)$ | → $4 \times 64 \times 64$ | — |
| 4 | U-Net $\epsilon_\theta(z_t, t, c)$, ~860M | $4 {\times} 64 {\times} 64$ → $4 {\times} 64 {\times} 64$ | **this is the diffusion model** |
|   | inside: channels × resolution | $320{\times}64^2 \to 640{\times}32^2 \to 1280{\times}16^2 \to 1280{\times}8^2$ | |
| 5 | Algorithm 2 with guidance, ~50 steps | $z_T \to z_0$ | — |
| 6 | VAE decoder $\mathcal{D}$ | $4 \times 64 \times 64 \to 3 \times 512 \times 512$ | pretrained, frozen |

Training is the loss of this lesson with $x_0$ replaced by $z_0 =
\mathcal{E}(x_0)$ and the prompt fed through cross-attention. Every step runs
on $16{,}384$ values instead of $786{,}432$: 48× fewer. The VAE handles the
perceptual detail once; the diffusion model spends its capacity on the layout
and the content.

---

## The Diffusion Transformer (DiT)

<!-- placeholder: image to add (DiT block with adaLN-Zero: patchify, N DiT blocks conditioned on timestep and class, linear decoder; Peebles & Xie 2023, Fig. 3) -->

Replace the U-Net by a ViT. The noisy latent is cut into patches, each patch is
a token, and a stack of transformer blocks predicts the noise per token.

For a $256 \times 256$ image, the VAE latent is $4 \times 32 \times 32$. With
patch size $p = 2$:

| # | Step | Shape |
|---|---|---|
| 1 | noisy latent $z_t$ | $4 \times 32 \times 32$ |
| 2 | patchify: $(32/2)^2$ patches of $2 \times 2 \times 4$ | $256 \times 16$ |
| 3 | linear embedding + 2D sin-cos positions | $256 \times d$ ($d = 1152$ for DiT-XL) |
| 4 | 28 DiT blocks (self-attention $256 \times 256$ per head, FFN) | $256 \times d$ |
| 5 | final LN + linear: $p^2 \cdot 2C$ values per token | $256 \times 32$ |
| 6 | unpatchify, split | $8 \times 32 \times 32$ → noise $\hat{\epsilon}$ ($4$) + variance ($4$) |

**adaLN-Zero.** The condition is one vector, $c = \mathrm{emb}(t) +
\mathrm{emb}(y) \in \mathbb{R}^{d}$ ($y$ the class). An MLP maps it to six
vectors of size $d$ per block, which replace the learnt $\gamma, \beta$ of
layer norm and gate each residual branch:

$$
h \leftarrow h + \alpha_1 \odot \mathrm{Attn}\big(\gamma_1 \odot \mathrm{LN}(h) + \beta_1\big), \qquad
h \leftarrow h + \alpha_2 \odot \mathrm{FFN}\big(\gamma_2 \odot \mathrm{LN}(h) + \beta_2\big)
$$

$\alpha$ is initialised to zero: every block starts as the identity. DiT-XL/2
(675M parameters) reaches FID 2.27 on ImageNet $256 \times 256$ with guidance.
Recent text-to-image models (Stable Diffusion 3, FLUX) follow this design.

---

## Three answers

| | VAE | GAN | Diffusion |
|---|---|---|---|
| learnt | encoder $q_\phi(z \mid x)$, decoder $p_\theta(x \mid z)$ | generator $G_\theta$, discriminator $D_\psi$ | denoiser $\epsilon_\theta(x_t, t, c)$ |
| loss | ELBO: reconstruction + KL | minimax, learnt critic | $\| \epsilon - \epsilon_\theta \|^2$ |
| sampling | 1 decoder pass | 1 generator pass | 50–1000 network passes |
| weakness | blurry (mean of $x \mid z$) | unstable, mode collapse | slow sampling |
| role today | the latent space of diffusion | adversarial loss term | text-to-image, video |

A modern text-to-image system uses all three: a VAE (trained with a GAN loss)
defines the latent space, a diffusion U-Net or transformer generates in it, and
a transformer text encoder conditions it through cross-attention.

---

## Exercise (1/2): the forward process

A toy schedule with $T = 4$: $\beta = (0.1,\; 0.2,\; 0.3,\; 0.4)$.

1. Compute $\alpha_t$ and $\bar{\alpha}_t$ for $t = 1, \dots, 4$.
2. Write $x_4$ as a function of $x_0$ and $\epsilon$. What fraction of the
   standard deviation of $x_4$ comes from the image?
3. With the DDPM linear schedule, $\bar{\alpha}_{1000} = 4.0 \times 10^{-5}$.
   Why must $\bar{\alpha}_T$ be close to 0 for Algorithm 2 to work?

<!-- notes: 15 minutes for both parts. Question 3: sampling starts from
N(0, I); if x_T still contains the image, the model is asked at t = T to
denoise inputs it never saw during training. -->

---

## Solution (1/2)

**1.**

| $t$ | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| $\alpha_t$ | 0.9 | 0.8 | 0.7 | 0.6 |
| $\bar{\alpha}_t$ | 0.9 | 0.72 | 0.504 | 0.3024 |

**2.** $x_4 = \sqrt{0.3024}\, x_0 + \sqrt{0.6976}\, \epsilon = 0.550\, x_0 +
0.835\, \epsilon$. The image carries 0.550 of a unit standard deviation: after
4 steps it is still clearly visible. This schedule is far too short.

**3.** Sampling starts from $x_T \sim \mathcal{N}(0, I)$, which contains no
image. Training at $t = T$ must see the same distribution:
$q(x_T \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_T} x_0, (1 - \bar{\alpha}_T)I)
\approx \mathcal{N}(0, I)$ requires $\bar{\alpha}_T \approx 0$. At $4.0 \times
10^{-5}$, the image coefficient is $0.006$.

---

## Exercise (2/2): shapes

**A. Stable Diffusion cross-attention.** At the $32 \times 32$ level of the
U-Net, $C_b = 640$, 8 heads. The prompt is encoded to $77 \times 768$.

1. Number of image tokens, shapes of $Q$, $K$, $V$, and of one head's
   cross-attention matrix.
2. Shape of one head's **self**-attention matrix at the same level. Which
   attention is more expensive?
3. 50 sampling steps with classifier-free guidance: how many U-Net
   evaluations per image?

**B. DiT.** Latent $4 \times 32 \times 32$.

4. Number of tokens and values per patch for $p = 2, 4, 8$.
5. Self-attention matrix size for $p = 2$ and $p = 4$. Which is more expensive,
   and why does DiT-XL/2 still use $p = 2$?

---

## Solution (2/2)

**A1.** $32 \times 32 = 1024$ tokens. $Q$: $1024 \times 640$. $K, V$: $77 \times
640$ (via $W_K, W_V \in \mathbb{R}^{768 \times 640}$). Per head, $d_k = 80$:
the cross-attention matrix is $1024 \times 77$.

**A2.** Self-attention: $1024 \times 1024$ per head, about 13× larger than
$1024 \times 77$. The text is short; the image is long. At the $64 \times 64$
level it is $4096 \times 4096 \approx 16.8$M entries per head.

**A3.** $2 \times 50 = 100$ (conditional and unconditional at each step).

**B4.**

| $p$ | tokens $(32/p)^2$ | values per patch $p^2 \cdot 4$ |
|---|---|---|
| 2 | 256 | 16 |
| 4 | 64 | 64 |
| 8 | 16 | 256 |

**B5.** $256 \times 256 = 65{,}536$ against $64 \times 64 = 4{,}096$ entries
per head: $p = 2$ costs 16× more in attention. It is used because smaller
patches give more tokens and more compute per image, and in the DiT paper FID
improves steadily as $p$ decreases.
