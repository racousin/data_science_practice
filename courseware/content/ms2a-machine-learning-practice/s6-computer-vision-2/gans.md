# GANs

The VAE writes down a likelihood and pays for it with blurry samples. A GAN
writes down no likelihood at all. A second network, the discriminator, learns
to tell real images from generated ones, and the generator is trained to fool
it. The loss is learnt, and it does not average.

<!-- notes: 25 minutes. The optimal-discriminator derivation is short and is the
only theory in the lesson; do it on the board. Then spend the time on the
non-saturating loss and on the failure modes, which is what students hit first
when they train one. FID is defined here once; the diffusion lesson reuses it. -->

---

## Two networks

![Generator and discriminator](assets/cv/gan-architecture.png)

| Network | Input | Output | Learnt |
|---|---|---|---|
| generator $G_\theta$ | $z \in \mathbb{R}^{d}$, $z \sim \mathcal{N}(0, I)$ | $x \in \mathbb{R}^{3 \times H \times W}$ | $\theta$ |
| discriminator $D_\psi$ | $x \in \mathbb{R}^{3 \times H \times W}$ | $D_\psi(x) \in [0, 1]$ | $\psi$ |

$D_\psi(x)$ is the probability that $x$ is a real image. $D$ sees real and
generated images; $G$ never sees a real image. The only information about the
data that reaches $G$ is the gradient of $D$.

The generator is the decoder of the VAE with no encoder: $z \sim p(z)$, one
forward pass, an image. The generated images follow a distribution $p_g$,
defined implicitly: it can be sampled, but $p_g(x)$ cannot be evaluated.

---

## The minimax objective

$$
\min_{G} \max_{D}\; V(D, G) = \mathbb{E}_{x \sim p_{data}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]
$$

$V$ is the negative binary cross-entropy of a classifier with real images
labelled 1 and generated images labelled 0.

- $D$ **maximises** $V$: $D(x) \to 1$ on real images, $D(G(z)) \to 0$ on
  generated ones.
- $G$ **minimises** $V$, through the second term only: $D(G(z)) \to 1$.

This is a two-player zero-sum game. The solution is a saddle point, not a
minimum: gradient descent is not guaranteed to reach it.

---

## The optimal discriminator

Fix $G$. Write $V$ as one integral over $x$, with $p_g$ the distribution of
$G(z)$:

$$
V(D, G) = \int \Big( p_{data}(x) \log D(x) + p_g(x) \log\big(1 - D(x)\big) \Big)\, dx
$$

For $a, b > 0$, $y \mapsto a \log y + b \log(1 - y)$ is maximal at
$y = a / (a + b)$. Pointwise:

$$
D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

Substitute $D^*$, with $m = \tfrac{1}{2}(p_{data} + p_g)$:

$$
V(D^*, G) = \mathrm{KL}(p_{data} \,\Vert\, m) + \mathrm{KL}(p_g \,\Vert\, m) - \log 4 = 2\, \mathrm{JS}(p_{data} \,\Vert\, p_g) - \log 4
$$

With an optimal discriminator, the generator minimises the **Jensen-Shannon
divergence** to the data. Its minimum, $-\log 4$, is reached only at
$p_g = p_{data}$, where $D^* = \tfrac{1}{2}$ everywhere: the discriminator can
do no better than chance.

---

## The non-saturating generator loss

Write $D(x) = \mathrm{sigmoid}(s(x))$, with $s$ the logit. Early in training
$G$ is bad and $D$ rejects its samples easily: $D(G(z)) \approx 0.01$.

| Generator loss | Gradient w.r.t. the logit $s$ | at $D = 0.01$ |
|---|---|---|
| $\log(1 - D(G(z)))$, minimised (minimax) | $-D$ | $-0.01$ |
| $-\log D(G(z))$, minimised (non-saturating) | $-(1 - D)$ | $-0.99$ |

The minimax loss saturates exactly when the generator most needs a signal. The
non-saturating loss has the same fixed point, $D(G(z)) \to 1$, and a gradient
99 times larger at the start. Every implementation uses it: train $G$ with the
binary cross-entropy and the label "real".

---

## The training loop

![Data flow in a GAN](assets/cv/d2l-gan.png)
*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Alternate one step on $D$ and one step on $G$, on each minibatch:

```python
fake = G(torch.randn(B, 100))
d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
opt_D.zero_grad(); d_loss.backward(); opt_D.step()
g_loss = bce(D(fake), ones)                      # non-saturating
opt_G.zero_grad(); g_loss.backward(); opt_G.step()
```

`.detach()` stops the discriminator loss from updating $G$. The discriminator
is **used** in the $G$ step, but only `opt_G` steps, so $\psi$ does not move.

Neither loss is a measure of progress: each is measured against an opponent
that changes every step. Progress is judged on samples from a fixed batch of
$z$, and with FID (last slide).

---

## DCGAN: dimensional flow

<!-- placeholder: image to add (DCGAN generator: z=100 projected to 4x4x1024, four fractionally-strided convolutions to 64x64x3; Radford, Metz, Chintala 2016, Fig. 1) -->

The generator is a stack of transposed convolutions, $4 \times 4$ kernels,
stride 2, padding 1. Output size: $(H - 1) \cdot 2 - 2 + 4 = 2H$.

| # | Layer | Output shape |
|---|---|---|
| 1 | $z \sim \mathcal{N}(0, I)$ | $100$ |
| 2 | linear, reshape, BN, ReLU | $1024 \times 4 \times 4$ |
| 3 | transposed conv, BN, ReLU | $512 \times 8 \times 8$ |
| 4 | transposed conv, BN, ReLU | $256 \times 16 \times 16$ |
| 5 | transposed conv, BN, ReLU | $128 \times 32 \times 32$ |
| 6 | transposed conv, $\tanh$ | $3 \times 64 \times 64$ |

About 12.7M generator weights. The discriminator is the mirror: strided
convolutions $3 \times 64 \times 64 \to 128 \times 32 \times 32 \to \dots \to
1024 \times 4 \times 4$, then one logit. No pooling and no fully connected
hidden layers: resolution changes only through strides. Pixels are scaled to
$[-1, 1]$ to match the $\tanh$.

---

## Mode collapse and instability

**Mode collapse.** The objective asks each sample to look real, never the set
of samples to cover the data. If one output $x^\star$ fools $D$, $G$ can map
every $z$ near $x^\star$: all generated digits are sevens. $D$ then learns to
reject $x^\star$, $G$ jumps to another mode, and the pair can cycle.

**Vanishing signal.** When $p_{data}$ and $p_g$ lie on disjoint
low-dimensional sets, a perfect $D$ exists and the JS divergence is constant,
$\log 2$: it gives no direction toward the data.

$$
p_{data} \perp p_g \;\Rightarrow\; \mathrm{JS}(p_{data} \,\Vert\, p_g) = \log 2 \text{ for any } G
$$

**Non-convergence.** Alternating gradient steps on a saddle point can
oscillate forever. The stabilisers that are used in practice all bound how
fast $D$ can change: spectral normalisation of $D$'s weights, a gradient
penalty on $D$ (WGAN-GP), and a smaller learning rate for $G$.

---

## Conditioning, and StyleGAN

**Conditional GAN.** Give the condition $y$ to both networks: $G(z, y)$ and
$D(x, y)$, so that $D$ rejects a realistic image paired with the wrong label.

**StyleGAN.** A mapping MLP turns $z$ into a style vector $w \in
\mathbb{R}^{512}$, and $w$ sets the per-channel scale and shift of the
normalisation in every generator layer. The same mechanism, a condition that
modulates normalisation, conditions the Diffusion Transformer of the next
lesson.

GANs sample in one forward pass and give sharp images. They lost image
synthesis to diffusion on stability and on diversity, and survive as the
adversarial **loss term** inside other models: the Stable Diffusion VAE
decoder is trained with one.

---

## Evaluating a generator: FID

A GAN has no likelihood to report. The **Fréchet Inception Distance** compares
the distribution of generated images with the distribution of real ones, in
the 2048-dimensional feature space of a pretrained Inception-v3.

Fit a Gaussian to each feature set, $(\mu_r, \Sigma_r)$ on real images,
$(\mu_g, \Sigma_g)$ on generated ones, and take the Fréchet distance between
the two Gaussians:

$$
\mathrm{FID} = \| \mu_r - \mu_g \|^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

Lower is better; 0 for identical Gaussians. The mean term measures quality,
the covariance term measures diversity: a collapsed generator has a small
$\Sigma_g$ and a large FID even when every sample looks real. $\mu \in
\mathbb{R}^{2048}$, $\Sigma \in \mathbb{R}^{2048 \times 2048}$, so FID needs
many samples (typically 50k) and is biased at small sample sizes. It is the
standard metric for GANs and diffusion models alike.

---

## Exercise

1. At a point $x$, $p_{data}(x) = 0.3$ and $p_g(x) = 0.1$. What is $D^*(x)$?
   At a point where $p_g(x) > 0$ but $p_{data}(x) = 0$?
2. The generator is perfect, $p_g = p_{data}$. Give $D^*$, $V(D^*, G)$ and the
   binary cross-entropy of the discriminator (per example, averaged over real
   and fake).
3. At the start of training $D(G(z)) = 0.05$. Gradient with respect to the
   logit for the minimax and for the non-saturating generator losses? Ratio?
4. A DCGAN generator must output $3 \times 128 \times 128$ from $1024 \times 4
   \times 4$ with the same transposed convolutions. How many layers?

<!-- notes: 10 minutes. Question 2: the discriminator loss converging to log 2
= 0.693 per example is what a healthy GAN run shows. -->

---

## Solution

**1.** $D^*(x) = 0.3 / (0.3 + 0.1) = 0.75$. Where $p_{data}(x) = 0$:
$D^*(x) = 0$, the point is certainly fake.

**2.** $D^* = \tfrac{1}{2}$ everywhere. $V(D^*, G) = \log \tfrac12 + \log
\tfrac12 = -\log 4 \approx -1.386$. The discriminator's BCE is $-V/2 = \log 2
\approx 0.693$ per example: chance level. A discriminator loss that settles
near 0.69 is a sign of a balanced game, not of a broken discriminator.

**3.** Minimax: $-D = -0.05$. Non-saturating: $-(1 - D) = -0.95$. Ratio 19.

**4.** Each layer doubles the side: $4 \to 8 \to 16 \to 32 \to 64 \to 128$,
**5** transposed convolutions (one more than at $64 \times 64$).
