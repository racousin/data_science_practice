# Autoencoders and VAEs

Every model so far learned $p(y \mid x)$, a label given an image. A generative
model learns $p_\theta(x)$ itself, so that new images can be drawn from it. This
lesson and the next two give three answers to the same question: VAEs, GANs
and diffusion models.

<!-- notes: 30 minutes. State the generative problem once, here: the next two
lessons assume it. The ELBO derivation is three lines, do it on the board. The
reparameterisation slide is the one to slow down on: ask why the naive version
has no gradient before showing the answer. End on the Stable Diffusion slide,
it is the bridge to the diffusion lesson. -->

---

## The generative task

![Four families of generative models](assets/cv/generative-models-examples.png)

Data: images $x^{(1)}, \dots, x^{(N)}$ drawn from an unknown $p_{data}$, with
$x \in \mathbb{R}^{3 \times H \times W}$. Goal: a model $p_\theta$ close to
$p_{data}$ that we can **sample from**.

A $64 \times 64$ RGB image lives in $\mathbb{R}^{12288}$, and natural images
fill a vanishingly small part of it. Every family in the figure uses the same
trick: draw a simple random variable $z$ (or $x_T$) from $\mathcal{N}(0, I)$
and learn a network that maps it to that small part.

| Model | What is learnt | Training signal |
|---|---|---|
| VAE | encoder + decoder | likelihood lower bound |
| GAN | generator + discriminator | a learnt critic |
| Diffusion | a denoiser | noise regression |

---

## The autoencoder

Two networks and a bottleneck, $d \ll 3HW$:

$$
z = f_\phi(x) \in \mathbb{R}^d, \qquad \hat{x} = g_\theta(z) \in \mathbb{R}^{3 \times H \times W}, \qquad
\mathcal{L}(\phi, \theta) = \frac{1}{N} \sum_{i} \| x^{(i)} - g_\theta(f_\phi(x^{(i)})) \|^2
$$

| Component | Input | Output | Learnt |
|---|---|---|---|
| encoder $f_\phi$ | $3 \times 64 \times 64$ | $d = 128$ | $\phi$ |
| decoder $g_\theta$ | $128$ | $3 \times 64 \times 64$ | $\theta$ |

No labels: the input is the target. The bottleneck is what forces learning.
With $d \geq 3HW$ the identity map is a perfect solution and nothing is
learnt. With $d = 128$ the network must keep the 128 directions that best
explain the images: a non-linear PCA.

```python
z = encoder(x)                 # (B, 3, 64, 64) -> (B, 128)
x_hat = decoder(z)             # (B, 128) -> (B, 3, 64, 64)
loss = F.mse_loss(x_hat, x)
```

---

## Why the latent space cannot be sampled

To generate, draw $z$ and compute $g_\theta(z)$. But draw $z$ from **which
distribution**?

The loss only constrains the $N$ points $f_\phi(x^{(i)})$. Where they land, how
spread they are, and what lies between them is left free. The encoder can
place the training codes on a thin curved set with large empty regions, and
the decoder is never trained on those regions.

$$
z \sim \mathcal{N}(0, I) \;\Rightarrow\; z \text{ falls where no training code lies} \;\Rightarrow\; g_\theta(z) \text{ is not an image}
$$

The autoencoder learns a **code**, not a **distribution**. The VAE adds the
missing constraint: the codes must follow a distribution fixed in advance,
$p(z) = \mathcal{N}(0, I)$.

---

## A latent variable model

Declare how an image is generated: first a code, then an image given the code.

$$
z \sim p(z) = \mathcal{N}(0, I_d), \qquad x \sim p_\theta(x \mid z) = \mathcal{N}\big(g_\theta(z), \sigma^2 I\big)
$$

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz
$$

Maximum likelihood on $p_\theta(x)$ would train the decoder, and sampling is
built in. The integral is the problem: over $\mathbb{R}^{128}$, almost every
$z$ gives $p_\theta(x \mid z) \approx 0$, so Monte Carlo from $p(z)$ never hits
the codes that explain $x$.

The fix: a second network that proposes, for each $x$, the codes that are
likely to have produced it.

$$
q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x),\; \mathrm{diag}\, \sigma^2_\phi(x)\big), \qquad
\mu_\phi(x),\; \log \sigma^2_\phi(x) \in \mathbb{R}^d
$$

The encoder no longer outputs a point: it outputs $2d$ numbers, a mean and a
variance per latent dimension.

---

## The evidence lower bound

Multiply and divide by $q_\phi$, then apply Jensen ($\log$ is concave):

$$
\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\right]
\;\geq\; \mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x \mid z)\big] - \mathrm{KL}\big(q_\phi(z \mid x)\,\Vert\, p(z)\big)
$$

The right-hand side is the **ELBO**. The gap is exactly
$\mathrm{KL}\big(q_\phi(z \mid x) \,\Vert\, p_\theta(z \mid x)\big) \geq 0$: the
bound is tight when the encoder matches the true posterior.

Both terms are computable:

- **reconstruction.** With the Gaussian decoder,
  $-\log p_\theta(x \mid z) = \frac{1}{2\sigma^2}\| x - g_\theta(z)\|^2 + \text{const}$:
  the autoencoder's squared error.
- **regularisation.** The KL pulls each $q_\phi(z \mid x)$ toward
  $\mathcal{N}(0, I)$. This is the constraint the autoencoder was missing.

Training maximises the ELBO in $\phi$ and $\theta$ jointly.

---

## The KL in closed form

Between two Gaussians the KL has a closed form. For
$q = \mathcal{N}(\mu, \mathrm{diag}\,\sigma^2)$ and $p = \mathcal{N}(0, I_d)$:

$$
\mathrm{KL}(q \,\Vert\, p) = \frac{1}{2} \sum_{j=1}^{d} \left( \mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2 \right)
$$

Each term is zero only at $\mu_j = 0$, $\sigma_j = 1$. The $\mu_j^2$ term pulls
codes toward the origin; the $\sigma_j^2 - \log\sigma_j^2$ term forbids
$\sigma_j \to 0$, so each image occupies a **region** of latent space, not a
point. Neighbouring images get overlapping regions, and the holes of the
autoencoder are filled.

The loss, per image, summed over pixels and latent dimensions:

```python
recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
kl = 0.5 * torch.sum(mu**2 + logvar.exp() - 1 - logvar) / x.size(0)
loss = recon + beta * kl
```

The encoder predicts $\log\sigma^2$, not $\sigma$: it is unconstrained in sign.
$\beta = 1$ is the ELBO; $\beta$ trades reconstruction against a smooth latent
space.

---

## The reparameterisation trick

![VAE with reparameterised sampling](assets/cv/vae-architecture.png)

The reconstruction term is an expectation over $z \sim q_\phi(z \mid x)$,
estimated with one sample. But a sampling step has no derivative with respect
to $\mu_\phi$ and $\sigma_\phi$, so the gradient stops at the draw.

Move the randomness out of the graph:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I_d)
$$

$z$ has the same distribution, and is now a deterministic, differentiable
function of $\mu_\phi$ and $\sigma_\phi$, with $\epsilon$ a constant input:
$\partial z_j / \partial \mu_j = 1$, $\partial z_j / \partial \sigma_j = \epsilon_j$.

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)       # drawn outside the graph
z = mu + eps * std                # gradients reach mu and logvar
```

---

## A convolutional VAE: dimensional flow

Input $3 \times 64 \times 64$, latent $d = 128$. Every conv is $4 \times 4$,
stride 2, padding 1: it halves $H$ and $W$. Every transposed conv doubles them.

| # | Layer | Output shape |
|---|---|---|
| 1 | input $x$ | $3 \times 64 \times 64$ |
| 2 | conv, conv, conv, conv | $32{\times}32{\times}32 \to 64{\times}16{\times}16 \to 128{\times}8{\times}8 \to 256{\times}4{\times}4$ |
| 3 | flatten | $4096$ |
| 4 | two linear heads: $\mu_\phi$, $\log\sigma^2_\phi$ | $128$ each |
| 5 | $z = \mu + \sigma \odot \epsilon$ | $128$ |
| 6 | linear, reshape | $256 \times 4 \times 4$ |
| 7 | four transposed convs | $128{\times}8{\times}8 \to 64{\times}16{\times}16 \to 32{\times}32{\times}32 \to 3{\times}64{\times}64$ |
| 8 | sigmoid, $\hat{x}$ | $3 \times 64 \times 64$ |

About 3.0M parameters in total. The decoder mirrors the encoder, as in the
U-Net of the segmentation lesson, but **without skip connections**: every bit
of information must pass through the 128 numbers of row 5. Compression:
$12288 / 128 = 96\times$.

**Generation** uses rows 6 to 8 only: $z \sim \mathcal{N}(0, I_{128})$, decode.

---

## Why the samples are blurry

For a fixed $z$, the Gaussian decoder minimises the expected squared error over
all images the encoder may have mapped near $z$. The minimiser of a squared
error is a **mean**:

$$
g_\theta^*(z) = \mathbb{E}\big[x \mid z\big]
$$

Because the KL forces $\sigma_\phi(x) > 0$, the regions of different images
overlap, and several images share each $z$. Their mean is an image where the
edges and textures that disagree between them have been averaged out.

This is a property of the objective, not a training failure: more epochs do
not remove it. Removing it requires a loss that does not average, which is the
adversarial loss of the next lesson.

---

## The VAE inside Stable Diffusion

Stable Diffusion does not generate pixels. It generates in the latent space of
a pretrained convolutional VAE with downsampling factor $f = 8$:

$$
x \in \mathbb{R}^{3 \times 512 \times 512}, \qquad
z = \mathcal{E}(x) \in \mathbb{R}^{4 \times 64 \times 64}, \qquad
\hat{x} = \mathcal{D}(z) \in \mathbb{R}^{3 \times 512 \times 512}
$$

$786{,}432$ values become $16{,}384$: a factor of 48. The latent keeps a
spatial layout: it is a $64 \times 64$ image with 4 channels.

Two changes to the loss of this lesson make the reconstructions sharp:

- the KL weight is tiny ($\beta = 10^{-6}$): the latent only needs to be
  well-behaved, not exactly $\mathcal{N}(0, I)$;
- the squared error is replaced by a perceptual loss plus an **adversarial**
  loss from a patch discriminator, the GAN of the next lesson.

The VAE is trained once and frozen. The diffusion model of the last lesson then
learns $p(z)$ in this 48× smaller space.

---

## Exercise

A VAE has a 2-dimensional latent space. For one image $x$ the encoder returns

$$
\mu_\phi(x) = (1,\; 0), \qquad \sigma_\phi(x) = (1,\; 0.5)
$$

1. Write $q_\phi(z \mid x)$. Give one sample $z$ for $\epsilon = (0.2, -2)$.
2. Compute $\mathrm{KL}\big(q_\phi(z \mid x) \,\Vert\, \mathcal{N}(0, I_2)\big)$
   in nats. Which dimension contributes more, and why?
3. For which $\mu, \sigma$ is the KL zero? What would the decoder receive then,
   whatever the image?
4. Compression ratios. (a) the conv VAE of this lesson, $3 \times 64 \times 64
   \to 128$; (b) the Stable Diffusion VAE, $3 \times 512 \times 512 \to 4 \times
   64 \times 64$.

<!-- notes: 10 minutes. Question 3 is posterior collapse: if the KL wins, the
code carries no information about x and the decoder outputs the dataset mean. -->

---

## Solution

**1.** $q_\phi(z \mid x) = \mathcal{N}\big((1, 0),\; \mathrm{diag}(1, 0.25)\big)$.
$z = \mu + \sigma \odot \epsilon = (1 + 0.2,\; 0 + 0.5 \cdot (-2)) = (1.2,\; -1)$.

**2.**

| $j$ | $\mu_j^2$ | $\sigma_j^2$ | $-1 - \log\sigma_j^2$ | sum |
|---|---|---|---|---|
| 1 | 1 | 1 | $-1 - 0 = -1$ | 1 |
| 2 | 0 | 0.25 | $-1 + 1.386 = 0.386$ | 0.636 |

$\mathrm{KL} = \frac{1}{2}(1 + 0.636) = 0.818$ nats. Dimension 1 contributes
more: its mean is off the origin. Dimension 2 pays for being too narrow.

**3.** $\mu = 0$, $\sigma = 1$ for every image. Then $q_\phi(z \mid x) = p(z)$:
$z$ carries no information about $x$, and the best the decoder can do is output
the average image. This is **posterior collapse**, the failure of a too large
$\beta$.

**4.** (a) $12288 / 128 = 96$. (b) $786432 / 16384 = 48$. The Stable Diffusion
latent compresses less, and keeps a $64 \times 64$ spatial grid instead of a
flat vector.
