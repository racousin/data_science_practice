# Autoencoders and VAEs

Every model so far learned $p(y|x)$ — a label given an image. A generative
model learns $p(x)$ itself, and can therefore draw new samples from it.

<!-- notes: 30 minutes. State the generative problem once, here, because the
next two lessons assume it. The reparameterization trick is the slide to slow
down on: students accept the formula and cannot say why the naive version has
no gradient. Ask them. -->

---

## The generative problem

![Samples from VAE, GAN, flow and diffusion models](/api/academic_courses/assets/lessons/93/generative-models-examples.png)

Given samples $x$ from an unknown distribution, learn a model you can **sample
from**. Two requirements, and they pull in different directions:

- the samples must look like the data (quality)
- the samples must cover the data (diversity)

A model that memorises one training image scores perfectly on the first and
zero on the second. Every failure mode in this session is a collapse of one
into the other.

---

## Four families

| Family | Learns | Sampling | Weakness |
|---|---|---|---|
| Autoencoder / VAE | a latent code + decoder | one forward pass | blurry |
| GAN | a generator, adversarially | one forward pass | unstable, mode collapse |
| Normalizing flow | an invertible map, exact $p(x)$ | one forward pass | architecturally constrained |
| Diffusion | a denoiser | many passes | slow |

All four are still in use. Diffusion dominates image synthesis; VAEs survive as
components *inside* diffusion systems; GANs survive where latency matters.

---

## The autoencoder

Two networks trained to reproduce the input through a narrow bottleneck.

```python
z = encoder(x)          # (B, 3, 64, 64) -> (B, 128)
x_hat = decoder(z)      # (B, 128)       -> (B, 3, 64, 64)
loss = F.mse_loss(x_hat, x)
```

No labels are needed — the input is the target. The bottleneck is what forces
learning: with `latent_dim` equal to the input size the network learns the
identity and nothing else.

The interesting object is not `x_hat`. It is `z`.

---

## What the latent code is good for

- **Compression.** A 12288-dimensional image becomes 128 numbers, lossily but
  semantically.
- **Denoising.** Train with corrupted inputs and clean targets — a denoising
  autoencoder learns the data manifold rather than the identity.
- **Anomaly detection.** Reconstruction error is high for inputs unlike
  anything in training. Fit on normal production data, threshold the error,
  flag the rest.

```python
err = ((x - model(x)) ** 2).mean(dim=[1, 2, 3])
anomalies = err > threshold          # threshold from a validation quantile
```

The anomaly-detection use is the one that pays for itself in industry, and it
needs no labels at all.

---

## Why a plain autoencoder is a bad generator

Sample a random `z` and decode it. You get noise.

Nothing in the objective constrains the *shape* of the latent space. The
encoder is free to scatter training points anywhere — clusters far apart, vast
empty regions between them. Reconstruction only cares that each training point
maps somewhere it can be decoded from.

> An autoencoder learns a code for the data it has seen. It does not learn a
> distribution, so there is nothing to sample.

Fixing that is the entire contribution of the VAE: force the latent space to
match a distribution you *can* sample from.

---

## The VAE

![VAE architecture](/api/academic_courses/assets/lessons/93/vae-architecture.png)

The encoder no longer outputs a point. It outputs the parameters of a Gaussian
over latent codes:

```python
h = encoder(x)
mu, logvar = self.fc_mu(h), self.fc_logvar(h)   # each (B, d)
```

Training pushes every such Gaussian toward the prior $\mathcal{N}(0, I)$ while
still requiring the decoder to reconstruct. The two forces together fill the
latent space — overlapping blobs covering the unit ball, no holes — so sampling
becomes trivial: draw `z ~ N(0, I)`, decode.

---

## The reparameterization trick

The naive version — sample `z` from `N(mu, sigma)` and backpropagate — does not
work. Sampling is not differentiable, so no gradient reaches `mu` or `logvar`.

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)          # the randomness, detached from theta
    return mu + eps * std                # differentiable in mu and std
```

Move the randomness into a constant `eps` drawn outside the computation graph,
and make `z` a deterministic function of `mu`, `std` and `eps`. The gradient
now flows through the arithmetic.

This is the single idea that made VAEs trainable, and the same trick appears
throughout stochastic optimisation.

---

## The objective

Maximise the evidence lower bound on $\log p(x)$:

$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) | p(z))
$$

The first term is reconstruction. The second is a regulariser pulling the
encoder's Gaussian toward the prior. With a Gaussian encoder and an
$\mathcal{N}(0, I)$ prior the KL term has a closed form and is three lines of
code:

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
$$

---

## The loss in code

```python
recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
loss = recon + beta * kl
```

`reduction="sum"` then divide by the batch, not `reduction="mean"` — otherwise
the reconstruction term is scaled by `1/(C·H·W)` relative to the KL and the
balance is silently wrong by four orders of magnitude. This is the most common
VAE bug.

`beta` sets the trade-off. Too low and the KL is ignored: you have an
autoencoder again, sharp reconstructions and unusable samples. Too high and the
posterior collapses to the prior: the decoder ignores `z` and emits the dataset
mean. Anneal `beta` from 0 over the first few epochs.

---

## Blurriness

VAE samples are recognisable and soft. Two reasons compound:

- The Gaussian likelihood makes the reconstruction term an L2 loss, and L2 is
  minimised by the *average* of the plausible outputs. Averaging sharp edges
  produces a blurred one.
- The KL term deliberately smooths the latent space, so nearby codes decode to
  similar images.

This is a property of the objective, not a training failure — you do not tune
it away. If sharpness is the requirement the answer is a GAN or diffusion, or a
VQ-VAE, which replaces the Gaussian latent with a discrete codebook.

---

## Latent interpolation

The test that tells you whether the latent space is actually structured:

```python
z1, z2 = encode(x1), encode(x2)
frames = [decode(z1 + t * (z2 - z1)) for t in torch.linspace(0, 1, 10)]
```

A good latent space gives a smooth semantic morph — the digit 3 bending into an
8, a face slowly turning. A bad one gives a crossfade: two ghosts, one fading
out and one fading in.

Interpolate in a VAE and it morphs; in a plain autoencoder it crossfades. That
figure is worth more than a page of loss curves.

---

## Where VAEs actually live now

Nobody ships a VAE as an image generator. They ship as **compressors inside
other systems**:

- Stable Diffusion runs its diffusion process in a VAE latent space at 1/8
  resolution — the next lesson.
- VQ-VAE tokenises images into discrete codes so a transformer can model them.
- Anomaly detection and representation learning, where the latent *is* the
  deliverable.

Learn the VAE for the encoder–latent–decoder pattern and the
reparameterization trick. Both reappear immediately.
