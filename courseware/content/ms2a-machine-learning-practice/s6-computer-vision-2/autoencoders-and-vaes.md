# Generation models

Every model so far learned $p(y|x)$ — a label given an image. A generative
model learns $p(x)$ itself, and can therefore draw new samples from it.

<!-- notes: 30 minutes. State the generative problem once, here, because the
next two lessons assume it. The reparameterization trick is the slide to slow
down on: students accept the formula and cannot say why the naive version has
no gradient. Ask them. -->

---

## The generative problem

![Samples from VAE, GAN, flow and diffusion models](assets/cv/generative-models-examples.png)

Given samples $x$ from an unknown distribution, learn a model you can **sample
from**. Two requirements, and they pull in different directions:

- the samples must look like the data (quality)
- the samples must cover the data (diversity)

A model that memorises one training image scores perfectly on the first and
zero on the second. Every failure mode in this session is a collapse of one
into the other.

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

![VAE architecture](assets/cv/vae-architecture.png)

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
    eps = torch.randn_like(std)  # the randomness, detached from theta
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


## GANs

A VAE learns a distribution by writing down a likelihood. A GAN never writes
one down: it trains a generator against a critic that is itself learning, and
takes sharp samples as the prize for a much harder optimisation.

<!-- notes: 25 minutes. Do not derive the theory — spend the time on the
failure modes, because that is what students hit within ten minutes of running
one. If there is a GPU in the room, start a DCGAN on MNIST at the beginning of
the lesson and look at the samples at the end. -->

---

## Two networks, opposed


![gan-architecture.webp](assets/cv/gan-architecture.webp)


- The **generator** $G$ maps noise $z$ to an image. It never sees real data.
- The **discriminator** $D$ takes an image and outputs the probability it is
  real. It sees both.

$D$ is trained to be right, $G$ to make $D$ wrong. The only learning signal
reaching the generator is the discriminator's gradient, which is why everything
about GAN training is a question of keeping that signal alive.

---

## The objective

$$
\min_G \max_D \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))]
$$

$D$ maximises: push $D(x)$ toward 1 on real data and $D(G(z))$ toward 0 on
fakes. $G$ minimises the second term: make $D(G(z))$ large.

At the theoretical optimum $G$ reproduces the data distribution and
$D(x) = 0.5$ everywhere — the discriminator can do no better than a coin flip.
You will never observe this, but it is the target the dynamics move toward.

---

## The training loop

```python
d_loss = bce(D(real), ones) + bce(D(G(z).detach()), zeros)
opt_D.zero_grad(); d_loss.backward(); opt_D.step()

g_loss = bce(D(G(z)), ones)          # non-saturating form
opt_G.zero_grad(); g_loss.backward(); opt_G.step()
```

---

## Diffusion Models

Destroying an image is easy: add noise until nothing is left. Diffusion models
learn to run that process backwards, one small step at a time, and generating
becomes a sequence of easy problems instead of one impossible one.

<!-- notes: 30 minutes. The whole lesson is one idea — a hard sampling problem
decomposed into T easy denoising problems. Get that across and the equations
follow. Run the diffusers snippet live; it is three lines and it lands better
than any diagram. -->

---

## The forward process


![imagesss.jpeg](assets/cv/imagesss.jpeg)



Take a training image and add a little Gaussian noise, $T$ times:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t ; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

The variance schedule $\beta_t$ is fixed, not learned, and small — around
$10^{-4}$ rising to $0.02$ over 1000 steps. After $T$ steps the image is
indistinguishable from pure noise.

Nothing is trained here. It is a data-corruption procedure, and the source of
the model's training pairs.

---

## Jumping to any step

Composing $t$ Gaussians gives a Gaussian, so there is a closed form and no loop
is needed at training time:

$$
q(x_t | x_0) = \mathcal{N}(x_t ; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

with $\bar{\alpha}_t$ the running product of $(1 - \beta_s)$.

```python
noise = torch.randn_like(x0)
ab = alpha_bar[t].view(-1, 1, 1, 1)  # (B,) -> (B,1,1,1): broadcasts
xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
```

Sample a random `t` per image in the batch, corrupt in one operation, done.
This is why diffusion training is cheap and stable: every step is an
independent supervised regression, no opponent and no sampling loop.

---

## The reverse process is what you learn

Reversing one noising step is intractable in general, but for small $\beta_t$
the reverse transition is *also* approximately Gaussian. So parameterise it:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1} ; \mu_\theta(x_t, t), \Sigma_t)
$$

and learn $\mu_\theta$ with a network. The algebra of DDPM shows that
predicting the mean is equivalent to predicting the **noise that was added**,
which is a much better-conditioned regression target.

The network is a U-Net — encoder, decoder, skip connections, exactly the
segmentation architecture of this session — taking the noisy image and the
timestep, and returning a noise-shaped tensor.

---

## The training objective

$$
L = \mathbb{E}_{t, x_0, \epsilon} \left[ | \epsilon - \epsilon_\theta(x_t, t) |^2 \right]
$$

Mean squared error between the noise that was added and the noise the model
predicts. That is the entire loss.

```python
t = torch.randint(0, T, (x0.size(0),), device=x0.device)
noise = torch.randn_like(x0)
ab = alpha_bar[t].view(-1, 1, 1, 1)
xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
loss = F.mse_loss(model(xt, t), noise)
```

The `.view(-1, 1, 1, 1)` is not cosmetic. Without it a per-image `t` gives a
`(B,)` tensor that broadcasts against the *width* axis: it raises on most
shapes, and silently applies the wrong noise level to every image on the shapes
where `B` happens to equal `W`.

Compare with the previous lesson: one network, one loss, a curve that goes down
and means something. The stability is not a detail — it is the reason diffusion
displaced GANs.

---

## Sampling

```python
x = torch.randn(shape)
for t in reversed(range(T)):
    eps = model(x, t)
    x = scheduler.step(eps, t, x).prev_sample
```

Start from pure noise, predict the noise, remove a fraction of it, add a little
fresh noise back, repeat. The added noise is not an accident: without it the
process is deterministic and collapses toward the dataset mean.

The cost is now visible: a GAN generates in **one** forward pass, DDPM in
**1000**. That is the trade you accepted for stability.
