# Diffusion Models

Destroying an image is easy: add noise until nothing is left. Diffusion models
learn to run that process backwards, one small step at a time, and generating
becomes a sequence of easy problems instead of one impossible one.

<!-- notes: 30 minutes. The whole lesson is one idea — a hard sampling problem
decomposed into T easy denoising problems. Get that across and the equations
follow. Run the diffusers snippet live; it is three lines and it lands better
than any diagram. -->

---

## The forward process

![Forward noising and learned reverse](/api/academic_courses/assets/lessons/95/diffusion-process.png)

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
xt = alpha_bar[t].sqrt() * x0 + (1 - alpha_bar[t]).sqrt() * noise
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

The network is a U-Net — encoder, decoder, skip connections, exactly Session 6's
segmentation architecture — taking the noisy image and the timestep, and
returning a noise-shaped tensor.

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
xt = alpha_bar[t].sqrt() * x0 + (1 - alpha_bar[t]).sqrt() * noise
loss = F.mse_loss(model(xt, t), noise)
```

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

---

## Buying the steps back

Better solvers treat the reverse process as an ODE and take larger steps.

| Sampler | Typical steps | Note |
|---|---|---|
| DDPM | 1000 | the original, stochastic |
| DDIM | 50 | deterministic, reproducible from a seed |
| DPM-Solver++ | 20–30 | the practical default |
| Distilled / consistency models | 1–4 | quality cost, huge latency win |

Swapping the sampler is a one-line change requiring no retraining — the model
learned a denoiser, not a fixed trajectory. Sweep it before concluding the
model is bad.

---

## Latent diffusion

A 512×512 RGB image is 786k dimensions, and running a U-Net over it a thousand
times is why early diffusion models needed a datacentre.

Latent diffusion runs the entire process inside a pretrained VAE's latent
space, at 64×64×4 — a **48× reduction** — and decodes once at the end.

```python
z = vae.encode(x).latent_dist.sample() * 0.18215
# ... all T diffusion steps operate on z ...
image = vae.decode(z / 0.18215).sample
```

That single change is what made Stable Diffusion run on a consumer GPU. The
VAE absorbs the perceptually irrelevant high-frequency detail so the diffusion
model only has to model semantics — and the previous lesson's "nobody ships a
VAE as a generator" gets its footnote: they ship inside this.

---

## Conditioning on text

Three components, and none of them is the diffusion model itself:

1. A frozen text encoder (CLIP, or T5 in larger systems) turns the prompt into
   a sequence of token embeddings.
2. **Cross-attention** layers inside the U-Net attend from image positions to
   those embeddings — queries from the image, keys and values from the text.
3. Timestep embeddings are added at every block so the network knows how much
   noise it is looking at.

The text never becomes a single vector concatenated once. Cross-attention lets
different spatial regions attend to different words, which is what places the
colours correctly in "a red cube on a blue table". Session 7 covers attention.

---

## Classifier-free guidance

Train the same model with the condition dropped 10% of the time, so it learns
both a conditional and an unconditional denoiser. At sampling, extrapolate away
from the unconditional prediction:

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t) + s \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t) \right)
$$

The guidance scale $s$ is the knob every image-generation UI exposes. At
$s = 1$ there is no guidance. Around 7 is the usual default. Push it to 20 and
prompt adherence rises while diversity and realism collapse into
oversaturated, contrast-blown images.

It costs two forward passes per step instead of one. Everyone pays it.

---

## Using one

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")
image = pipe("a cross-section of a diesel engine, technical drawing",
             num_inference_steps=30, guidance_scale=7.5).images[0]
```

`diffusers` is the library. For a custom domain do not train from scratch —
fine-tune, and prefer the cheap adapters: **LoRA** (a few MB of low-rank
weights) or **DreamBooth** (a handful of images of one subject). ControlNet
adds a spatial condition — edges, a pose, a depth map — on top of a frozen base
model. Full pretraining is a seven-figure compute bill; fine-tuning is an
afternoon.

---

## Normalizing flows, for completeness

![Invertible transformation between distributions](/api/academic_courses/assets/lessons/95/normalizing-flow-diagram.png)

The fourth family. Build the generator from *invertible* layers, so the change
of variables formula gives the exact likelihood:

$$
\log p(x) = \log p_Z(f^{-1}(x)) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|
$$

Train by maximum likelihood — no adversary, no lower bound, an exact number you
can compare across models. The price is architectural: every layer must be
invertible with a cheaply computable Jacobian determinant, which rules out most
of what works in vision.

Flows lost the image-generation race. They remain useful where an exact density
is the deliverable — anomaly scoring, variational inference, normalising a
latent space inside another model.

---

## Choosing

| Need | Use |
|---|---|
| Best image quality, text conditioning | latent diffusion, fine-tuned with LoRA |
| One-pass inference, tight latency | GAN, or a distilled diffusion model |
| A structured latent space to manipulate | VAE / VQ-VAE |
| An exact likelihood | normalizing flow |
| Anomaly detection | autoencoder reconstruction error, or a flow |

The honest default for images is a pretrained latent diffusion model plus a
cheap adapter. Training one from scratch is a research project, not a feature.
