# GANs

A VAE learns a distribution by writing down a likelihood. A GAN never writes
one down: it trains a generator against a critic that is itself learning, and
takes sharp samples as the prize for a much harder optimisation.

<!-- notes: 25 minutes. Do not derive the theory — spend the time on the
failure modes, because that is what students hit within ten minutes of running
one. If there is a GPU in the room, start a DCGAN on MNIST at the beginning of
the lesson and look at the samples at the end. -->

---

## Two networks, opposed

![Generator and discriminator](assets/cv/gan-architecture.png)

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

Two facts hide in four lines. The `.detach()` in the discriminator step stops
the generator being updated by the discriminator's loss — forget it and the
generator is trained to *help* the discriminator. And the generator's target is
`ones`, not `1 - zeros`: minimising $\log(1 - D(G(z)))$ has vanishing gradient
exactly when the generator is bad, so everyone maximises $\log D(G(z))$
instead.

---

## There is no loss curve to read

In every other model in this course, a falling loss means progress. Here the
two losses are measured against a moving opponent. `d_loss` going down can mean
the discriminator is winning, which means the generator is about to stop
learning.

> A GAN's loss curves are uninterpretable. Judge it by looking at samples on a
> fixed noise vector every epoch, and by FID.

Fix a batch of `z` at the start of training and decode it at every checkpoint.
That filmstrip is the diagnostic.

---

## Mode collapse

The generator finds one output that reliably fools the discriminator and
produces it for every input. On MNIST: a thousand samples, all sevens.

Nothing in the objective rewards diversity. $D$ only asks "is this real", never
"have I seen this before". The generator has found a legitimate local optimum
of the game.

Symptoms: samples in a batch nearly identical; FID stuck high while `g_loss`
looks healthy; latent interpolation that does not move.

Mitigations: minibatch discrimination (let $D$ see a batch, not one image),
unrolled or two-timescale updates, and switching to a Wasserstein objective.

---

## Non-convergence and imbalance

| Failure | What you see | Fix |
|---|---|---|
| Discriminator too strong | `d_loss` → 0, `g_loss` explodes, no gradient for $G$ | lower `lr_D`, label smoothing, noisy labels |
| Generator too strong | `d_loss` high, samples still bad | more $D$ steps per $G$ step |
| Oscillation | samples cycle between modes, never settle | TTUR, EMA of generator weights |

The equilibrium is a saddle point, not a minimum, and gradient descent has no
convergence guarantee there — a GAN can circle forever without diverging *or*
improving.

---

## The fixes that actually work

- **WGAN-GP.** Replace the classification objective with the Wasserstein
  distance, and enforce the required Lipschitz constraint by penalising the
  critic's gradient norm. The critic's output becomes a meaningful quality
  score, and the vanishing-gradient failure disappears.
- **Spectral normalization.** Divide each weight matrix by its largest singular
  value. One line per layer, no extra loss term, and it bounds the
  discriminator's Lipschitz constant directly.

```python
D_layer = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
```

- **TTUR.** Two time-scale update rule: a larger learning rate for $D$ than for
  $G$ — typically `2e-4` and `1e-4`. Cheapest stabiliser on the list.
- **EMA of generator weights** for sampling. Nearly free, and it removes most
  of the epoch-to-epoch sample jitter.

Start with spectral norm plus TTUR. Reach for WGAN-GP if it still oscillates.

---

## Conditional GANs

Feed the condition to both networks — a class label, an embedding, a whole
image.

```python
z_y = torch.cat([z, embed(y)], dim=1)                   # generator input, (B, dz + E)
ymap = embed(y)[..., None, None].expand(-1, -1, x.size(2), x.size(3))   # (B, E, H, W)
d_in = torch.cat([x, ymap], dim=1)                      # (B, 3 + E, H, W)
```

The embedding has to be broadcast to a spatial map before it can be
concatenated with an image: `expand_as(x[:, :1])` cannot turn `(B, E)` into
`(B, 1, H, W)` and raises.

If only $G$ sees the label, nothing forces it to be used: the generator ignores
`y` and produces unconditional samples. $D$ must be able to reject a correct
image paired with the wrong label.

**pix2pix** is a conditional GAN where the condition is the input image —
sketch to photo, map to satellite — and it needs *paired* data. **CycleGAN**
removes that requirement with two generators and a cycle-consistency loss:
translate to the other domain and back, and demand you recover the original.
Horses to zebras, summer to winter, no pairs.

---

## Evaluating a generator

There is no held-out likelihood to report, so evaluation compares
*distributions* of samples.

**FID** embeds real and generated images with an Inception network and compares
the two Gaussians fitted to those features:

$$
FID = |\mu_r - \mu_g|^2 + Tr\left( \Sigma_r + \Sigma_g - 2 (\Sigma_r \Sigma_g)^{1/2} \right)
$$

Lower is better. It is sensitive to both quality and diversity, which is why it
caught mode collapse when the Inception Score did not.

Caveats worth stating whenever you report one: FID is biased by sample count
(use ≥10k, always the same count when comparing), depends on the resizing and
the Inception weights, and looks through a network trained on ImageNet — so it
means much less on medical or satellite images. Report FID *and* show samples.

---

## Where GANs stand now

Diffusion beat them on image quality and on training stability, and took the
text-to-image field. GANs remain the right tool when:

- **inference must be one forward pass** — real-time video, super-resolution,
  on-device generation
- the domain is narrow and a StyleGAN-class model is already tuned for it
  (faces, textures)
- you need an adversarial *loss term* inside another model — the perceptual
  discriminator in a VAE decoder or a neural codec is a GAN

Learn the adversarial loss as a component; as a standalone image generator,
the next lesson has replaced it.

---

## Check yourself

1. `d_loss` falls to nearly zero over a few hundred steps while `g_loss`
   explodes. Who is winning, and what does this lesson tell you to change?

   **Answer.** The discriminator. It is right often enough that almost no
   gradient reaches the generator. Lower `lr_D`, add label smoothing or noisy
   labels — and reach for spectral normalization plus TTUR before anything more
   elaborate.

2. The generator's loss is `bce(D(G(z)), ones)` and not `bce(D(G(z)), zeros)`
   negated. Why does everyone use that form?

   **Answer.** Minimising $\log(1 - D(G(z)))$ has a vanishing gradient exactly
   when the generator is bad, which is when it needs the signal most. Maximising
   $\log D(G(z))$ — the non-saturating form — keeps the gradient alive early.

3. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn
   embed = nn.Embedding(10, 32)
   x = torch.randn(16, 3, 64, 64)
   y = torch.randint(0, 10, (16,))
   ymap = embed(y)[..., None, None].expand(-1, -1, x.size(2), x.size(3))
   print(torch.cat([x, ymap], dim=1).shape)   # -> torch.Size([16, 35, 64, 64])
   ```
