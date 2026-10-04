# Generation Lab — GAN and Diffusion

Two Colab notebooks, one task: generate faces from the CelebA dataset. Run
the GAN first, then the diffusion model, and compare what each costs to train
and to sample. Both run on a free Colab T4 GPU: **Runtime → Change runtime
type → T4 GPU**.

---

## 1. Faces with a GAN (DCGAN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s7-gan-faces.ipynb)

Build a DCGAN generator and discriminator, train them against each other
for 20 epochs on 30,000 faces at $64 \times 64$, watch the samples improve,
then walk the latent space. About 30 minutes.

## 2. Faces with a diffusion model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s7-diffusion-faces.ipynb)

The same faces at $32 \times 32$:
- train a small DDPM from scratch;
- sample it with DDPM (1,000 steps) and DDIM (50 steps);
- fine-tune a pretrained CIFAR-10 DDPM on the faces with half the budget;
- sample a large pretrained CelebA-HQ model at $256 \times 256$;
- interpolate between two noises.

About 25 minutes.

---

## What to bring back

One table, filled from your own two runs: training time, sampling time per
image, and which samples you prefer — with one sentence on why.
