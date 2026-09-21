# Lab — Building Blocks, Measured

Challenge 8, *1 Minute Permuted MNIST*, gives your agent 60 seconds to train and 60
more to predict, on 3 CPU cores, on MNIST with the pixel positions and the label
meanings permuted. The notebook explains the problem, trains a baseline (logistic
regression in PyTorch), evaluates it as the challenge does (accuracy, and the time
of each call), and submits it. The rest is yours: the building blocks of this
session (architecture, optimizer and schedule, normalization, dropout, data
augmentation, early stopping) are the levers, and the notebook's last section says
how to test each one against the clock.

**Time:** 60 minutes. **Deliverable:** your agent on the leaderboard of challenge 8,
above the logistic-regression baseline, and the notebook's results table: one row
per change you tested.

<!-- notes: The notebook is a starting point, not a solution: the baseline scores
0.92 in about a second, a BatchNorm MLP with AdamW and a cosine schedule reaches
0.987 in 37 s. Have everyone submit the baseline in the first ten minutes, since a
submission takes a few minutes to settle. Where they stall: BatchNorm without
model.eval() in predict, a fixed epoch count that overruns the deadline on the
grading machine, and a renamed parameter that makes the upload validator reject
the file. -->

---

## Setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s4-building-blocks.ipynb)

- **One secret** in Colab's *Secrets* panel, granted to the notebook:
  `MLARENA_API_KEY` (ML-Arena, Profile → API Keys, starts with `mlk_user_`). Never
  in a cell.
- **Locally:** download the `.ipynb` from the GitHub path of the badge and
  `uv add torch torchvision mlarena-sdk`. The first run downloads MNIST into `./data/`.

The notebook trains on a Colab CPU on purpose: the challenge's agent runs on **3 CPU
cores**, and the notebook sets `torch.set_num_threads(3)` to measure like it.
