# Lab 5 — Classify with a Pretrained Backbone

Fine-tune a pretrained CNN on a small image dataset, against a from-scratch
baseline, and say in numbers what the pretraining was worth.

<!-- notes: They will all want to skip the baseline. Do not let them — it is 20%
of the grade and it is the only thing that makes the headline number mean
anything. Have a GPU runtime ready for the room — it is not optional here.
On a laptop CPU one 13-epoch ResNet-18 fine-tune at 128 pixels is about
seventeen minutes, and Parts B and C are two of those before the ablation even
starts. -->

---

## Starter notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s5-blood-cells.ipynb)

[Challenge 173](https://ml-arena.com/viewchallenge/173), *Blood Cell Classification*:
8 cell types, 28×28 RGB images, scored by F1-macro.
