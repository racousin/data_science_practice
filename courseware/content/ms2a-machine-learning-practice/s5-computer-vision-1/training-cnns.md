# Training CNNs

Session 4 covered how to make a training loop converge, and the previous lesson
covered augmentation. This one is the rest of what is specific to images:
imbalanced classes, small batches, noisy labels, and a recipe to start from.

<!-- notes: 20 minutes. The BatchNorm slide is the one students get wrong in the
lab: they reach for gradient accumulation and expect it to fix small batches. -->

---

## Class imbalance

Vision datasets are rarely balanced: 95% healthy scans, 3 images of the rarest
species. Accuracy on such a set is a measure of the majority class and nothing
else.

```python
w = 1.0 / np.bincount(train_labels)
sampler = WeightedRandomSampler(w[train_labels], len(train_labels))
```

| Approach | When |
|---|---|
| Weighted sampler | moderate imbalance, plenty of data |
| Class-weighted cross-entropy | strong imbalance, cannot afford to over-sample |
| Focal loss | extreme imbalance, mostly detection |
| Collect more of the rare class | always better than any of the above |

Report **per-class recall** and macro-F1. Session 3's model-selection lesson
applies unchanged; only the data type is different.

---

## Batch norm and small batches

BatchNorm estimates the mean and variance of each channel *from the current
batch*. With batch size 4 those estimates are noise, training destabilises, and
the running statistics used at eval time no longer match anything.

- batch ≥ 32 — BatchNorm is fine
- batch 8–32 — acceptable, watch the train/eval gap
- batch < 8 — switch to `nn.GroupNorm(32, C)`, which is batch-independent

Gradient accumulation does **not** fix this. It accumulates gradients across
micro-batches, but each BatchNorm forward pass still sees only the micro-batch.
If you accumulate to reach an effective batch of 64 from micro-batches of 4,
your normalization statistics are still statistics of 4.

---

## Label noise

Web-scraped and crowd-labelled image sets carry 3–10% wrong labels routinely.
The symptom is a training loss that keeps falling while validation accuracy
plateaus early and then degrades.

- **Label smoothing** (`nn.CrossEntropyLoss(label_smoothing=0.1)`) — cheap,
  almost always helps, stops the network being certain about a wrong label
- **Early stopping** — networks fit the clean majority first and memorise the
  noise later, so stopping early is itself a denoiser
- **Look at the highest-loss training examples.** Twenty minutes with the top 50
  tells you whether you have a model problem or a labelling problem — and the
  second is more common than students expect. No hyperparameter fixes it.

---

## A recipe with numbers

| Knob | Value |
|---|---|
| Resolution | 224, unless the objects are small |
| Batch size | 32 or 64, the largest that fits |
| Optimizer | AdamW, `weight_decay=0.05` |
| Learning rate | 3e-4 from scratch, 1e-4 fine-tuning |
| Schedule | cosine decay, 3 warm-up epochs |
| Epochs | 30–50 from scratch, 10–15 fine-tuning |
| Augmentation | `RandomResizedCrop` + `HorizontalFlip` + mild `ColorJitter` |
| Loss | cross-entropy, `label_smoothing=0.1` |
| Precision | `torch.autocast` mixed precision — ~2× throughput |
| Stopping | best validation checkpoint, patience 10 |

Start here, change one thing at a time, and write down what each change was
worth. A run you cannot attribute is a run you wasted.

---

## Check yourself

1. You accumulate gradients over 16 micro-batches of 4 to reach an effective
   batch of 64. Does that repair BatchNorm, and what do you do instead?

   **Answer.** No. Each BatchNorm forward pass still sees only the 4 examples of
   its micro-batch, so the statistics are statistics of 4. Below batch 8, swap
   it for `nn.GroupNorm(32, C)`, which is batch-independent.
