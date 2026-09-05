# Training CNNs

Session 4 covered how to make a training loop converge. This lesson is about the
part that is specific to images, and it is almost entirely about the data: with
a few thousand labelled pictures, augmentation moves the validation number more
than any architecture or optimizer choice you can make.

<!-- notes: 30 minutes. Show a batch of augmented images on screen before
explaining any of it — half the room will spot an augmentation that destroys
their own label. The train/eval transform split is the slide that shows up in
the lab grading. -->

---

## Why augmentation dominates

A CNN with 25 million parameters trained on 5,000 images will reach 100%
training accuracy and memorise the set. Weight decay and dropout slow that down;
augmentation attacks it directly, by making the training set effectively larger
and by encoding what you already know.

Every augmentation is a claim: *this transformation does not change the label.*
That claim is domain knowledge, and it is the cheapest domain knowledge you will
ever inject into a model.

---

## Flip

![Horizontal and vertical flip](assets/cv/flip.png)

```python
transforms.RandomHorizontalFlip(p=0.5)
```

The single most effective augmentation on natural images, and free. Most objects
are equally plausible mirrored.

Vertical flip is a different claim entirely — it is right for satellite and
microscopy, wrong for anything photographed by a person standing up.

---

## Rotation

![Rotation](assets/cv/rotate.png)

```python
transforms.RandomRotation(degrees=15)
```

Small rotations model camera tilt. Keep them small for photographs — ±10 to ±15
degrees — because a 45-degree chair is not a chair anyone photographs. Full
360-degree rotation is correct only where there is no canonical orientation:
satellite tiles, cell images, astronomical plates.

Rotation fills the corners it exposes. Check what it fills them with; black
corners are a feature the network will happily learn.

---

## Scale and crop

![Scaling](assets/cv/scale.png)

```python
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
```

This is the workhorse. It picks a random sub-region covering 70–100% of the
area, then resizes it to 224 — so it varies scale, translation and framing in
one operation, and it is what every ImageNet recipe uses.

Push `scale` too low and you crop the object out of the picture while keeping
its label. That is manufactured label noise.

---

## Perspective and colour

![Perspective warp](assets/cv/perspective.png)

```python
transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
```

Perspective models viewpoint change — valuable for documents, signage and
anything photographed at an angle by a handheld camera.

Colour jitter models lighting and white balance. Keep `hue` small: 0.05 is a
different bulb, 0.5 is a different object.

---

## When each one is wrong

| Augmentation | Breaks |
|---|---|
| Horizontal flip | text and digits (`b`/`d`, `2`/`5`), medical laterality — a flipped chest X-ray moves the heart to the wrong side |
| Vertical flip | any photograph with a horizon in it |
| Large rotation | scene photographs, road signs, handwriting |
| Colour jitter | anything where colour *is* the label: ripeness, rust, skin lesions, traffic lights |
| Random crop | small objects, or any task where context is the signal |
| Grayscale | histopathology stains, mineral classification |

> An augmentation that changes the correct label is not regularization. It is
> label noise you paid compute to generate.

Look at fifty augmented images from your own pipeline before you train on them.
Every time.

---

## The train/eval split

```python
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                      T.RandomHorizontalFlip(), T.ToTensor(), normalize])
eval_tf  = T.Compose([T.Resize(256), T.CenterCrop(224),
                      T.ToTensor(), normalize])
```

Augmentation is a **training-time** operation. Evaluation must be deterministic:
the same image must produce the same prediction on every run, or your validation
curve is measuring dice.

Two dataset objects, two transforms, and the eval transform contains no `Random`
anything. Reusing `train_tf` for validation is the most common silent bug in
this session — it does not crash, it just makes validation noisy and pessimistic,
and it will cost you a model selection.

---

## mixup and cutmix

$$
\tilde{x} = \lambda x_a + (1 - \lambda) x_b
$$

**mixup** blends two images and their labels with the same $\lambda$ drawn from
a Beta distribution. **cutmix** pastes a rectangle of one image into another and
sets $\lambda$ to the pixel fraction.

```python
lam = np.random.beta(0.2, 0.2)
idx = torch.randperm(x.size(0))
out = model(lam * x + (1 - lam) * x[idx])
loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[idx])
```

Both are worth 1–2 points on ImageNet-scale runs and both reduce overconfidence.
Both need long schedules to pay off — on a 20-epoch fine-tune they usually make
things worse. Reach for them after the basics are exhausted.

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

1. You built one transform pipeline and passed it to both loaders. Validation
   accuracy sits four points below training from the first epoch and the gap
   never widens. Which bug is it, and why is it not overfitting?

   **Answer.** The evaluation transform is not deterministic — `train_tf` was
   reused for validation, so every validation image is randomly cropped and
   flipped. Overfitting opens a gap gradually; this one is there at epoch 1 and
   stays flat.

2. Run this. You should get exactly the output shown.

   ```python
   import numpy as np, torch
   import torchvision.transforms as T
   from PIL import Image
   img = Image.fromarray(np.random.RandomState(0).randint(0, 256, (256, 256, 3), dtype=np.uint8))
   train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.ToTensor()])
   eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
   torch.manual_seed(0)
   print(torch.equal(train_tf(img), train_tf(img)))   # -> False
   print(torch.equal(eval_tf(img), eval_tf(img)))     # -> True
   ```

3. You accumulate gradients over 16 micro-batches of 4 to reach an effective
   batch of 64. Does that repair BatchNorm, and what do you do instead?

   **Answer.** No. Each BatchNorm forward pass still sees only the 4 examples of
   its micro-batch, so the statistics are statistics of 4. Below batch 8, swap
   it for `nn.GroupNorm(32, C)`, which is batch-independent.
