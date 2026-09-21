# Regularization

A network can fit its training set perfectly and still be wrong on the next
image. The MLP of this lesson has 669,706 parameters; given 2,000 MNIST images it
reaches 100% training accuracy and 92.5% on the 10,000 images it has not seen.
Regularization is everything you do to raise the second number, usually at the
price of the first.

The lesson puts every regularizer of the session on one footing — same network,
same 2,000 images, three seeds each, every strength chosen on a validation set
and scored on a test set nothing was chosen on — and then asks when any of it
matters. The mechanics live elsewhere and are linked, not repeated: AdamW's
decoupled decay and the early-stopping restore in
[Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules),
augmentation in
[Data Pipelines and the Training Loop](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/data-pipelines-and-training-loop),
label smoothing in
[Making Training Work](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/making-training-work).
[Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout)
has the next lesson to itself.

Every figure marked *measured* comes from a run made while writing this lesson,
on MNIST, in torch 2.x on a laptop CPU.

<!-- notes: 35 minutes. Before the comparison slide, ask the room to rank the
regularizers. Weight decay and dropout usually come first; on this data they
are near the bottom, and the data-side ones are on top. The ηλT slide is the one they
will use. Session 3 derived ridge and lasso — point back to it, do not re-derive.
End on the training-set-size figure: regularization is a small-data tool. -->

---

## The gap is the quantity

![A flexible fit through five points, and a regularized one](assets/tabular/regularization-effect.png)

Session 3 drew this for a polynomial through five points. A network with
670,000 parameters and 2,000 images is the same picture in more dimensions: it
has enough freedom to pass through every training point, and nothing tells it
what to do in between.

![Training and generalization loss against model complexity, with underfitting on the left and overfitting on the right](assets/nn/d2l-capacity-vs-error.png)

*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, [Dive into Deep Learning](https://d2l.ai), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

---

## Read the two numbers together

Always side by side: accuracy on the data you trained on, and on data you did
not.

| Training | Validation | Diagnosis | What to change |
|---|---|---|---|
| low | low | underfitting | bigger model, longer training, *less* regularization |
| high | much lower | overfitting | regularization, or more data |
| high | close to it | done | nothing |

Regularize in the middle row only. Every technique in this lesson makes the
training fit worse on purpose; on a model that is not overfitting, that is all it
does.

Measure the training number in `eval()` mode on the un-augmented training
images. In `train()` mode, dropout and augmentation lower it by themselves, and a
gap that closes that way has closed on paper only.

---

## L2: a price on the size of the weights

$$
\tilde{\mathcal{L}}(w) = \mathcal{L}(w) + \frac{\lambda}{2}\|w\|_2^2
\qquad
\nabla\tilde{\mathcal{L}} = \nabla\mathcal{L} + \lambda w
$$

One step of SGD on the penalized loss:

$$
w \leftarrow w - \eta\,(\nabla\mathcal{L} + \lambda w) = (1 - \eta\lambda)\,w - \eta\nabla\mathcal{L}
$$

The penalty *is* a multiplicative shrink of every weight toward zero at every
step, which is why PyTorch calls the argument `weight_decay`. It is Session 3's
[ridge](https://ml-arena.com/courses/ms2a-machine-learning-practice/s3-tabular-models/the-tabular-landscape),
applied to every weight matrix of the network. The data has to pay, in loss, for
every weight it holds away from zero; weights no example needs decay away, and a
network with small weights is a smoother function of its input.

The equivalence of penalty and decay holds for SGD and breaks for Adam, which
divides the penalty's gradient by its running scale like any other gradient.
[Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules)
measured the consequence and the fix: `AdamW`, which shrinks the weight
directly, with biases and normalization parameters left out.

---

## How much weight decay, measured

![Training and validation accuracy, and validation loss, against AdamW's weight_decay, measured](assets/nn/regularization-weight-decay-sweep.png)

`AdamW` at `lr = 1e-3`, 3,000 steps, seven values of `weight_decay`. Up to 0.1
almost nothing moves: validation accuracy goes from 92.52% to 92.80%. The best
value is **1.0**, at 93.25%; at 3.0 accuracy is back to 92.20%, and at 10 the
network can no longer fit its own training set (97.6%) and falls to 91.23% —
the left side of the capacity figure, reached from the right.

The right panel is the larger effect. Validation loss **halves**, from 0.517 to
0.278, while accuracy moves by 0.7 points. An unregularized network that has
memorized its training set is confidently wrong on the test images it gets
wrong; decay shrinks the weights — their norm from 29.2 to 19.4 — and with them
the logits, so the wrong answers are less certain. If you are graded on a loss or
on calibrated probabilities, weight decay matters more than the accuracy column
suggests.

---

## Think in ηλT, not in λ

A best `weight_decay` of 1.0 looks absurd next to the usual 0.01–0.1. It is not,
because in `AdamW` the decay acts through the learning rate. On its own, it
multiplies each weight by $(1 - \eta\lambda)$ per step, so over $T$ steps by

$$
(1 - \eta\lambda)^T \approx e^{-\eta\lambda T}
$$

The exponent $\eta\lambda T$ is the quantity that transfers between runs:

| Run | $\eta$ | $\lambda$ | $T$ | $\eta\lambda T$ |
|---|---|---|---|---|
| this lesson, best | 1e-3 | 1.0 | 3,000 | 3 |
| this lesson, the "usual" value | 1e-3 | 0.01 | 3,000 | 0.03 |
| a long run with the usual value | 3e-4 | 0.1 | 100,000 | 3 |

At $\eta\lambda T = 0.03$ the decay removes 3% of an unused weight over the whole
run, which is why 0.01 did nothing measurable here. Under a schedule, replace
$\eta T$ by the sum of the learning rates. Treat this as a starting point for a
log-scale search, not a law: when you change the number of steps or the
learning rate, move $\lambda$ to keep the product, then search around it.

---

## L1: sparse in theory, jittery in practice

$$
\tilde{\mathcal{L}}(w) = \mathcal{L}(w) + \lambda\|w\|_1
\qquad
\nabla\tilde{\mathcal{L}} = \nabla\mathcal{L} + \lambda\,\operatorname{sign}(w)
$$

L2 pulls in proportion to the weight and never quite reaches zero; L1 pulls with
the same force whatever the size, and in Session 3's lasso that drove
coefficients to exactly zero. In a network it asks which *inputs* are worth a
weight at all.

![Norm of the first-layer weights leaving each of the 784 pixels, for no regularization, weight decay, an L1 penalty through Adam and a proximal L1 step, measured](assets/nn/regularization-l1-l2-first-layer.png)

Each image is the first layer seen from the input side: pixel $(r, c)$ is
coloured by the norm of the 512 weights that leave it. Without regularization,
the border — pixels that are black in every training image — keeps its random
initial weights forever: an input of exactly 0 gives those weights a gradient of
exactly 0, so nothing ever moves them. Weight decay shrinks them with everything
else, and 18.2% of the layer ends below $10^{-3}$.

The L1 penalty goes much further: 96.7% of the first layer ends below
$10^{-3}$, and the network still scores 92.9% on validation, better than with
no regularization. But not one weight is exactly zero. Through Adam, the
penalty's gradient $\lambda\operatorname{sign}(w)$ is normalized like any other,
so a weight with no data gradient is pushed toward zero at the full learning
rate, overshoots, and jitters around zero by about one step.

---

## Exact zeros: the proximal step

Exact zeros need the **proximal** step: take the optimizer step on the data loss
alone, then shrink every weight by $\eta\lambda$ and clip at zero.

```python
opt.step()
with torch.no_grad():                    # soft-threshold: exact zeros
    for w in weights:
        w.copy_(w.sign() * (w.abs() - lr * l1).clamp_min(0))
```

With $\lambda = 0.01$, 53.6% of the first layer ends at exactly zero — the
right-hand image, where the border is simply gone — at 93.1% validation
accuracy.

On accuracy, L1 ties with weight decay for the smallest gain of the lesson: 0.7
points at its best, $\lambda = 3 \times 10^{-6}$. Use it when you want the
zeros: to prune inputs or units, or to read which inputs a model uses.

---

## Noise: change the data, not the weights

Three regularizers act on what the network sees rather than on its parameters:

- **Augmentation** — a transformation you know does not change the label
  ([Data Pipelines and the Training Loop](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/data-pipelines-and-training-loop),
  and Session 5's
  [Training CNNs](https://ml-arena.com/courses/ms2a-machine-learning-practice/s5-computer-vision-1/training-cnns)).
  Here: random rotations of ±10°, shifts of ±2 pixels, scaling by 0.9–1.1.
- **Input noise** — $x + \sigma\varepsilon$, a new $\varepsilon$ at every step.
  For small $\sigma$ this is approximately a penalty on the network's gradient
  with respect to its input (Bishop, 1995): it rewards a function that is flat
  around each training point.
- **Label smoothing** — noise on the target
  ([Making Training Work](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/making-training-work)).

```python
xb = xb + sigma * torch.randn_like(xb)                  # training only
loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
```

On validation, input noise was best at $\sigma = 0.6$ on pixels in $[0, 1]$ —
95.71% — label smoothing at 0.3 (94.93%, and flat from 0.3 to 0.5), and
augmentation reached 96.83%. Each of the three beat weight decay and L1.

Input noise has one more property, which matters in Lab 4: it acts on each pixel
independently, so it **commutes with a pixel permutation**. On challenge 8, where
the pixels are permuted and spatial augmentation is meaningless, it is the
data-side regularizer still available.

---

## Early stopping, measured: it bought nothing here

Stopping after $t$ steps limits how far the weights can travel from their
initialization; for a linear least-squares model trained by gradient descent it
behaves approximately like an L2 penalty with $\lambda \approx 1/(\eta t)$. It is a
regularizer that costs nothing but a validation pass, and the mechanics — the
patience, and the restore that is the point — are in
[Optimization and Schedules](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/optimization-and-schedules).

On this problem it did nothing. The unregularized runs were evaluated every
16 steps; their validation *accuracy* never turned down. The best checkpoint came
at step 2,811 on average, 94% of the way through, and restoring it changed the
test accuracy by nothing. What turned was the *loss*: it bottomed at 0.30
around step 110 — epoch 7 — and climbed to 0.52 by the end, while accuracy rose
from 91.5% to 92.5% under it. Early stopping on the loss would have cost a
point.

Keep it on anyway, for the runs where the curve does turn — noisy labels, a high
learning rate, many more epochs than this — and stop on the metric you are graded
on.

---

## The comparison, measured

![Training and test accuracy for every regularizer at its best strength, and for two combinations, measured](assets/nn/regularization-comparison.png)

Every regularizer at its validation-best strength, scored on the 10,000 test
images; the small dots are the three seeds. Read the grey dots first: every
single regularizer except augmentation still fits the 2,000 training images to
99.97% or more. None of them stopped the network from memorizing — they changed
what it does *between* the training points.

Weight decay and dropout, the two usually reached for first, are near the
bottom. Early stopping bought nothing, L1 and weight decay 0.7 points, dropout
1.1.
Everything that acts on the data or on the target did better: label smoothing
2.1, input noise 3.0, augmentation 4.2. Augmentation, dropout and weight decay
together reached 97.1%, and adding label smoothing 97.7% — 5.1 points above
none, from four lines of code and no new data.

The ranking belongs to this problem — a small, clean image set with an obvious
invariance — not to regularizers in general. What transfers is the method: one
setting, every strength chosen on validation, every result on a test set nothing
was chosen on, several seeds.

---

## Data beats all of them

![Test accuracy against training-set size without regularization, with input noise, and with augmentation, dropout and weight decay, measured](assets/nn/regularization-vs-training-set-size.png)

The same network and the same strengths — chosen at 2,000 images — for training
sets from 500 to 50,000 images, 6,000 steps each.

Without regularization, test accuracy climbs from 86.9% at 500 images to 98.1%
at 50,000. The stack of augmentation, dropout and weight decay adds 8.2 points
at 500 images, 4.4 at 2,000, 0.8 at 10,000 and nothing at 20,000. At 50,000 it
*costs* 0.8 points: strengths chosen for 2,000 images keep the network from
fitting 50,000 — 97.2% on its own training set. Input noise follows the same
curve, lower: +3.6 at 500, +0.8 at 10,000, −0.1 at 50,000.

Two lessons in one figure. Regularization is a small-data tool, and its value
shrinks as the data grows: 500 regularized images (95.1%) are worth about as
much as 5,000 unregularized ones (95.0%). And its strength is tuned to a data
size — carry it to twenty-five times more data and it underfits.

---

## What this means for challenge 8

[1 Minute Permuted MNIST](https://ml-arena.com/viewchallenge/8) gives your agent
60,000 images and 60 seconds on three cores. That is the right-hand end of the
last figure, with a clock on top: the model is limited by how many steps it can
take, not by how much data it has. Lab 4's Part 4 measured it — dropout and
weight decay closed the train–validation gap from +0.029 to +0.009 and moved
validation accuracy from 0.9635 to 0.9659.

Spend the budget on steps. Dropout in particular slows convergence, which is
exactly the wrong trade under a clock. If you regularize at all, use what is
cheap and still valid after the permutation: input noise, a little weight decay,
label smoothing.

---

## An order to try them in

1. **Confirm the gap.** Training accuracy in `eval()` mode against validation.
   No gap, no regularization.
2. **More data**, or augmentation that encodes an invariance you are sure of.
   Nothing else on this page comes close when it applies.
3. **Weight decay** with `AdamW`, biases and norms excluded, $\lambda$ set from
   $\eta\lambda T$ and searched on a log scale.
4. **Label smoothing** for classification, 0.1 to start.
5. **Dropout** in the wide layers —
   [Dropout](https://ml-arena.com/courses/ms2a-machine-learning-practice/s4-advanced-neural-networks/dropout).
6. **Early stopping** with the restore, always on, as insurance.

Change one thing at a time and keep the run that justified it. Strengths chosen
one at a time are a starting point for a combination, not its answer: every
regularizer you add takes some of the work the others were doing.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn as nn

   def fit(opt_cls, weight_decay, penalty):
       torch.manual_seed(0)
       m = nn.Linear(20, 1)
       x, y = torch.randn(64, 20), torch.randn(64, 1)
       opt = opt_cls(m.parameters(), lr=0.1, weight_decay=weight_decay)
       for _ in range(100):
           l2 = sum(p.pow(2).sum() for p in m.parameters())
           loss = ((m(x) - y) ** 2).mean() + penalty * l2
           opt.zero_grad(); loss.backward(); opt.step()
       return m.weight

   for opt in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
       same = torch.allclose(fit(opt, 0.01, 0.0), fit(opt, 0.0, 0.005))
       print(opt.__name__, same)
   # -> SGD True
   # -> Adam True
   # -> AdamW False
   ```

   Why is the penalty 0.005 and not 0.01? Why is `Adam`'s `weight_decay` the
   same as the penalty, and why is that the problem `AdamW` exists to fix?

2. Your run uses `AdamW` at `lr = 3e-4` for 20,000 steps with
   `weight_decay = 0.01`. What is $\eta\lambda T$, what do you expect the decay
   to do, and which $\lambda$ would give the product that worked in this lesson?

3. A colleague adds dropout and reports that the train–validation gap fell from
   8 points to 2. They measured training accuracy in `train()` mode. What is
   wrong with the comparison, and what would you measure instead?

4. Without regularization, a first-layer weight attached to a pixel that is
   black in every training image never changes. Why? What does `AdamW`'s decay
   do to it, and why might that matter on a test image?

5. Which regularizers of this lesson can you use on challenge 8, where the
   pixel positions are permuted? Which can you not, and why?

6. Early stopping bought nothing on this problem. Describe a training run in
   which you would expect it to buy the most.
