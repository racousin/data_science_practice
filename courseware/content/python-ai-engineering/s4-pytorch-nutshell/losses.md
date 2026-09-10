# Losses

Session 2 defined the loss as the average cost of the model's mistakes and chose
one per task. In PyTorch a loss is a scalar tensor computed from the predictions,
so a model can be trained on the objective the problem actually has, not only on
the ones a library ships.

<!-- notes: 6 minutes. Short on the built-ins: Session 2 taught them, so map
them to their PyTorch names and move on. The time goes to two ideas the lab
needs: a loss is any expression autograd can differentiate that reduces to one
number, and the pinball loss, whose asymmetry sends the prediction to a
quantile. Slow down on the gradient slide: it is the argument for why the lab's
promises are kept 9 times out of 10. If asked about the kink at 0, where
Session 2 called MAE not differentiable: autograd uses a value there
(torch.maximum splits a tie, giving -0.4 for this pinball loss at d = 0), and
exact ties are rare on real data. -->

---

## A loss is a scalar you can differentiate

```python
pred = torch.tensor([2., 4.])
y = torch.tensor([1., 7.])
((pred - y) ** 2).mean()          # tensor(5.): written by hand
nn.MSELoss()(pred, y)             # tensor(5.): the same, built in
```

- Any tensor expression reduced to one number can be a loss, if `backward()`
  can differentiate it ([lesson 3](/courses/python-ai-engineering/s4-pytorch-nutshell/course/autograd)).
  An error rate cannot: `>` and `argmax` return results with no `grad_fn`.
- A built-in loss is an object: create it once, `loss_fn = nn.MSELoss()`, then
  call it on every batch. `nn.L1Loss()` is the MAE: `tensor(2.)` here.
- PyTorch takes `(prediction, target)`; Session 2 wrote $L(y, \hat{y})$. Same
  function, arguments swapped: it matters for an asymmetric loss.
- Recall: the best constant is the mean under MSE, the median under MAE.

---

## Classification takes logits

A **logit** is a raw score, any real number (lesson 5). The **softmax** turns
$K$ of them into probabilities that sum to 1:

$$
\mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

```python
logits = torch.tensor([[2.0, 0.5, -1.0]])    # 1 sample, 3 classes
torch.softmax(logits, dim=1)    # tensor([[0.7856, 0.1753, 0.0391]])
label = torch.tensor([0])                    # a class index: int64
nn.CrossEntropyLoss()(logits, label)         # tensor(0.2413)
```

- `CrossEntropyLoss` applies the softmax itself: $-\log 0.7856 = 0.2413$.
  `BCEWithLogitsLoss` does the same with the sigmoid, for one logit and a
  float 0/1 target.
- So the model ends on a bare `Linear`. A softmax inside it would run twice: no
  error, but with three classes the loss could never fall below 0.55.
- For a batch of $B$ rows: logits $(B, K)$, labels $(B,)$ of int64 class
  indices, 0 to $K - 1$. As floats, the same indices fail:
  `expected target dtype to be Long or Byte, but got Float` (`Long` is int64).

---

## An uncommon objective: the pinball loss

The lab's task: **promise an arrival time the ride beats 9 times out of 10.**
Being late then costs more than padding the estimate, so the loss must be
asymmetric.

![Pinball loss at tau = 0.9 against the squared and absolute losses, as a function of the residual in minutes](assets/s4-pytorch-nutshell/losses/pinball-loss.png)

The pinball loss has one setting, $\tau$ between 0 and 1: a minute late costs
$\tau$, a minute early $1 - \tau$. With $\tau = 0.9$, that is 0.9 and 0.1. MSE
and MAE are symmetric: they cannot tell a late trip from an early one.

---

## The pinball loss, checked by hand

$$
L_\tau(y, \hat{y}) = \max(\tau\,(y - \hat{y}),\ (\tau - 1)\,(y - \hat{y}))
$$

```python
pred = torch.tensor([10., 10.])     # two promises of 10 minutes
y = torch.tensor([12., 7.])         # late by 2, early by 3
d = y - pred
torch.maximum(0.9 * d, -0.1 * d)    # tensor([1.8000, 0.3000])
```

`torch.maximum` keeps the larger of two tensors, element by element. Late by 2
costs $0.9 \times 2 = 1.8$, early by 3 costs $0.1 \times 3 = 0.3$; their
`.mean()`, `tensor(1.0500)`, is the loss. Written by hand, a loss checks no
shapes: give `pred` and `y` the same shape
([lesson 2](/courses/python-ai-engineering/s4-pytorch-nutshell/course/tensor-mechanics)).

---

## Why it lands on the 90th percentile

```python
pred = torch.tensor([10., 10.], requires_grad=True)
loss = torch.maximum(0.9 * (y - pred), -0.1 * (y - pred)).sum()
loss.backward()
pred.grad                           # tensor([-0.9000,  0.1000])
```

A descent step moves against the gradient: the late trip pulls its prediction
up with weight 0.9, the early trip pushes it down with 0.1 (`.mean()` would
halve both). For one constant $c$ fitted to many trips, the pulls cancel when
10% are late: per hundred trips, $0.9 \times 10 = 0.1 \times 90$.

$$
\arg\min_{c} \sum_i L_\tau(y_i, c) = q_\tau(y)
$$

The right-hand side is the $\tau$-quantile: the value below which a share $\tau$
of the $y_i$ lie. At $\tau = 0.5$ the pinball loss is half the MAE: Session 2's
median. A model trained on it predicts, for each input $x$, the 90th percentile
of trips like $x$.

---

## Check yourself

1. A classifier for 10 classes receives a batch of 32 rows. What shape is its
   output, and in which dtype and shape does `nn.CrossEntropyLoss` take the
   labels?

   **Answer.** The output is $(32, 10)$: logits, with no softmax. The labels are
   int64 class indices of shape $(32,)$, with values from 0 to 9.

2. Both promises below miss by 5 minutes. Run this: what does it print, and why
   do the two numbers differ?

   ```python
   pred = torch.tensor([20., 30.])
   y = torch.tensor([25., 25.])
   print(torch.maximum(0.9 * (y - pred), -0.1 * (y - pred)))
   ```

   **Answer.** `tensor([4.5000, 0.5000])`. The first trip arrives late and pays
   0.9 a minute, the second early and pays 0.1 a minute: nine times less.

3. An app wants its promises kept 3 times out of 4. Which $\tau$, and how does a
   minute late compare with a minute early?

   **Answer.** $\tau = 0.75$: the loss then aims at the 75th percentile. A
   minute late costs 0.75, a minute early 0.25: a third as much.
