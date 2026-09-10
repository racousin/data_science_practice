# Losses

In PyTorch a loss is a scalar tensor computed from the predictions,
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
  can differentiate it.
  An error rate cannot: `>` and `argmax` return results with no `grad_fn`.
- A built-in loss is an object: create it once, `loss_fn = nn.MSELoss()`, then
  call it on every batch. `nn.L1Loss()` is the MAE: `tensor(2.)` here.

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

![softmax.webp](assets/s4-pytorch-nutshell/losses/softmax.webp)


---

## An uncommon objective: the pinball loss


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
