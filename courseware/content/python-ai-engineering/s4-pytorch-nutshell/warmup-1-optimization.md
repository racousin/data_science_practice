# Warm-up 1 — Optimization & Training

A runnable notebook, not a lecture. It does by hand, on one number, what the
last four lessons described — and then shows that `torch.optim` is doing the
same arithmetic on a few thousand parameters instead of one.

**Do it before the challenges.** It needs no ML-Arena account and no API key:
`torch` and `matplotlib` already ship with Colab.

[Open in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-ai-engineering/challenges/aie-s4-optimization-warmup.ipynb)

<!-- notes: 45 minutes, hands on keyboards. Run Part 1 together to the end of
the "one step in detail" cell, then let them loose on Exercise 1 — the learning
rate is the parameter they will fight all year and five minutes of watching it
diverge is worth more than a slide about it. Adapted from the SCAI-4EU AI in
Medicine workshop; the plotting helpers are inlined so there is nothing to
install. -->

---

## What it covers

Three parts, each one strictly contained in the next:

| part | you write | the point |
|---|---|---|
| 1 | gradient descent, as a `for` loop over one number | the update rule, and what the learning rate trades off |
| 2 | linear regression in PyTorch | the five lines every training step is made of |
| 3 | an MLP on data no line can fit | why the activation function is the model |

Part 1 minimises $f(\theta) = (3\theta - 7)^2$, whose minimum you can find on
paper. That is the point of choosing it: knowing the answer in advance is how
you tell a working optimizer from a broken one.

---

## The rule, once more

$$
\theta_{new} = \theta_{old} - \eta \cdot \nabla f(\theta_{old})
$$

The minus sign is the whole idea — the gradient points uphill, so you subtract
it. Everything in Session 4 is this line applied to more numbers at once.

---

## The five lines

Part 2 prints the state of the model between each of them. Every training loop
in the rest of the course is these five, in this order:

```python
y_pred = model(X)             # 1. forward
loss = loss_fn(y_pred, y)     # 2. how wrong
optimizer.zero_grad()         # 3. clear last step's gradients
loss.backward()               # 4. differentiate
optimizer.step()              # 5. update
```

Step 3 is the one people omit, and it does not raise. It just trains badly,
because each step then uses the sum of every gradient so far.

---

## What you should be able to explain afterwards

Not run — **explain**:

1. Why the update rule has a minus sign in it.
2. What too small and too large a learning rate each look like on a loss curve.
3. What happens if you delete `optimizer.zero_grad()`.
4. Why `torch.no_grad()` belongs around evaluation.
5. Why removing every `nn.ReLU()` collapses a three-layer network into a
   one-layer one.

If any of those is shaky, re-run the cell that shows it. The challenge notebook
assumes all five.
