# Autograd

Training needs the gradient of the loss with respect to every parameter.
Session 3 derived it by hand for an MLP, layer by layer. PyTorch derives it for
any code written with tensor operations: you write the forward pass, and the
backward pass is derived from it.

<!-- notes: 9 minutes. One running example for this lesson and the next: one
parameter w = 1, one input x = 2, one target y = 5, loss (wx - y)^2 = 9,
gradient -12. Everything is checkable by hand, which is the point. Leave the
chain-rule derivation to Session 3; here, show that the machine agrees with it.
Run the snippets live, in order, in one session: each builds on the previous.
Re-running the backward() cell on its own reproduces the last row of the error
table; worth doing once. If asked why gradients add up rather than overwrite:
so that several batches can contribute to one update. If asked why backward
costs about two forward passes: each matrix product of the forward pass needs
two in the backward pass, one for the weight's gradient and one to pass the
gradient on to the layer before. Measured on an Apple M4 CPU (torch 2.14.0,
4 threads): backward took 1.5x to 2.6x the forward time, over two runs, on
MLPs of 4 to 50 million parameters. -->

---

## What autograd does


![augmented_computational_graph.png](assets/s4-pytorch-nutshell/autograd/augmented_computational_graph.png)


1. While the code runs, PyTorch **records** each operation on a tensor that
   needs gradients: the computation graph.
2. `loss.backward()` walks that record **backwards**, applying the chain rule.
3. The gradient lands in `.grad` of every parameter, all of them in one pass.

---

## Mark the parameters; the graph records itself

```python
w = torch.tensor(1.0, requires_grad=True)   # a parameter: to learn
x = torch.tensor(2.0)                       # data: nothing to learn
pred = w * x              # tensor(2., grad_fn=<MulBackward0>)
loss = (pred - 5.0) ** 2  # tensor(9., grad_fn=<PowBackward0>)
```

- `requires_grad=True` marks what to learn; `x.requires_grad` is `False`. In
  [lesson 5](/courses/python-ai-engineering/s4-pytorch-nutshell/course/modules-and-optimizers),
  a network's layers set it on their weights for you.
- Each result remembers the operation that produced it, its `grad_fn`, the
  unnamed `pred - 5.0` included: that chain is the graph in the figure. It is
  rebuilt at every forward pass, so a model may contain ordinary Python `if`s
  and loops.

---

## `backward()` applies the chain rule

```python
loss.backward()           # from loss back to w
w.grad, x.grad            # (tensor(-12.), None)
```

With $\ell = (wx - y)^2$ and the target $y = 5$:

$$
\frac{\partial \ell}{\partial w} = 2(wx - y)\,x = 2(2 - 5)(2) = -12
$$

The machine and the hand calculation agree. A **leaf** is a tensor created
directly, such as `w` or `x`, rather than by a recorded operation. Gradients
are kept only on leaves with `requires_grad=True`: `x` gets none, and the
intermediate `pred` keeps none; its gradient is used during the pass, then
dropped.

---

## `backward()` needs one number

```python
v = torch.tensor([1., 2., 3.], requires_grad=True)
(v ** 2).sum().backward()
v.grad                    # tensor([2., 4., 6.])
```

- On `v ** 2`, three numbers, `backward()` fails:
  `grad can be implicitly created only for scalar outputs`. A loss is one
  number, a mean over the batch.
- From that one number, one backward pass gives the gradient of every
  parameter (reverse mode), for about the cost of two forward passes, whether
  the model has one parameter or millions.
- The price is memory: the intermediate values the backward pass needs are
  kept until `backward()` has used them.

---

## Gradients accumulate

```python
w = torch.tensor(1.0, requires_grad=True)
for _ in range(2):
    ((w * x - 5.0) ** 2).backward()
    print(w.grad)         # tensor(-12.)  then  tensor(-24.)
```

`backward()` **adds** to `.grad`; it does not replace it. Between two updates
the gradient must be reset: `w.grad = None` by hand, or the optimizer's
`zero_grad()` for every parameter at once
([next lesson](/courses/python-ai-engineering/s4-pytorch-nutshell/course/optimizers)).

---

## Turning recording off

```python
with torch.no_grad():
    pred = w * x                     # no graph recorded
pred.requires_grad, pred.grad_fn     # (False, None)
```

- `torch.no_grad()`: for code whose gradient is never taken, such as
  validation, prediction, and the parameter update itself.
- Nothing inside it is recorded, so no intermediate values are kept in memory
  for a backward pass.

---

## One step by hand

```python
w = torch.tensor(1.0, requires_grad=True)
((w * x - 5.0) ** 2).backward()     # w.grad is tensor(-12.)
with torch.no_grad():
    w -= 0.1 * w.grad               # 1 - 0.1 * (-12) = 2.2
```

Session 2's update, $w \leftarrow w - \eta \partial \ell / \partial w$ with
$\eta = 0.1$; `w` is now `tensor(2.2000, requires_grad=True)`. The update runs
under `no_grad` because it is not part of the model, and PyTorch refuses an
in-place change (`-=` overwrites `w` itself) to a leaf that requires grad while
it records. The next lesson packages the update and the reset as an optimizer.
