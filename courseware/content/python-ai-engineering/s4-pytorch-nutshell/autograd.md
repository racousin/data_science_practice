# Autograd

PyTorch computes derivatives for you. This lesson is what it actually does,
enough that you can debug it — not the mathematics, which is in
[Reference → Autograd, the Mathematics](../paie-reference/autograd-mathematics).

<!-- notes: 35 minutes. The gradient-accumulation slide is the one that fixes
real bugs. Do not skip zero_grad. -->

---

## The problem

To train, you need $\frac{\partial L}{\partial \theta}$ for every parameter. A
network has millions of them, and the function is a composition of hundreds of
operations.

Deriving that by hand, once, is a research paper. Deriving it again after you
change a layer is not viable.

---

## What autograd does

While your forward pass runs, PyTorch records every operation into a graph.
Calling `.backward()` walks that graph in reverse, applying the chain rule at
each node.

You write the forward pass. The backward pass is derived from it.

---

## requires_grad

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

y.backward()
print(x.grad)        # tensor([4.])
```

$y = x^2$, so $\frac{dy}{dx} = 2x = 4$ at $x = 2$. Correct, and you wrote no
derivative.

Only tensors with `requires_grad=True` accumulate gradients. Model parameters
get it automatically; your input data should not have it.

---

## The graph

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = torch.sin(y)

print(y.grad_fn)     # <PowBackward0 object>
print(z.grad_fn)     # <SinBackward0 object>
```

Each result carries a `grad_fn` — the function that knows how to reverse that
step. Chained together, they are the computational graph.

```text
x --[Pow]--> y --[Sin]--> z
   <--------    <--------
     backward pass
```

---

## Leaf vs intermediate

```python
x.is_leaf        # True  — you created it
y.is_leaf        # False — it came from an operation
```

After `backward()`, `.grad` is populated on **leaves only**. `y.grad` is `None`,
and PyTorch warns you if you ask — intermediate gradients are computed, used, and
discarded.

To keep one:

```python
y.retain_grad()
```

---

## backward() needs a scalar

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2

y.backward()          # RuntimeError: grad can be implicitly created
                      # only for scalar outputs
y.sum().backward()    # fine
```

"The gradient of a vector" is a Jacobian, not a vector. Your loss is always
reduced to one number — which is why `loss = criterion(...)` returns a scalar.

---

## Gradients accumulate

This is the behaviour that bites everybody once.

```python
x = torch.tensor([2.0], requires_grad=True)

(x ** 2).backward()
print(x.grad)        # tensor([4.])

(x ** 2).backward()
print(x.grad)        # tensor([8.])  — added, not replaced
```

`.backward()` **adds** to `.grad`. Without a reset, batch 2's gradient contains
batch 1's, batch 3's contains both, and your model diverges for no visible
reason.

---

## Which is why every loop has this

```python
optimizer.zero_grad()    # clear last step's gradients
loss.backward()          # compute this step's
optimizer.step()         # apply them
```

Forgetting `zero_grad()` produces a model that trains badly rather than one that
crashes. That is the worst kind of bug, and it is why the three lines are always
written together.

<!-- notes: Ask them what accumulation is *for* — gradient accumulation across
micro-batches to simulate a larger batch. It is a feature, not an oversight. -->

---

## Turning it off

At inference you do not need gradients, and building the graph costs memory and
time.

```python
with torch.no_grad():
    predictions = model(x_test)
```

Roughly halves memory use during evaluation. Always wrap your validation loop.

---

## detach()

Take a tensor out of the graph:

```python
y = x ** 2
z = y.detach()       # same values, no history, requires_grad=False
```

Use it when storing a value for logging or for numpy:

```python
losses.append(loss.detach().cpu().item())
```

Appending `loss` itself keeps its whole graph alive. Do that in a loop and you
leak memory until the process dies — a real and common bug.

---

## `.item()`

```python
loss.item()          # Python float from a 1-element tensor
```

Use it for anything you print, log, or compare. It also detaches, which is why
the logging idiom above is safe.

---

## Worked example — one gradient step by hand

```python
w = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([2.0])
target = torch.tensor([5.0])

pred = w * x                      # 2.0
loss = (pred - target) ** 2       # 9.0
loss.backward()

print(w.grad)                     # tensor([-12.])
```

Check it: $L = (wx - t)^2$, so $\frac{\partial L}{\partial w} = 2(wx - t)x
= 2(2 - 5)(2) = -12$. The gradient is negative, so increasing $w$ decreases the
loss — which is right, since $w$ needs to reach 2.5.

---

## Applying it

```python
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()
```

`no_grad` because the update itself is not part of the model. This is exactly
what `optimizer.step()` does — next lesson replaces these three lines with one.

---

## Debugging autograd

| Symptom | Cause |
|---|---|
| `.grad` is `None` | not a leaf, or `requires_grad=False`, or `backward()` never ran |
| "element 0 does not require grad" | the graph was broken — a `.detach()`, a `.numpy()`, or an in-place op |
| Loss decreases then explodes | missing `zero_grad()` |
| Memory grows every epoch | storing tensors that still carry a graph |
| "backward through the graph a second time" | two `backward()` calls on one graph — pass `retain_graph=True`, or restructure |

---

## Recap

1. `requires_grad=True` starts recording.
2. `.backward()` on a **scalar** walks the graph in reverse.
3. Gradients land on **leaves** and **accumulate** — reset every step.
4. `no_grad()` for inference, `detach()` for logging.
