# Optimizers

An optimizer is the update rule of the
[autograd lesson](/courses/python-ai-engineering/s4-pytorch-nutshell/course/autograd),
packaged: it holds the parameters, reads their `.grad`, and moves them. One
call, `opt.step()`, replaces the hand-written step, for one parameter or for
every weight of a network.

<!-- notes: 6 minutes. Same running example as the autograd lesson: w = 1,
x = 2, target y = 5, loss (2w - 5)^2 = 9, gradient -12, best w = 2.5, so every
number on these slides can be checked by hand. Run the blocks live, in order,
in one session: each continues from the previous one. Slow down on the three
lines and their order: every training loop in the course contains them. The
learning-rate table is Session 2's picture, measured. Momentum and Adam get one
slide each: the declaration and the mechanism in a sentence, no derivation. The
hats on Adam's m and v correct for their start at zero; say so if asked, no
further. Numbers measured with torch 2.14.0 on CPU. The MLP gradient sizes on
the Adam slide: mean |grad| per tensor after one MSE backward on 256 rows of
randn inputs, targets 15 + 5 randn; seeds 0 to 2 give 0.07-0.09 on the first
weights and 29.5-30.2 on the output bias. -->

---

## Declaring an optimizer

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \, \ell(\theta_t)
$$

```python
w = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)      # theta = [w], eta = 0.1
```

- Two arguments: the **tensors** to update, as a list, and the learning rate
  `lr`. In [lesson 5](/courses/python-ai-engineering/s4-pytorch-nutshell/course/modules-and-optimizers)
  the list becomes every weight of a network.
- Always pass `lr`: its default, 0.001, is the same for every problem.
- `SGD` is Session 2's gradient descent. Its S, *stochastic*, refers to
  computing each step on a random sample of the data (lesson 7).
- The optimizer holds **references**, not copies: `step()` changes `w` itself.
  A tensor created after the optimizer, even under the same name, is not in its
  list: it never moves, and nothing warns you.

---

## `step()` is the update you wrote by hand

```python
((w * 2.0 - 5.0) ** 2).backward()  # w.grad = -12
opt.step()                         # w = 1 - 0.1 * (-12)
w                                  # tensor(2.2000, requires_grad=True)
```

For every tensor it holds, `step()` runs the autograd lesson's
`w -= lr * w.grad` under `torch.no_grad()`: here $1 - 0.1 \times (-12) = 2.2$.
It reads `.grad`, never the loss, so it does nothing before a `backward()`: a
tensor whose `.grad` is `None` is skipped.

---

## `zero_grad()` resets the gradients

```python
w.grad                 # tensor(-12.): step() does not clear it
opt.zero_grad()
w.grad is None         # True
```

`backward()` **adds** to `.grad` (autograd lesson). Without the reset, the next
`backward()` at $w = 2.2$ would leave $-12 - 2.4 = -14.4$ in `.grad` instead of
this step's $-2.4$: each step would use the sum of every gradient so far.
`zero_grad()` resets `.grad` to `None` for every tensor the optimizer holds.

---

## The three lines, in this order

```python
for _ in range(20):
    opt.zero_grad()                    # 1. forget the last gradient
    ((w * 2.0 - 5.0) ** 2).backward()  # 2. this step's gradient
    opt.step()                         # 3. update w
```

- Continuing from $w = 2.2$: 2.44, 2.488, 2.4976, …; from the 10th step on,
  `w` is exactly 2.5, the minimum.
- Delete line 1 and rerun from a fresh $w = 1$ and optimizer: 2.2, 3.64, 4.168,
  3.362, 1.866, …; after 20 steps $w = 0.844$ and the loss is 10.96, above the
  starting 9.0. No error is raised.
- In a network only line 2 grows: a forward pass, then a loss (lessons 5–7).

---
