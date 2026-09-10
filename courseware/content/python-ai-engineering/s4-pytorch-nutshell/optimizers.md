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

## The learning rate is the hyperparameter

Ten steps of the loop from a fresh $w = 1$; the minimum is $w = 2.5$.

| `lr` | $w$ after 10 steps | loss | what happens |
|---|---|---|---|
| 0.001 | 1.116 | 7.66 | too small: barely moves |
| 0.01 | 1.848 | 1.70 | slow |
| 0.1 | 2.500 | 0.00 | converged |
| 0.2 | 2.491 | 0.0003 | overshoots every step, converges |
| 0.25 | 1.000 | 9.00 | bounces between 4 and 1 |
| 0.3 | −40.888 | 7530 | diverges |

The gradient is $8(w - 2.5)$, so each step multiplies the distance to 2.5 by
$1 - 8\eta$: the distance shrinks only while $\eta < 0.25$. It is Session 2's
argument on $\ell(\theta) = \theta^2$, whose factor was $1 - 2\eta$. On a
network the symptoms are the same: a loss that barely moves, or `nan`.

---

## Momentum: steps that build up

```python
opt = torch.optim.SGD([w], lr=0.1, momentum=0.9)
```

- For each weight the optimizer keeps a decaying sum of past gradients,
  $m \leftarrow \beta m + g$ ($g$: this step's gradient), and steps by
  $\eta m$. `momentum` is $\beta$.
- While the slope keeps its sign, steps grow: on a constant slope of 1 (the
  loss $\ell = w$), this optimizer moves $w$ by 0.1, 0.19, then 0.271; plain
  SGD moves it by 0.1 each time. When the slope changes sign, the terms partly
  cancel.
- Session 2 wrote $m \leftarrow \beta m + (1 - \beta) g$. PyTorch drops the
  $(1 - \beta)$, so at the same `lr` every one of its steps is
  $1/(1 - \beta) = 10$ times Session 2's.

---

## Adam: a step size for each weight

```python
opt = torch.optim.Adam([w], lr=1e-3)
```

$$
\theta \leftarrow \theta - \eta \, \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

- $m$ is Session 2's momentum average of the gradient ($\beta = 0.9$), $v$ the
  same average of its square ($\beta = 0.999$). The hats correct both for
  starting at zero; $\epsilon = 10^{-8}$ avoids a division by zero.
- Dividing by $\sqrt{\hat v}$ cancels the gradient's scale: each weight moves
  by about `lr` per step. Two weights with gradients 100 and 0.01, one step at
  `lr=0.1`: SGD moves them by 10 and 0.001, Adam moves both by 0.1.
- A network needs this. On a fresh MLP 12 → 64 → 64 → 1 (MSE, random inputs,
  targets near 15), the average gradient size is about 0.08 on the first
  layer's weights and 30 on the output bias. Adam at `lr=1e-3`, its default,
  is the usual starting point.
- The cost: $m$ and $v$ are two extra numbers per weight, the memory counted in
  [lesson 1](/courses/python-ai-engineering/s4-pytorch-nutshell/course/why-tensors).

---

## Check yourself

1. Put these in order for one training step: `opt.step()`, `loss.backward()`,
   `opt.zero_grad()`.

   **Answer.** `opt.zero_grad()`, `loss.backward()`, `opt.step()`. The reset may
   also come right after `step()`; what matters is that it runs between two
   `backward()` calls, and that `step()` comes after `backward()`.

2. After these four lines, what is `w`?

   ```python
   w = torch.tensor(0.0, requires_grad=True)
   opt = torch.optim.SGD([w], lr=0.5)
   ((w - 4.0) ** 2).backward()
   opt.step()
   ```

   **Answer.** `tensor(4., requires_grad=True)`. The gradient at $w = 0$ is
   $2(w - 4) = -8$, and $0 - 0.5 \times (-8) = 4$: one step lands on the
   minimum.

3. At `lr=0.3`, ten steps on the toy loss end at a loss of 7530. Below which
   learning rate does it converge, and why?

   **Answer.** 0.25. Each step multiplies the distance to 2.5 by $1 - 8\eta$,
   which shrinks it only while $|1 - 8\eta| < 1$, that is $0 < \eta < 0.25$. At
   exactly 0.25, $w$ bounces between 4 and 1.
