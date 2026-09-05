# CNN Backpropagation — the Mathematics

Reference only — not covered in class, because autograd computes all of this
and you can build a working CNN without ever seeing it. Read it if you want to
know what the framework is doing, or if you are about to write a custom layer
whose backward pass you have to supply yourself. Session 5 covers the forward
convolution this differentiates.

<!-- notes: Self-study, linked from Session 5. Never lectured. -->

---

## Notation

Zero padding, stride 1, one training example. Other configurations only change
the indexing.

| Symbol | Meaning |
|---|---|
| $a_{i,j,d}$ | activation at position $(i,j)$ in input channel $d$ |
| $w_{p,q,d,k}$ | kernel weight at offset $(p,q)$, input channel $d$, output channel $k$ |
| $b_k$ | bias of output channel $k$ |
| $z_{i,j,k}$ | pre-activation of output channel $k$ at $(i,j)$ |
| $\delta_{i,j,k}$ | the error signal, $\partial L / \partial z_{i,j,k}$ |

What frameworks call *convolution* is cross-correlation: the kernel is not
flipped. The flip reappears on its own in the backward pass, which is the
interesting part of this page.

---

## Forward

$$
z_{i,j,k} = b_k + \sum_{d} \sum_{p=0}^{f_h-1} \sum_{q=0}^{f_w-1} w_{p,q,d,k} \, a_{i+p,\,j+q,\,d}
$$

Then $a^{out}_{i,j,k} = g(z_{i,j,k})$ for an activation $g$.

Two properties drive everything below. **Weight sharing**: the same weight
appears at every output position, so its gradient is a sum over all of them.
**Local connectivity**: each input appears in only a small window of outputs,
so its gradient is a sum over that window and nothing else.

---

## The error signal

Assume the layer above has already handed you

$$
\delta_{i,j,k} = \frac{\partial L}{\partial z_{i,j,k}} = \frac{\partial L}{\partial a^{out}_{i,j,k}} \cdot g'(z_{i,j,k})
$$

$\delta$ has the shape of the layer's *output*. Everything below follows from it
by the chain rule, and each derivation answers one question: which $z$ did this
quantity contribute to?

---

## Gradient with respect to the kernel

$w_{p,q,d,k}$ contributes to every output position, so sum over all of them:

$$
\frac{\partial L}{\partial w_{p,q,d,k}} = \sum_{i,j} \delta_{i,j,k} \, a_{i+p,\,j+q,\,d}
$$

Read the right-hand side as an operation on two images and it is a
cross-correlation of the input map with the error map, evaluated at offset
$(p,q)$: the backward pass of a convolution is another convolution, with the
error map playing the role of the kernel. The bias gradient is the same sum
with the input factor removed:

$$
\frac{\partial L}{\partial b_k} = \sum_{i,j} \delta_{i,j,k}
$$

---

## Gradient with respect to the input

$a_{i,j,d}$ enters $z_{p',q',k}$ whenever $i = p' + p$ and $j = q' + q$ for some
kernel offset $(p,q)$. Substituting $p' = i - p$:

$$
\frac{\partial L}{\partial a_{i,j,d}} = \sum_{k} \sum_{p,q} \delta_{i-p,\,j-q,\,k} \, w_{p,q,d,k}
$$

with out-of-range $\delta$ treated as zero. The minus signs are the signature of
a true convolution: this is a **full convolution** of $\delta$ with $w$, or
equivalently a cross-correlation of $\delta$ with the kernel rotated by 180°,
padded so the output regains the input's spatial size. That is the whole
content of "the backward pass of a convolution is a convolution with the
flipped kernel" — not a convention, but a consequence of the substitution.

---

## A worked example

Input $3 \times 3$, kernel $2 \times 2$, no padding, stride 1, bias 0.

```python
a = torch.tensor([[[[1., 2, 3], [4, 5, 6], [7, 8, 9]]]], requires_grad=True)
w = torch.tensor([[[[1., 0], [0, -1]]]], requires_grad=True)
z = F.conv2d(a, w)          # [[-4, -4], [-4, -4]]
```

Each output is a top-left minus bottom-right difference over its window, which
is $-4$ everywhere on this input. Take the incoming error to be
$\delta = [[1, 2], [3, 4]]$.

The kernel gradient sums $\delta$ against the input shifted by $(p,q)$. At
$(p,q) = (0,0)$ that is $1(1) + 2(2) + 3(4) + 4(5) = 37$. The input gradient is
the full convolution of $\delta$ with $w$: corner $(0,0)$ touches only
$\delta_{0,0} w_{0,0} = 1$, while centre $(1,1)$ collects $4(1) + 1(-1) = 3$.

```python
z.backward(torch.tensor([[[[1., 2], [3, 4]]]]))
w.grad     # [[37, 47], [67, 77]]
a.grad     # [[1, 2, 0], [3, 3, -2], [0, -3, -4]]
```

Every number there is reproducible by hand from the two formulas above. Do it
once; it is worth more than re-reading the derivation.

---

## Pooling

Max pooling has no parameters, so there is only an input gradient. Let
$m(p,q)$ be the position that won the max in output window $(p,q)$, and write
$[\cdot]$ for an indicator that is 1 when its condition holds:

$$
\frac{\partial L}{\partial a_{i,j}} = \sum_{p,q} \delta_{p,q} \, [\,(i,j) = m(p,q)\,]
$$

The layer is a **router**: the winner takes the whole gradient, every other
position in the window gets zero. This is why the argmax indices must be stored
during the forward pass — `nn.MaxPool2d(return_indices=True)` exposes them.
Average pooling instead splits the gradient evenly:

$$
\frac{\partial L}{\partial a_{i,j}} = \frac{1}{k_h k_w} \, \delta_{p,q}
$$

---

## Stride and padding

Stride $s$ changes nothing structurally: it inserts $s-1$ zeros between the
elements of $\delta$ before the backward convolutions, because only every
$s$-th output position exists. Padding $p$ crops $p$ rows and columns off the
input gradient at the end. Both are index bookkeeping — worth knowing only so
that a wrong output *shape* points you at the right suspect.

---

## Verify, do not trust

If you ever write a backward pass by hand, check it numerically before you
check it against a loss curve.

```python
from torch.autograd import gradcheck

x = torch.randn(1, 2, 6, 6, dtype=torch.float64, requires_grad=True)
w = torch.randn(3, 2, 3, 3, dtype=torch.float64, requires_grad=True)
assert gradcheck(MyConv.apply, (x, w), eps=1e-6, atol=1e-4)
```

`float64` is mandatory. In `float32` the finite-difference estimate is noise at
the tolerance that matters, `gradcheck` fails on correct code, and you spend an
afternoon debugging arithmetic that was right.

**Failure mode.** A wrong backward pass rarely crashes. The model trains, the
loss goes down a little, and the result is quietly 5% worse than it should be
forever. `gradcheck` costs two lines and catches it immediately.

---

## What to take away

- The backward pass of a convolution is two more convolutions: one against the
  input for the weights, one against the flipped kernel for the input.
- Weight sharing becomes a sum over positions — that sum *is* the gradient
  accumulation you see in the profiler.
- Max pooling routes the gradient to the argmax and discards the rest.
- Every claim above is checkable with `gradcheck` in `float64`.

---

## Check yourself

1. The lesson derives the input gradient as a sum over $\delta_{i-p,\,j-q,\,k}$.
   Where does the famous "flipped kernel" come from — a convention, or
   something else?

   **Answer.** Something else. It falls out of the substitution $p' = i - p$
   when you ask which pre-activations a given input contributed to. The minus
   signs make the expression a full convolution of $\delta$ with $w$, i.e. a
   cross-correlation with the kernel rotated 180°.

2. Run this. You should get exactly the output shown.

   ```python
   import torch, torch.nn.functional as F
   a = torch.tensor([[[[1., 2, 3], [4, 5, 6], [7, 8, 9]]]], requires_grad=True)
   w = torch.tensor([[[[1., 0], [0, -1]]]], requires_grad=True)
   z = F.conv2d(a, w)
   print(z.squeeze().tolist())        # -> [[-4.0, -4.0], [-4.0, -4.0]]
   z.backward(torch.tensor([[[[1., 2], [3, 4]]]]))
   print(w.grad.squeeze().tolist())   # -> [[37.0, 47.0], [67.0, 77.0]]
   print(a.grad.squeeze().tolist())   # -> [[1.0, 2.0, 0.0], [3.0, 3.0, -2.0], [0.0, -3.0, -4.0]]
   ```

   **Answer.** Both grids come from the two formulas above: 37 is
   $1(1) + 2(2) + 3(4) + 4(5)$ at offset $(0,0)$, and the centre of `a.grad` is
   $4(1) + 1(-1) = 3$.

3. You wrote a custom layer's backward pass by hand and `gradcheck` fails.
   Before you touch the maths, what is the one thing to check about the inputs
   you passed it — and why?

   **Answer.** That they are `float64`. In `float32` the finite-difference
   estimate is noise at the tolerance that matters, so `gradcheck` fails on
   correct code.
