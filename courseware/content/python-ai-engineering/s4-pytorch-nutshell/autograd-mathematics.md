# Autograd — the Mathematics

Reference only — not covered in class. Session 4's autograd lesson treats
`.backward()` as a tool. This page is what it computes.

Read it if you want to understand *why* reverse mode is the right choice for
neural networks, or if you plan to write a custom `autograd.Function`.

---

## The chain rule

For $z = f(g(x))$:

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}, \quad y = g(x)
$$

A neural network is a deep composition:

$$
L = \ell \circ f_n \circ f_{n-1} \circ \dots \circ f_1
$$

so the gradient is a long product of local derivatives. Automatic
differentiation is bookkeeping for that product.

---

## Three ways to get derivatives

| Method | How | Problem |
|---|---|---|
| **Symbolic** | manipulate expressions (SymPy) | expression swell; needs a closed form |
| **Numerical** | $\frac{f(x+h) - f(x)}{h}$ | truncation and round-off error; $O(n)$ evaluations |
| **Automatic** | chain rule on the computation graph | none of the above — exact to machine precision |

Numerical differentiation is still useful, once: as a **gradient check** on a
hand-written backward pass.

---

## Forward mode

Propagate derivatives **with** the computation. Carry a tangent alongside each
value:

$$
(v, \dot{v}) \quad \text{where} \quad \dot{v} = \frac{\partial v}{\partial x_j}
$$

One pass gives you $\frac{\partial y_i}{\partial x_j}$ for **all outputs** with
respect to **one input**.

Cost: one pass per input. Good when $n_{\text{in}} \ll n_{\text{out}}$.

---

## Reverse mode

Propagate derivatives **against** the computation. Carry an adjoint:

$$
\bar{v} = \frac{\partial L}{\partial v}
$$

One backward pass gives you $\frac{\partial L}{\partial x_j}$ for **all inputs**
with respect to **one output**.

Cost: one pass per output. Good when $n_{\text{in}} \gg n_{\text{out}}$.

---

## Why reverse mode wins here

A neural network has $10^6$–$10^{11}$ parameters and **one** scalar loss.

$$
n_{\text{in}} = |\theta| \approx 10^9, \qquad n_{\text{out}} = 1
$$

Forward mode would need $10^9$ passes. Reverse mode needs one. That asymmetry is
the entire reason deep learning is computationally feasible, and the reason
`.backward()` insists on a scalar.

The cost is memory: the forward activations must be kept until the backward pass
consumes them. That is why batch size is limited by memory, not by compute.

---

## Backpropagation for an MLP

Layer $l$ computes:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = \sigma(z^{(l)})
$$

Define the error signal:

$$
\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}
$$

---

## Output layer

$$
\delta^{(L)} = \nabla_a L \odot \sigma'(z^{(L)})
$$

where $\odot$ is element-wise multiplication.

For squared error with a linear output, $\nabla_a L = (a^{(L)} - y)$ and
$\sigma' = 1$, so $\delta^{(L)} = a^{(L)} - y$ — the residual, directly.

---

## Hidden layers

$$
\delta^{(l)} = \big( (W^{(l+1)})^T \delta^{(l+1)} \big) \odot \sigma'(z^{(l)})
$$

The error is pushed backwards through the transpose of the forward weights, then
gated by the local activation derivative. This recursion is the algorithm.

---

## Parameter gradients

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \big(a^{(l-1)}\big)^T,
\qquad
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

Note what this needs: $a^{(l-1)}$, an activation from the **forward** pass. That
is the memory cost named above.

---

## The full algorithm

1. **Forward** — compute and store $z^{(l)}, a^{(l)}$ for every layer.
2. **Backward** — $\delta^{(L)}$ at the output; recurse down to $\delta^{(1)}$.
3. **Gradients** — form $\partial L / \partial W^{(l)}$ from $\delta^{(l)}$ and
   $a^{(l-1)}$.
4. **Update** — $\theta \leftarrow \theta - \eta \nabla_\theta L$.

Steps 2 and 3 are `loss.backward()`. Step 4 is `optimizer.step()`.

---

## Vanishing and exploding gradients

The recursion multiplies by $(W^{(l+1)})^T$ and $\sigma'(z^{(l)})$ at every
layer. Over $n$ layers, the gradient scales roughly as the product of $n$ such
factors.

- Factors consistently $< 1$ → the gradient **vanishes**; early layers stop
  learning.
- Factors consistently $> 1$ → the gradient **explodes**; the loss becomes `nan`.

---

## Why sigmoid was abandoned

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 0.25
$$

Ten sigmoid layers multiply the gradient by at most $0.25^{10} \approx 10^{-6}$.
Depth was not usable until this was addressed.

The fixes, in the order they matter:

- **ReLU** — derivative is exactly 1 on the positive side, so no shrinkage
- **Residual connections** — an additive path the gradient reaches undiminished
- **Normalisation layers** — keep $z$ in the regime where $\sigma'$ is healthy
- **Careful initialisation** — He/Xavier scaling keeps variance stable per layer

---

## Gradient clipping

For explosion, cap the norm before the update:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Placed between `loss.backward()` and `optimizer.step()`. Standard in recurrent
networks and in most reinforcement learning, where the loss scale varies wildly
between batches.

---

## Checking a hand-written gradient

If you ever write a custom backward, verify it numerically:

```python
torch.autograd.gradcheck(my_function, (x.double().requires_grad_(),))
```

It compares your analytic gradient against a finite-difference estimate. Use
`float64` — `float32` does not have the precision for the comparison to mean
anything.

---

## Check yourself

1. Why does a neural network use reverse mode rather than forward mode, and what
   does that choice cost?

   **Answer.** Reverse mode costs one pass per **output**, and a network has one
   scalar loss against $10^6$–$10^{11}$ parameters; forward mode costs one pass
   per input, so it would need $10^9$ of them. The cost is memory — every forward
   activation $a^{(l-1)}$ must be kept until the backward pass consumes it, which
   is why batch size is limited by memory rather than by compute.

2. One hidden layer, by hand and then by autograd. For a 2-1-1 network with
   $\sigma = \tanh$ and squared error, derive $\partial L / \partial W^{(1)}$
   from the recursion above, then run this.

   ```python
   import torch
   W1 = torch.tensor([[0.5, -0.5]], dtype=torch.float64, requires_grad=True)
   W2 = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
   x  = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
   y  = torch.tensor([[1.0]], dtype=torch.float64)

   a1 = torch.tanh(W1 @ x)
   loss = ((W2 @ a1 - y) ** 2).sum()
   loss.backward()
   print(W1.grad)   # -> tensor([[ -6.0532, -12.1065]], dtype=torch.float64)
   ```

   **Answer.** $z^{(1)} = -0.5$ and $a^{(1)} = \tanh(-0.5) = -0.46212$. The output
   is linear, so $\delta^{(2)} = 2(W^{(2)}a^{(1)} - y) = -3.8485$. Then
   $\delta^{(1)} = (W^{(2)})^T \delta^{(2)} \odot (1 - \tanh^2 z^{(1)})
   = 2 \times (-3.8485) \times 0.78645 = -6.0532$, and
   $\partial L / \partial W^{(1)} = \delta^{(1)} (a^{(0)})^T
   = [-6.0532, -12.1065]$. If your derivation disagrees, the discrepancy is in a
   transpose or in $\sigma'$ — it is always one of those two. `float64` here
   just matches the precision `torch.autograd.gradcheck` needs when you check a
   hand-written backward; for this comparison `float32` prints the same four
   decimals.

3. Ten stacked sigmoid layers. What happens to the gradient, by roughly how much,
   and which of the fixes listed here removes the cause most directly?

   **Answer.** It vanishes. $\sigma' \le 0.25$, so ten layers scale the gradient
   by at most $0.25^{10} \approx 10^{-6}$ and the early layers stop learning.
   ReLU is the most direct fix — its derivative is exactly 1 on the positive
   side, so there is no per-layer shrinkage at all.
