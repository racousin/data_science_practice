
## The artificial neuron

![What a neuron computes](assets/nn/neuron-computation.png)

| | |
|---|---|
| **Input** | $X = (x_1, \ldots, x_{r_0})$ |
| **Parameters** | $w = (w_1, \ldots, w_{r_0})$ — one per input dimension — plus a bias $b$ |
| **Pre-activation** | $z = \sum_i w_i x_i + b = w \cdot X + b$ |
| **Activation** | $a = \sigma(z)$ |
| **Output** | dimension 1 |

> With an identity activation, a neuron **is** a linear regression.

That is the whole novelty: a neuron is the model you already know, with a
non-linear function stuck on the end.

---

## Activation functions

![Sigmoid, tanh, ReLU, leaky ReLU](assets/nn/activation-functions.png)

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
\qquad
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
\qquad
\text{ReLU}(z) = \max(0, z)
$$

ReLU is the default in modern networks — cheap to compute, and its gradient does
not vanish for positive inputs.

---


## The multi-layer perceptron

Stack neurons into layers, and layers into a network:

$$
a = \big(\sigma(w_1 \cdot X + b_1), \; \sigma(w_2 \cdot X + b_2)\big) = \sigma(WX + b)
$$

| | |
|---|---|
| **Input** | $X = (x_1, \ldots, x_{r_0})$ |
| **Parameters** | $W^1, \ldots, W^L$ with $W^k \in \mathbb{R}^{r_k \times r_{k-1}}$, and biases $b^k \in \mathbb{R}^{r_k}$ |
| **Operation** | $f_\theta(X) = \sigma\big(W^L \sigma(\ldots \sigma(W^1 X + b^1) \ldots) + b^L\big)$ |
| **Output** | $r_L$ — the number of neurons in the last layer |

![mlp.jpeg](assets/nn/mlp.jpeg)

---

## Quiz: count the parameters

1. How many parameters must be learned in an MLP with input dimension 3, a
   hidden layer of size 5, a hidden layer of size 3, and an output layer of
   size 1?
2. What is the general formula?
3. Is there any point in stacking two layers with no activation between them?

---

### Answers

**1.** $L = 3$ layers, with widths $r_0 = 3 \rightarrow r_1 = 5 \rightarrow r_2 = 3 \rightarrow r_3 = 1$.

$$
r_1(r_0 + 1) + r_2(r_1 + 1) + r_3(r_2 + 1) = 5 \times 4 + 3 \times 6 + 1 \times 4 = 20 + 18 + 4 = 42
$$

**2.** Weights $W^k \in \mathbb{R}^{r_k \times r_{k-1}} \rightarrow r_k \cdot r_{k-1}$;
biases $b^k \in \mathbb{R}^{r_k} \rightarrow r_k$; so $r_k(r_{k-1} + 1)$ per
layer, summed over layers.

**3.** **No.** Two linear maps compose into one:

$$
W^2(W^1 X + b^1) + b^2 = (W^2 W^1) X + (W^2 b^1 + b^2) = W'X + b'
$$

with $W' = W^2 W^1$ and $b' = W^2 b^1 + b^2$.

It is equivalent to a single linear layer. The non-linearity is the *only* reason
depth buys anything.

---

## Training

Same objective and same procedure as every other parametric model:

$$
\arg\min_{\theta \in \mathbb{R}^p} \ell\big(Y, f_\theta(X)\big)
$$

0. Initialise the weights randomly.
1. **Train:** compute the gradient of the loss; update the parameters by gradient
   descent; iterate.
2. **Inference:** use the model to predict.

![Cost falling during gradient descent](assets/nn/cost-convergence.png)

---

## Backpropagation

Computing $\nabla \ell$ directly for a network looks intractable:

$$
\nabla \ell\big(Y, f_\theta(X)\big) = \frac{\partial \ell\big(Y, f(X)\big)}{\partial W^k} = \;?
\qquad
\forall k \in [1, L]
$$

The chain rule makes it cheap — at the cost of memory.

$$
F'(x) = f'\big(g(x)\big) \cdot g'(x)
$$

---

## Two passes, one gradient

![Forward and backward pass](assets/nn/forward-backward-pass.png)

1. **Forward pass** — for each layer $k$, compute the pre-activation and
   activation values, and keep them.
2. **Backward pass** — compute each gradient from the stored forward values and
   the gradient of the following layer.

Every derivative reduces to a product of simple local terms.


---

## Mini-batch gradient descent

Session 2-3 computed the loss and its gradient on all $n$ rows at every step.
Mini-batch gradient descent takes each step on a random subset $\mathcal{B}$ of
$B$ rows, a **batch**, and descends on its mean loss:

$$
\ell_{\mathcal{B}}(\theta) = \frac{1}{B} \sum_{i \in \mathcal{B}} L(y_i, f_\theta(x_i))
$$

![Batch, stochastic and mini-batch gradient descent on the same loss](assets/nn/batch-sgd-minibatch.png)

- All $n$ rows: the exact gradient, but a full pass over the data per step. One
  row: cheap steps on an erratic path. $B$ from 32 to 512: the usual compromise.
- The batch mean estimates the full mean: right on average, noisier as $B$
  shrinks.

---

## Batch size and epoch

![The lab's 33,475 training rows cut into batches of 256: one update per batch, 131 per epoch, the last batch holding the 195 rows left over](assets/nn/batch-and-epoch.png)

- **Batch size** $B$: the rows behind one update of $\theta$, one
  **iteration**.
- **Epoch**: one pass over all $n$ training rows, $\lceil n / B \rceil$
  iterations: 131 for the lab's 33,475 rows at $B = 256$, as in the figure.


---

## Standardise the inputs

One learning rate serves every weight, and a weight's gradient grows with the
size of its input. A year, about 2,000, gets gradients about 2,000 times those
of a feature of size 1: gradient descent fitting a line to it is stable only
for $\eta < 2.5 \times 10^{-7}$; standardised, for any $\eta < 1$.

```python
mean = X_tr.mean(dim=0)                 # the training part only
std = X_tr.std(dim=0)
X_tr = (X_tr - mean) / std
X_val = (X_val - mean) / std            # the same mean and std
```


---

## One epoch, by hand


```python
perm = torch.randperm(len(X_tr))        # a new random order each epoch
for i in range(0, len(X_tr), B):        # i = 0, 256, 512, ...
    xb = X_tr[perm[i:i + B]]            # the next B rows
    yb = y_tr[perm[i:i + B]]            # and their targets
```

`torch.randperm(n)` returns 0 to $n - 1$ in random order; `perm[i:i + B]` is
the next $B$ of those row numbers, and indexing both tensors with them keeps
each row with its target. The loop yields 131 batches: 130 of 256 rows, then
the 195 left over. A new order each epoch means new batches: the "stochastic"
in `torch.optim.SGD`.

---

## `DataLoader` does the same

```python
from torch.utils.data import TensorDataset, DataLoader
ds = TensorDataset(X_tr, y_tr)          # pairs row i of X_tr and y_tr
loader = DataLoader(ds, batch_size=B, shuffle=True)
len(loader)                             # 131, as by hand
```

- `for xb, yb in loader:` yields the epoch's batches, `xb` of shape (256, 12)
  and `yb` of shape (256, 1), the last of 195 rows, and reshuffles every
  epoch. This is the form most PyTorch code uses.
- `TensorDataset` keeps the tensors it was given: standardise before building
  it. Shuffle the training set only.

---

## The loop, in one picture

![One epoch: train on every batch, validate, keep the best, log; then restore the best](assets/nn/training-loop.png)

The dashed box runs once per epoch, the blue box once per batch. The objects it
calls, with a smaller network than the lab's:

```python
model = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 1))
loss_fn = nn.MSELoss()                  # the lab uses its pinball loss
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

## The training step

```python
for xb, yb in loader:                   # one epoch: every batch once
    opt.zero_grad()
    loss_fn(model(xb), yb).backward()
    opt.step()
```

- Call `model.train()` before it. Some layers act differently in training and
  in evaluation, such as dropout , which switches off random neurons
  during training only. Forgetting the switch raises no error.
- On a GPU, move each batch first: `xb, yb = xb.to(device), yb.to(device)`
