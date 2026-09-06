# Multi-Layer Perceptrons

The last model family of the session, and the one Session 4 implements from
scratch in PyTorch. Here it is just another parametric model: a family, a
parameter count, and the same `arg min`.

<!-- notes: The deck for this part was authored in French; the bodies are in
English to match the rest of the course. ~25 minutes. The parameter-counting
quiz is worth doing live — it is the moment the notation stops being abstract. -->

---

## The biological analogy

![From biological to artificial neuron](assets/s3-models-and-tuning/neural-networks/biological-to-artificial-neuron.png)

The artificial neuron is inspired — very loosely — by the biological one:

- **Dendrites** receive input signals (inputs)
- The **cell body** integrates those signals
- If the total exceeds a threshold, the neuron **activates** and transmits along
  the axon (output)

The analogy stops there. Do not over-read it.

---

## The artificial neuron

![What a neuron computes](assets/s3-models-and-tuning/neural-networks/neuron-computation.png)

| | |
|---|---|
| **Input** | $X = (x_1, \ldots, x_{r_0})$ |
| **Parameters** | $w = (b, w_1, \ldots, w_{r_0})$ — one per input dimension, plus a bias |
| **Operation** | $z = \sum_i w_i x_i + b$ |
| **Activation** | $o = \sigma(w \cdot X)$ |
| **Output** | dimension 1 |

> With an identity activation, a neuron **is** a linear regression.

That is the whole novelty: a neuron is the model you already know, with a
non-linear function stuck on the end.

---

## Activation functions

![Sigmoid, tanh, ReLU, leaky ReLU](assets/s3-models-and-tuning/neural-networks/activation-functions.png)

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
o = \big(\sigma(w_1 \cdot X), \; \sigma(w_2 \cdot X)\big) = \sigma(WX)
$$

| | |
|---|---|
| **Input** | $X = (x_1, \ldots, x_{r_0})$ |
| **Parameters** | $W^1, \ldots, W^l$ with $W^k \in \mathbb{R}^{r_{k-1} \times r_k}$, and biases $b^k \in \mathbb{R}^{r_k}$ |
| **Operation** | $f_\theta(X) = W^l \sigma\big(W^{l-1} \sigma(\ldots \sigma(W^1 X))\big)$ |
| **Output** | $r_l$ — the number of neurons in the last layer |

$l$ layers, $r_k$ neurons per layer. The parameter count is

$$
\text{total} = \sum_{k=1}^{l} r_k \cdot (r_{k-1} + 1)
$$

— the $+1$ being the bias of each neuron.

---

## Quiz: count the parameters

1. How many parameters must be learned in an MLP with input dimension 3, a
   hidden layer of size 5, a hidden layer of size 3, and an output layer of
   size 1?
2. What is the general formula?
3. Is there any point in stacking two layers with no activation between them?

### Answers

**1.** The layer widths are $r_0 = 3 \rightarrow r_1 = 5 \rightarrow r_2 = 3 \rightarrow r_3 = 1$.

$$
r_1(r_0 + 1) + r_2(r_1 + 1) + r_3(r_2 + 1) = 5 \times 4 + 3 \times 6 + 1 \times 4 = 20 + 18 + 4 = 42
$$

**2.** Weights $W^k \in \mathbb{R}^{r_{k-1} \times r_k} \rightarrow r_{k-1} \cdot r_k$;
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

![Cost falling during gradient descent](assets/s3-models-and-tuning/neural-networks/cost-convergence.png)

---

## Backpropagation

Computing $\nabla \ell$ directly for a network looks intractable:

$$
\nabla \ell\big(Y, f_\theta(X)\big) = \frac{\partial \ell(Y, f(X))}{\partial w_i^k} = \;?
\qquad
\forall k \in [0, l], \; \forall i \in [0, r_k]
$$

The chain rule makes it cheap — at the cost of memory.

$$
F'(x) = f'\big(g(x)\big) \cdot g'(x)
$$

---

## Two passes, one gradient

![Forward and backward pass](assets/s3-models-and-tuning/neural-networks/forward-backward-pass.png)

1. **Forward pass** — for each layer $k$, compute the pre-activation and
   activation values, and keep them.
2. **Backward pass** — compute each gradient from the stored forward values and
   the gradient of the following layer.
3. **Apply gradient descent.**

Every derivative reduces to a product of simple local terms. That is the entire
trick, and Session 4 shows PyTorch doing it for you with `loss.backward()`.

---

## Why it matters

![MLP decision boundaries on three datasets](assets/s3-models-and-tuning/neural-networks/mlp-decision-boundaries.png)

The circles dataset that defeated logistic regression, and the XOR dataset that
defeats every linear model, are both solved — with no kernel and no hand-built
features.

---

## Features, learned instead of designed

![Feature extraction, learned instead of designed](assets/s3-models-and-tuning/neural-networks/ml-vs-deep-learning.png)

> **The promise of deep learning:** the model learns the relevant features
> automatically from raw data (images, text, audio). Feature extraction and
> classification are unified in a single end-to-end network.

---

## Why they became the standard

| | |
|---|---|
| **Modularity** | architectures adapt to very different data types |
| **Parallelisation** | fast, efficient training on GPUs |
| **Performance** | capacity to model complex relationships |

| Field | Moment |
|---|---|
| Computer vision | **2012** — AlexNet wins ImageNet by a huge margin (15.3% vs 26.2% error) |
| Speech recognition | **2012** — deep networks replace GMM-HMMs as the standard acoustic model |
| Games | **2016** — AlphaGo defeats Lee Sedol |
| Machine translation | **2017** — the Transformer ("Attention Is All You Need") becomes the new paradigm |

![AlphaGo versus Lee Sedol](assets/s3-models-and-tuning/neural-networks/alphago.png)
