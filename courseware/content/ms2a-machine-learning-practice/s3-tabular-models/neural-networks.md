# Multi-Layer Perceptrons

The last model family of the session, and the one Session 4 builds and trains
in PyTorch. Here it is just another parametric model: a family, a parameter
count, and the same `arg min`. The Tabular Landscape argued that tree ensembles
beat it on medium-sized tables; this lesson says what it is, how it trains, and
when it is still worth a try on a table.

<!-- notes: 40 minutes. The parameter-counting quiz is worth doing live:
it is the moment the notation stops being abstract. Keep the training slides
conceptual, because Session 4 turns every one of them into PyTorch code; do not
pre-empt it. End on the tabular verdict, which is what they need for the
project. -->

---

## Machine learning and deep learning

![Classical machine learning versus deep learning](assets/tabular/ml-vs-deep-learning.png)

In classical machine learning a person builds the features and a model maps
them to the output. A deep network learns the features and the mapping
together, from raw input.

---

## Why that matters on a table

An image or a sentence arrives raw: pixels and tokens, with the useful features
buried in them. Learning those features is the part deep learning does better
than anyone.

A table arrives with its features already built — someone designed the columns.
That is most of why the advantage shrinks there, and why the network has to
earn its place against a tree ensemble instead of assuming it.

---

## The biological analogy

![From biological to artificial neuron](assets/tabular/biological-to-artificial-neuron.png)

The artificial neuron is inspired, very loosely, by the biological one:

- **Dendrites** receive input signals (inputs)
- The **cell body** integrates those signals
- If the total exceeds a threshold, the neuron **activates** and transmits along
  the axon (output)

The analogy stops there. Do not over-read it.

---

## The artificial neuron

![What a neuron computes](assets/tabular/neuron-computation.png)

A weighted sum of the inputs plus a bias, passed through an activation function.

---

## The neuron, formally

| Element | Definition |
|---|---|
| **Input** | one row, $x = (x_1, \ldots, x_{r_0})$ |
| **Parameters** | $w = (w_1, \ldots, w_{r_0})$, one per input dimension, plus a bias $b$ |
| **Pre-activation** | $z = \sum_i w_i x_i + b = w \cdot x + b$ |
| **Activation** | $a = \phi(z)$, for an activation function $\phi$ |
| **Output** | dimension 1 |

> With an identity activation, a neuron **is** a linear regression.

That is the whole novelty: a neuron is the model you already know — $w$ and $b$
are the $\beta$ of The Tabular Landscape — with a non-linear function stuck on
the end.

---

## Activation functions

![Sigmoid, tanh, ReLU, leaky ReLU](assets/tabular/activation-functions.png)

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
\qquad
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
\qquad
\text{ReLU}(z) = \max(0, z)
$$

Each is a candidate for $\phi$. The sigmoid keeps the name $\sigma$ it had in
the logistic regression of The Tabular Landscape.

---

## Which activation

ReLU is the default in modern networks: cheap to compute, and its gradient does
not vanish for positive inputs.

Sigmoid and tanh saturate: far from zero their slope is almost flat, so the
gradient that trains the layers below them all but disappears. Sigmoid survives
at the output of a binary classifier, where it turns a score into a probability.

Leaky ReLU keeps a small slope for negative inputs (the figure's $a = 0.2$ is
that slope, not the activation $a$), so a unit never goes completely silent.
Session 4 comes back to dead ReLU units.

---

## From neurons to a layer

Stack neurons into layers, and layers into a network. Two neurons reading the
same input form a layer:

$$
a = \big(\phi(w_1 \cdot x + b_1), \; \phi(w_2 \cdot x + b_2)\big) = \phi(Wx + b)
$$

![A multi-layer perceptron: an input layer, hidden layers, an output layer](assets/tabular/mlp.jpeg)

---

## The multi-layer perceptron, formally

| Element | Definition |
|---|---|
| **Input** | one row, $x = (x_1, \ldots, x_{r_0})$ |
| **Parameters** | a weight matrix $W^k$ and a bias vector $b^k$ for each layer $k = 1, \ldots, L$ |
| **Output** | one value per neuron of the last layer |

Layer $k$ has $r_k$ neurons, so its weights and bias have the shapes

$$
W^k \in \mathbb{R}^{r_k \times r_{k-1}}, \qquad b^k \in \mathbb{R}^{r_k}
$$

and the operation is a chain of layers, each an affine map and an activation:

$$
f_\theta(x) = \phi\big(W^L \phi(\ldots \phi(W^1 x + b^1) \ldots) + b^L\big)
$$

---

## What the hidden layers buy

![An MLP's decision boundary on linear, XOR and circles data](assets/tabular/mlp-decision-boundaries.png)

The same small MLP separates a linear problem, XOR and two concentric circles.
A logistic regression gets only the first: the non-linearity between the layers
is what bends the boundary.

---

## Quiz: count the parameters

1. How many parameters must be learned in an MLP with input dimension 3, a
   hidden layer of size 5, a hidden layer of size 3, and an output layer of
   size 1?
2. What is the general formula?
3. Is there any point in stacking two layers with no activation between them?

---

## Answers: the count

**1.** $L = 3$ layers, with widths
$r_0 = 3 \rightarrow r_1 = 5 \rightarrow r_2 = 3 \rightarrow r_3 = 1$.

$$
r_1(r_0 + 1) + r_2(r_1 + 1) + r_3(r_2 + 1) = 5 \times 4 + 3 \times 6 + 1 \times 4 = 42
$$

That is $20 + 18 + 4$.

**2.** Count each layer's weights and biases:

$$
W^k \in \mathbb{R}^{r_k \times r_{k-1}} \;\rightarrow\; r_k \, r_{k-1}, \qquad b^k \in \mathbb{R}^{r_k} \;\rightarrow\; r_k
$$

So each layer has $r_k(r_{k-1} + 1)$ parameters, summed over the layers.

---

## Answers: stacking linear layers

**3.** **No.** Two linear maps compose into one:

$$
W^2(W^1 x + b^1) + b^2 = (W^2 W^1) x + (W^2 b^1 + b^2) = W'x + b'
$$

with $W' = W^2 W^1$ and $b' = W^2 b^1 + b^2$.

It is equivalent to a single linear layer. The non-linearity is the *only* reason
depth buys anything.

---

## Training

Same objective and same procedure as every other parametric model:

$$
\arg\min_{\theta \in \mathbb{R}^P} \ell\big(Y, f_\theta(X)\big)
$$

$\theta$ collects every $W^k$ and $b^k$, the role $\beta$ played for the linear
models; $P$ is the parameter count from the quiz, not the feature count $p$.

1. **Initialise** the weights randomly.
2. **Train:** compute the gradient of the loss; update the parameters by
   gradient descent; iterate.
3. **Infer:** use the model to predict.

---

## The cost goes down

![Cost falling during gradient descent](assets/tabular/cost-convergence.png)

Each iteration moves the parameters a little way downhill, and the loss on the
training data falls: fast at first, then flattening as it nears a minimum.

---

## Backpropagation

Computing $\nabla \ell$ directly for a network looks intractable:

$$
\nabla_\theta \, \ell = \Big(\frac{\partial \ell}{\partial W^k}, \; \frac{\partial \ell}{\partial b^k}\Big)_{k = 1, \ldots, L} = \;?
$$

The chain rule makes it cheap, at the cost of memory:

$$
F'(x) = f'\big(g(x)\big) \cdot g'(x)
$$

---

## Two passes, one gradient

![Forward and backward pass](assets/tabular/forward-backward-pass.png)

1. **Forward pass**: for each layer $k$, compute the pre-activation and
   activation values, and keep them.
2. **Backward pass**: compute each gradient from the stored forward values and
   the gradient of the following layer.

Every derivative reduces to a product of simple local terms.

---

## Start at the last layer

Write each layer as

$$
z^k = W^k a^{k-1} + b^k, \qquad a^k = \phi(z^k), \qquad a^0 = x, \qquad a^L = f_\theta(x)
$$

The loss $\ell$ appears in one place only: the output. So the last layer is the
one term we can write down directly:

$$
\delta^L = \frac{\partial \ell}{\partial z^L} = \nabla_{a^L} \ell \;\odot\; \phi'(z^L)
$$

$$
\frac{\partial \ell}{\partial W^L} = \delta^L (a^{L-1})^\top,
\qquad
\frac{\partial \ell}{\partial b^L} = \delta^L
$$

---

## Then back one layer at a time

Everything else follows from the last layer. The chain rule moves the signal
back one layer:

$$
\delta^{k} = \big((W^{k+1})^\top \delta^{k+1}\big) \odot \phi'(z^k)
$$

and the two gradient formulas above hold at every $k$. One explicit term at the
end, one recursion for the rest: that is the whole algorithm.

The recursion needs $a^{k-1}$ and $z^k$ from the forward pass. That is the memory
cost announced earlier.

---

## Then apply gradient descent

Backpropagation returns the gradient of the loss with respect to every
parameter. Update the parameters with a gradient descent optimiser:

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \ell\big(Y, f_{\theta_t}(X)\big)
$$

The learning rate $\eta$ sets the step size. Session 4 replaces this plain
update with momentum and Adam.

---

## Mini-batch gradient descent

The full-dataset gradient is

$$
\nabla_\theta \mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta \, \ell\big(f_\theta(x_i), y_i\big)
$$

Computing it for every step is expensive. Three options:

| Variant | Gradient used | Data per step |
|---|---|---|
| Batch GD | the exact average above | the whole dataset |
| Stochastic GD | the gradient on one sample $x_i$ | a single sample |
| **Mini-batch GD** | the average over a batch $\mathcal{B}$ | a subset of size $m$ |

---

## The mini-batch estimate

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{m}\sum_{i \in \mathcal{B}} \nabla_\theta \, \ell\big(f_\theta(x_i), y_i\big)
$$

![Convergence paths for the three variants](assets/tabular/batch-sgd-minibatch.png)

Batch GD takes a smooth path and is slow. SGD is fast and noisy. Mini-batch is
the compromise everybody uses.

---

## Batch size and epoch

![Batch size and epoch](assets/tabular/batch-and-epoch.png)

- **Batch size**: the number of samples processed before the parameters are
  updated once.
- **Epoch**: one complete pass through the entire training set.

A dataset of 10,000 rows with `batch_size=100` is 100 updates per epoch.

---

## Wide or deep: a new hyperparameter

![A wide MLP and a deep MLP](assets/tabular/wide-vs-deep-mlp.png)

- **Wide MLP** (shallow and wide): few layers, many neurons per layer
- **Deep MLP** (deep and narrow): many layers, few neurons per layer

---

## Why deep and narrow

In practice, deep and narrow networks learn **hierarchical representations**
with fewer parameters. Each layer builds on the abstraction of the previous one
rather than re-deriving everything from the input.

The choice is a hyperparameter like any other, and it is searched the same way:
see *Hyperparameter Optimisation*.

---

## Overfitting in neural networks

Networks have a great many parameters, so they are **highly susceptible to
overfitting**: the model memorises the training data instead of learning
generalisable patterns.

This is the interpolation problem — a family rich enough to pass through every
training point will, if you let it, and *Model Selection and Validation*, next,
states it as a theorem — in the family with the most capacity to do it.

---

## Early stopping

![Training and validation error against iterations](assets/tabular/early-stopping.png)

The same device as `lgb.early_stopping` in *Gradient Boosting in Practice*,
with epochs in place of boosting rounds, plus one step that lesson got for
free from the library:

1. Hold out an early-stopping slice of the training rows and score it after
   every epoch.
2. When its loss has not improved for $N$ epochs, stop.
3. Return the weights from the **best** epoch, the one with the lowest
   held-out loss, not the last one.

---

## An MLP in scikit-learn

```python
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=10,
                  max_iter=500, random_state=0),
)
mlp.fit(X_train, y_train)
```

Scaling is not optional. Every input enters the same weighted sum, so a column
in thousands drowns a column in units, and gradient descent crawls along the
badly scaled directions. Keep the scaler inside the pipeline so it is fitted on
the training folds only.

---

## Early stopping in scikit-learn

`early_stopping=True` holds out `validation_fraction` of the training rows and
stops after `n_iter_no_change` epochs without improvement. It tracks the
validation **score**, accuracy or R², rather than the loss, and it restores the
weights from the best epoch.

Step 3 is the one people get wrong when they write the loop themselves, as you
will in Session 4. Stopping and *keeping the current weights* returns a model
that is already $N$ epochs into overfitting.

---

## The regressor: scale the target too

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

reg = TransformedTargetRegressor(
    regressor=make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                     max_iter=500, random_state=0)),
    transformer=StandardScaler(),
)
reg.fit(X_train, y_train)
```

The output layer starts near zero. A target in the tens of thousands leaves it
far to travel at the default learning rate: with `max_iter=500` the fit stops
on the iteration cap with a `ConvergenceWarning`, not on early stopping, still
far from the target's mean, and R² can land far below zero. Scaling `y` and
inverting it at prediction time converges in a couple of hundred epochs.

---

## Where an MLP stands on a table

On medium-sized tables, around ten thousand rows, tree ensembles still win. The
benchmark behind the Tabular Landscape (Grinsztajn et al., 2022) names three
reasons on the network's side:

- It is biased towards **smooth** functions; tabular targets are often
  irregular, jumping at thresholds.
- **Uninformative columns** hurt it far more than a tree, which ignores them.
- It is **rotation-invariant**, mixing all columns in the first layer, while a
  table's columns each carry a meaning that a split respects.

And it costs more: scaling, encoding, and a search over architecture, learning
rate and stopping that boosting mostly does not need.

---

## When an MLP is worth a try

- **Many rows.** The gap narrows as the table grows; with hundreds of
  thousands of rows a tuned network can compete.
- **Smooth, homogeneous inputs**: sensor readings, physical measurements,
  embeddings.
- **As an ensemble member.** It is built so differently from a booster that
  its errors tend to differ: exactly the diversity stacking needs.
- **Mixed inputs**: a table next to text or images, trained end to end.

Fit it after the gradient-boosting baseline, on the same split and metric, and
keep it only if it beats that number or earns its place in a stack.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.neural_network import MLPClassifier
   X = np.random.default_rng(0).normal(size=(200, 3))
   y = (X[:, 0] > 0).astype(int)
   mlp = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=2000,
                       random_state=0).fit(X, y)
   print(sum(p.size for p in mlp.coefs_ + mlp.intercepts_))   # -> 42
   ```

   The quiz network, $3 \rightarrow 5 \rightarrow 3 \rightarrow 1$: a binary
   `MLPClassifier` has a single sigmoid output, so the count matches the
   hand-worked 42.

2. Backpropagation is cheap in time and expensive in memory. What is the memory
   spent on?

   **Answer.** The forward pass keeps every layer's pre-activations $z^k$ and
   activations $a^k$, because the backward recursion needs $z^k$ and $a^{k-1}$
   to compute each layer's gradient.

3. A colleague's MLP loses to gradient boosting on a 5,000-row table of mixed
   numeric and categorical columns, several of them noise. Was that
   predictable?

   **Answer.** Yes. Five thousand rows starves a network while being plenty
   for boosting; heterogeneous columns suit per-feature splits better than a
   first layer that mixes them all; and uninformative columns hurt an MLP far
   more than a tree, which simply never splits on them.
