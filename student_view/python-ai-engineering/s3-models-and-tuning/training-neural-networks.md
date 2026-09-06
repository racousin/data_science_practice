# Training Neural Networks: Practical Considerations

Three hyperparameters that only appear once the model is a network, and the one
regularisation technique you will use every time.

<!-- notes: The last block of the session. Short. Everything here reappears in
Session 4 as PyTorch code, so keep it conceptual and do not pre-empt it. -->

---

## Wide or deep — a new hyperparameter

![A wide MLP and a deep MLP](/api/academic_courses/assets/lessons/164/wide-vs-deep-mlp.png)

- **Wide MLP** (shallow and wide) — few layers, many neurons per layer
- **Deep MLP** (deep and narrow) — many layers, few neurons per layer

In practice, deep and narrow networks learn **hierarchical representations** with
fewer parameters. Each layer builds on the abstraction of the previous one
rather than re-deriving everything from the input.

The choice is a hyperparameter like any other, and it is searched the same way —
see the previous lesson.

---

## Mini-batch gradient descent

The full-dataset gradient is

$$
\nabla_\theta \mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta \, \ell\big(f_\theta(x_i), y_i\big)
$$

Computing it for every step is expensive. Three options:

| Variant | Gradient used | Data per step |
|---|---|---|
| Batch GD | $\nabla_\theta \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} \nabla_\theta \ell(f_\theta(x_i), y_i)$ | the whole dataset |
| Stochastic GD | $\nabla_\theta \mathcal{L} \approx \nabla_\theta \ell(f_\theta(x_i), y_i)$ | a single sample |
| **Mini-batch GD** | $\nabla_\theta \mathcal{L} \approx \frac{1}{m}\sum_{i \in \mathcal{B}} \nabla_\theta \ell(f_\theta(x_i), y_i)$ | a subset $\mathcal{B}$ of size $m$ |

![Convergence paths for the three variants](/api/academic_courses/assets/lessons/164/batch-sgd-minibatch.png)

Batch GD takes a smooth path and is slow. SGD is fast and noisy. Mini-batch is
the compromise everybody uses.

### Two words that get confused

![Batch size and epoch](/api/academic_courses/assets/lessons/164/batch-and-epoch.png)

- **Batch size** — the number of samples processed before the parameters are
  updated once.
- **Epoch** — one complete pass through the entire training set.

A dataset of 10,000 rows with `batch_size=100` is 100 updates per epoch.

---

## Overfitting in neural networks

Networks have a great many parameters, so they are **highly susceptible to
overfitting**: the model memorises the training data instead of learning
generalisable patterns.

This is the interpolation problem from the start of the session, in the family
that has the most capacity to do it.

---

## Early stopping

![Training and validation error against iterations](/api/academic_courses/assets/lessons/164/early-stopping.png)

How it works:

1. Split the data into training and validation sets.
2. Train, evaluating on the validation set after each epoch.
3. Track the validation loss.
4. When it stops improving for $N$ epochs → stop.
5. Return the model from the **best** epoch — the one with the lowest validation
   loss, not the last one.

```python
from sklearn.neural_network import MLPClassifier

MLPClassifier(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
)
```

Step 5 is the one people get wrong. Stopping and *keeping the current weights*
returns a model that is already $N$ epochs into overfitting.
