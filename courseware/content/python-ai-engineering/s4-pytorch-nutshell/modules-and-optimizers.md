# Neural Networks with `nn.Module`

Session 3 wrote the multi-layer perceptron as layers of weights $W^k$ and
biases $b^k$, joined by an activation $\sigma$. `torch.nn` provides each layer
as an object that holds its own parameters, and `nn.Module` as the container
that joins layers into one model, with parameters $\theta$. This lesson builds the lab's
network and follows the shape of the data through it: what goes in, what comes
out.

<!-- notes: 8 minutes. The model is the lab's, 12 features -> 64 -> 64 -> 1,
built by make_model(); lesson 8 and the lab reuse that function, so it is
defined here. Keep asking "what shape comes out?": input and output are where students
go wrong, not the middle. Neurons, activations and parameter counting are
Session 3's: recall them, do not re-teach them. Subclassing gets one slide,
because it is what students will read in other people's code, not what the lab
needs. If asked whether the optimizer must be built after model.to(): the
parameters stay the same objects, so either order works in current PyTorch;
moving first is the documented habit. -->

---

## A layer is Session 3's $W^k$ and $b^k$

```python
X = torch.randn(256, 12)       # a batch: 256 rows of 12 features
layer = nn.Linear(in_features=12, out_features=64)
layer.weight.shape             # torch.Size([64, 12]): (out, in)
layer(X).shape                 # torch.Size([256, 64])
```

$$
Z = X\,(W^k)^{\top} + b^k
$$

- `layer.weight` is $W^k$, of shape $(r_k, r_{k-1})$; `layer.bias` is $b^k$,
  of shape `(64,)`, added to every row. Session 3 wrote $W^k x + b^k$ for one
  sample $x$, a column; here samples are rows, hence the transpose.
- Only the last dimension must equal `in_features`; the batch dimension passes
  through. Eleven features fail with `mat1 and mat2 shapes cannot be multiplied
  (256x11 and 12x64)`.
- Both start random, uniform in $\pm 1/\sqrt{12} \approx \pm 0.29$, and with
  `requires_grad=True` ([lesson 3](/courses/python-ai-engineering/s4-pytorch-nutshell/course/autograd)).

---

## The lab's network, with its shapes

![12 -> 64 -> 64 -> 1, with the tensor shape on every arrow and the parameters of every layer](assets/s4-pytorch-nutshell/modules-and-optimizers/mlp-shapes.png)

- **Input**: $(B, 12)$ float32, $B$ rows of 12 standardised features. float64
  from pandas fails at the first layer (lesson 2).
- **Output**: $(B, 1)$, one number per row: the promised trip duration.
- Depth and width are hyperparameters, chosen on validation (Session 3).

---

## `nn.Sequential`: a straight stack

```python
def make_model(hidden=64):
    return nn.Sequential(nn.Linear(12, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))
```

- The data runs through the layers in order: each `out_features` is the next
  layer's `in_features`. The width `hidden` is an argument, so one call
  rebuilds the same network at any width (lesson 8 and the lab do).
- `nn.ReLU()` is the activation $\sigma$: $\max(0, z)$ element by element, with
  no parameters. It maps `[-2., 0., 3.]` to `[0., 0., 3.]`. Without it the
  layers collapse into one linear map (Session 3's quiz).
- Nothing follows the last `Linear`. Session 3 applied $\sigma$ to the output
  too; here a regression output must be free to take any real value.

---

## Calling the model and counting its parameters

```python
model = make_model()
model(X).shape                              # torch.Size([256, 1])
sum(p.numel() for p in model.parameters())  # 5057
```

- `model(X)` runs every layer in order and returns one number per row.
- `model.parameters()` yields every weight and bias tensor; `p.numel()` counts
  the entries of one of them.

$$
\sum_k r_k (r_{k-1} + 1) = 64 \times 13 + 64 \times 65 + 1 \times 65 = 5057
$$

Session 3's formula, with the figure's three terms: 832, 4,160 and 65.

---

## Input and output are decided by the task

With $h$ the width of the last hidden layer:

| Task | Last layer | Output | Prediction |
|---|---|---|---|
| Regression (the lab) | `Linear(h, 1)` | $(B, 1)$, the value | the output itself |
| Binary | `Linear(h, 1)` | $(B, 1)$, one logit | `out.sigmoid() > 0.5` |
| $K$ classes | `Linear(h, K)` | $(B, K)$, $K$ logits | `out.argmax(dim=1)` |

- A **logit** is a raw score, any real number. The sigmoid turns one into a
  probability; the **softmax** turns $K$ of them into $K$ probabilities that
  sum to 1.
- The model still ends on a bare `Linear`: the loss applies the sigmoid or the
  softmax itself ([lesson 6](/courses/python-ai-engineering/s4-pytorch-nutshell/course/losses)).
  `argmax` needs neither: the largest logit is the most probable class.
- Targets: $(B, 1)$ float32 in the first two rows, $(B,)$ int64 class indices
  from 0 to $K - 1$ in the third.

---

## Writing your own `nn.Module`

A network whose data branches or merges does not fit `nn.Sequential`. It
subclasses `nn.Module`:

```python
class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()                  # required, first
        self.fc1, self.fc2 = nn.Linear(12, 64), nn.Linear(64, 1)
```

- Assigning a layer to `self` **registers** it: its weights join
  `parameters()`, and `TwoLayer()` holds 897. Layers kept in a plain Python
  list are not registered (0 parameters): use `nn.ModuleList`, a list that
  registers them.
- A second method, `forward(self, x)`, says how the data flows; here it returns
  `self.fc2(torch.relu(self.fc1(x)))`, where `torch.relu` is `nn.ReLU()` as a
  function. Without it, calling the model raises `missing the required
  "forward" function`.
- Call `model(x)`, not `model.forward(x)`: the call also runs the functions
  (hooks) that PyTorch and other libraries attach to a module.

---

## What a module gives you

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)                     # every weight moves, in place
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

- `model.parameters()` replaces the list
  [lesson 4](/courses/python-ai-engineering/s4-pytorch-nutshell/course/optimizers)
  wrote by hand, `[w]`: every registered weight, however many layers.
- `device` is chosen as in lesson 1. A module's `.to()` moves it in place; a
  tensor's `.to()` returns a copy. The data must follow, `X = X.to(device)`,
  or the first layer raises a `RuntimeError`.
- `model.train()` and `model.eval()` come in lesson 7; `model.state_dict()`,
  the weights by name, in lesson 8.

---

## Check yourself

1. What is the shape of `nn.Linear(12, 64).weight`, and how many parameters
   does the layer hold?

   **Answer.** `(64, 12)`, out by in; $64 \times 12 + 64 = 832$.

2. Run this. What does it print?

   ```python
   m = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 3),
                     nn.ReLU(), nn.Linear(3, 1))
   n_params = sum(p.numel() for p in m.parameters())
   print(n_params, m(torch.randn(7, 3)).shape)
   ```

   **Answer.** `42 torch.Size([7, 1])`. It is Session 3's quiz network,
   $20 + 18 + 4$ parameters, with one output per row.

3. A 3-class model ends with `nn.Linear(64, 3)`. What is its output shape for a
   batch of 32, and how do you get the predicted class?

   **Answer.** `(32, 3)`, three logits per row; `out.argmax(dim=1)` gives the
   32 class indices.
