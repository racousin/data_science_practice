# Essential Layers

A layer is a shape contract with parameters attached. PyTorch tells you
immediately when a matrix multiplication does not line up, and says nothing at all
when a tensor lines up for the wrong reason.

This lesson covers the two layers a practitioner assembles first, `Linear` and
`Embedding`: the shape each one accepts and returns, and the two ways of
putting them together.

<!-- notes: 30 minutes. They know nn.Linear and nn.Module from the 12h module — do
not re-teach those. Spend the time on shapes: put a wrong Conv2d on screen, run it,
and read the error out loud before fixing it. Convolution itself belongs to Session
5 — stop at the shape contract and say so. The permuted-pixels figure is the one to
dwell on: it is the argument for the MLP they will submit in Lab 4. -->

---

## A layer is a contract

Every layer promises three things: the shape it accepts, the shape it returns, and
the parameters it owns.

| Layer | Input | Output | Parameters |
|---|---|---|---|
| `Linear(d_in, d_out)` | `(..., d_in)` float | `(..., d_out)` | $d_{in}d_{out} + d_{out}$ |
| `Embedding(n, d)` | `(...)` **int64** | `(..., d)` | $nd$ |
| `Conv1d(c_in, c_out, k)` | `(B, c_in, T)` | `(B, c_out, T')` | $k\,c_{in}c_{out} + c_{out}$ |
| `Conv2d(c_in, c_out, k)` | `(B, c_in, H, W)` | `(B, c_out, H', W')` | $k^2 c_{in}c_{out} + c_{out}$ |
| `MaxPool2d(k)` | `(B, C, H, W)` | `(B, C, ⌊H/k⌋, ⌊W/k⌋)` | none |
| `LSTM(d_in, h, batch_first=True)` | `(B, T, d_in)` | `(B, T, h)` and the final state | $4(d_{in}h + h^2 + 2h)$ |

> `LSTM` defaults to `(T, B, d_in)`; the shapes above assume `batch_first=True`.

---

## Linear: the last axis, and nothing else

![nn.Linear applies one weight matrix to every row of the last axis; the leading dimensions are carried through](assets/nn/linear-acts-on-the-last-dimension.png)

```python
import torch, torch.nn as nn

lin = nn.Linear(3, 5)
lin(torch.randn(2, 4, 3)).shape        # torch.Size([2, 4, 5])
lin(torch.randn(7, 3)).shape           # torch.Size([7, 5])
lin.weight.shape, lin.bias.shape       # (5, 3), (5,)
```

**Number of parameters?**

<!-- answer: 3 × 5 + 5 = 20 -->

---

## Embedding: a lookup table that learns

For a *category* there is nothing to multiply. `Embedding` stores one row per
category and returns the rows you index.

![nn.Embedding gathers one row of a learned matrix per integer index; unindexed rows get no gradient](assets/nn/embedding-lookup-table.png)

```python
emb = nn.Embedding(num_embeddings=5000, embedding_dim=32, padding_idx=0)
idx = torch.tensor([[4, 17, 0], [9, 9, 0]])      # int64, shape (2, 3)
emb(idx).shape                                    # torch.Size([2, 3, 32])
sum(p.numel() for p in emb.parameters())          # 160000
```

> `padding_idx=0`: row 0 is initialised to zeros and never receives a gradient.

---

## Activation Functions
Non-linear functions applied element-wise to introduce non-linearity:
```python
# Common activation functions
relu = nn.ReLU()        # f(x) = max(0, x)
sigmoid = nn.Sigmoid()   # f(x) = 1/(1+e^(-x))
tanh = nn.Tanh()        # f(x) = (e^x - e^(-x))/(e^x + e^(-x))

# Apply activation
x = torch.randn(2, 5)
output = relu(x)        # Negative values become 0
```
---


## Assembling: `Sequential`, or a `forward` you wrote

`Sequential` is right for a straight line:

```python
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(32, 10),
)
```

As soon as the model takes several inputs or branches, subclass `nn.Module`
and write `forward` yourself:

![Shapes through a mixed tabular model: continuous columns and an embedding concatenated, then two Linear layers](assets/nn/shape-flow-embedding-concat-linear.png)

```python
class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 8)
        self.head = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_num, x_cat):                       # (B, 6) float32, (B, 3) int64
        e = self.emb(x_cat).flatten(1)                     # (B, 3, 8) -> (B, 24)
        return self.head(torch.cat([x_num, e], dim=1))     # (B, 30) -> (B, 1)

Mixed()(torch.randn(16, 6), torch.randint(0, 50, (16, 3))).shape    # (16, 1)
```

---

## The `nn.Module` class

`nn.Module` is the base class for every neural network component in PyTorch.
It registers parameters, submodules and buffers, and handles `.to(device)`,
`.train()` / `.eval()` and `state_dict()`. 

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)     # layers stored as attributes
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)
```

### `__init__`: register what the model owns

Always call `super().__init__()` first.

**Pre-built layers**: their parameters are tracked automatically.

```python
self.conv = nn.Conv2d(3, 64, kernel_size=3)
self.bn   = nn.BatchNorm2d(64)
self.fc   = nn.Linear(64 * 28 * 28, 10)    # assumes 30×30 input (no padding)
```

**Manual parameters**: must be wrapped in `nn.Parameter`.

```python
self.params = nn.Parameter(torch.zeros(10))
# self.params = torch.zeros(10, requires_grad=True)   # NOT registered: invisible to the optimizer
```

**Buffers**: tensors that move with the model and are saved in `state_dict`, but are not trained.

```python
self.register_buffer("running_mean", torch.zeros(64))
```

**Parameter-free modules**: still registered as submodules, but own no parameters.
Using the functional form (`torch.relu`) in `forward` is equivalent.

```python
self.activation = nn.ReLU()
```

> A plain Python `list` of layers is **not** registered — use `nn.ModuleList` / `nn.ModuleDict`.

### `forward`: the computation

Runs at every call; the autograd graph is built dynamically during execution.

```python
def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.relu(x)
    x = x.flatten(1)          # (B, 64, 28, 28) -> (B, 64*28*28)
    x = self.fc(x)
    return x

model = MyModel()
output = model(input_tensor)   # calls forward() through __call__ (hooks included)
# Avoid model.forward(input_tensor): it bypasses hooks
```
