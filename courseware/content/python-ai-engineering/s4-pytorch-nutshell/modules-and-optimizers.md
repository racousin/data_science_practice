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

## A layer  $W^k$ and $b^k$

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
- Both start random, uniform, and with
  `requires_grad=True`

---

## Activation Functions

# Common activation functions
relu = nn.ReLU()        # f(x) = max(0, x)
sigmoid = nn.Sigmoid()   # f(x) = 1/(1+e^(-x))
tanh = nn.Tanh()        # f(x) = (e^x - e^(-x))/(e^x + e^(-x))

# Apply activation
x = torch.randn(2, 5)
output = relu(x)        # Negative values become 0


---


## `nn.Sequential`: a straight stack

```python
# Example: 3-layer MLP
model = nn.Sequential(
    nn.Linear(784, 256),   # 784*256 + 256 = 200,960 params
    nn.ReLU(),
    nn.Linear(256, 128),   # 256*128 + 128 = 32,896 params  
    nn.ReLU(),
    nn.Linear(128, 10)     # 128*10 + 10 = 1,290 params
)

print(f"Total parameters: {count_parameters(model)}")  # 235,146
```


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



---

## How dimensions transform through the network

```python
# Define layers
layer1 = nn.Linear(784, 256)
layer2 = nn.Linear(256, 10)

# Process 2D tensor (50 observations)
x = torch.randn(50, 784)
x = layer1(x)  # [50, 784] → [50, 256]
x = layer2(x)  # [50, 256] → [50, 10]
print(f"2D output: {x.shape}")  # torch.Size([50, 10])

# Process 3D tensor (20 batches of 100 observations)
x = torch.randn(20, 100, 784)
x = layer1(x)  # [20, 100, 784] → [20, 100, 256]
x = layer2(x)  # [20, 100, 256] → [20, 100, 10]
print(f"3D output: {x.shape}")  # torch.Size([20, 100, 10])

# Note: Only the last dimension (features) changes
# All other dimensions remain constant
```
