import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const TensorFundamentals = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        {/* Introduction */}
        <div id="introduction">
          <Title order={1} mb="xl">
            Tensor Fundamentals in PyTorch
          </Title>
          <Text size="xl" className="mb-6">
            Understanding the Building Blocks of Deep Learning
          </Text>
          <Paper className="p-6 bg-blue-50 mb-6">
            <Text size="lg" mb="md">
              Tensors are the fundamental data structure in deep learning frameworks.
              They generalize matrices to arbitrary dimensions and enable efficient computation on GPUs.
            </Text>
            
            <Text className="mb-4">
              Mathematically, a tensor <InlineMath>{`\\mathbf{T} \\in \\mathbb{R}^{n_1 \\times n_2 \\times \\cdots \\times n_k}`}</InlineMath> is a multi-dimensional array where <InlineMath>k</InlineMath> is the number of dimensions (rank).
            </Text>
            <List>
              <List.Item><strong>Scalar:</strong> 0-dimensional tensor <InlineMath>{`s \\in \\mathbb{R}`}</InlineMath> (single number)</List.Item>
              <List.Item><strong>Vector:</strong> 1-dimensional tensor <InlineMath>{`\\mathbf{v} \\in \\mathbb{R}^n`}</InlineMath> (array of numbers)</List.Item>
              <List.Item><strong>Matrix:</strong> 2-dimensional tensor <InlineMath>{`\\mathbf{M} \\in \\mathbb{R}^{m \\times n}`}</InlineMath> (2D array)</List.Item>
              <List.Item><strong>Higher-order Tensor:</strong> <InlineMath>{`\\mathbf{T} \\in \\mathbb{R}^{n_1 \\times n_2 \\times \\cdots \\times n_k}`}</InlineMath> where <InlineMath>{`k \\geq 3`}</InlineMath></List.Item>
            </List>
          </Paper>
        </div>

        {/* Tensor Dimensions */}
        <div id="dimensions">
          <Title order={2} mb="xl">Understanding Tensor Dimensions</Title>
          
          <Grid gutter="xl">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50 mb-4">
                <Title order={4} mb="sm">Scalar (0D)</Title>
                <CodeBlock 
                  language="python" 
                  code={`import torch

# Scalar - single value
scalar = torch.tensor(3.14)
print(scalar.shape)  # torch.Size([])
print(scalar.ndim)   # 0`} 
                />
              </Paper>
              
              <Paper className="p-4 bg-gray-50">
                <Title order={4} mb="sm">Vector (1D)</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Vector - array of values
vector = torch.tensor([1, 2, 3, 4])
print(vector.shape)  # torch.Size([4])
print(vector.ndim)   # 1`} 
                />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50 mb-4">
                <Title order={4} mb="sm">Matrix (2D)</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Matrix - 2D array
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(matrix.shape)  # torch.Size([2, 3])
print(matrix.ndim)   # 2`} 
                />
              </Paper>
              
              <Paper className="p-4 bg-gray-50">
                <Title order={4} mb="sm">3D Tensor</Title>
                <CodeBlock 
                  language="python" 
                  code={`# 3D Tensor - e.g., RGB image
tensor_3d = torch.randn(3, 224, 224)
# [channels, height, width]
print(tensor_3d.shape)  # torch.Size([3, 224, 224])
print(tensor_3d.ndim)   # 3`} 
                />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Creating Tensors */}
        <div id="creation">
          <Title order={2} mb="xl">Creating Tensors in PyTorch</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Stack spacing="md">
                <Paper className="p-4 bg-green-50">
                  <Title order={4} mb="sm">From Python Lists</Title>
                  <CodeBlock 
                    language="python" 
                    code={`# From list
tensor = torch.tensor([1, 2, 3])

# From nested lists (2D)
matrix = torch.tensor([[1, 2], 
                       [3, 4]])`} 
                  />
                </Paper>
                
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} mb="sm">Random Initialization</Title>
                  <CodeBlock 
                    language="python" 
                    code={`# Uniform random [0, 1)
rand_tensor = torch.rand(3, 4)

# Normal distribution (μ=0, σ=1)
randn_tensor = torch.randn(3, 4)

# Random integers
randint_tensor = torch.randint(0, 10, (3, 4))`} 
                  />
                </Paper>
              </Stack>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Stack spacing="md">
                <Paper className="p-4 bg-purple-50">
                  <Title order={4} mb="sm">Special Tensors</Title>
                  <CodeBlock 
                    language="python" 
                    code={`# Zeros
zeros = torch.zeros(3, 4)

# Ones  
ones = torch.ones(3, 4)

# Identity matrix
eye = torch.eye(3)

# Filled with specific value
full = torch.full((3, 4), 7.0)`} 
                  />
                </Paper>
                
                <Paper className="p-4 bg-orange-50">
                  <Title order={4} mb="sm">From NumPy</Title>
                  <CodeBlock 
                    language="python" 
                    code={`import numpy as np

# NumPy to Tensor
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# Tensor to NumPy
tensor = torch.tensor([1, 2, 3])
np_array = tensor.numpy()`} 
                  />
                </Paper>
              </Stack>
            </Grid.Col>
          </Grid>
        </div>

        {/* Tensor Operations */}
        <div id="operations">
          <Title order={2} mb="xl">Essential Tensor Operations</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-gray-50 mb-4">
                <Title order={4} mb="sm">Arithmetic Operations</Title>
                <Text className="mb-3">
                  <strong>Element-wise operations:</strong> Applied component-wise using broadcasting rules.
                </Text>
                <BlockMath>{`
                  \\begin{aligned}
                  \\text{Addition: } (\\mathbf{A} + \\mathbf{B})_{ij} &= A_{ij} + B_{ij} \\\\
                  \\text{Hadamard Product: } (\\mathbf{A} \\odot \\mathbf{B})_{ij} &= A_{ij} \\cdot B_{ij}
                  \\end{aligned}
                `}</BlockMath>
                <CodeBlock 
                  language="python" 
                  code={`# Element-wise operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition: c_i = a_i + b_i
c = a + b  # or torch.add(a, b)

# Element-wise multiplication (Hadamard product)
d = a * b  # or torch.mul(a, b)

# Matrix multiplication: C = AB where C_ik = Σ_j A_ij B_jk
x = torch.randn(2, 3)
y = torch.randn(3, 4)
z = torch.mm(x, y)  # or x @ y  # Result: (2, 4)`} 
                />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Reshaping Operations</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Reshape
x = torch.randn(4, 6)
y = x.view(2, 12)  # Must be compatible
z = x.view(-1, 8)  # -1 infers dimension

# Squeeze & Unsqueeze
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()  # Remove dims of size 1
z = y.unsqueeze(0)  # Add dim at position 0`} 
                />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Aggregation Operations</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Aggregations
x = torch.randn(3, 4)

mean_val = x.mean()
sum_val = x.sum()
max_val = x.max()
min_val = x.min()

# Along specific dimension
mean_rows = x.mean(dim=0)  # Mean of each column
sum_cols = x.sum(dim=1)    # Sum of each row`} 
                />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* GPU Acceleration */}
        <div id="gpu">
          <Title order={2} mb="xl">GPU Acceleration with CUDA</Title>
          
          <Paper className="p-6 bg-yellow-50 mb-6">
            <Text size="lg" mb="md">
              PyTorch tensors can leverage GPU acceleration for massive speedups in computation.
              Moving tensors to GPU memory enables parallel processing of operations.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} mb="sm">Device Management</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Get device count
if torch.cuda.is_available():
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")`} 
                />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} mb="sm">Moving Tensors to GPU</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Create tensor on GPU
gpu_tensor = torch.randn(3, 4, device='cuda')

# Move existing tensor to GPU
cpu_tensor = torch.randn(3, 4)
gpu_tensor = cpu_tensor.to('cuda')
# or
gpu_tensor = cpu_tensor.cuda()

# Move back to CPU
cpu_tensor = gpu_tensor.cpu()

# Operations must be on same device
result = gpu_tensor @ gpu_tensor.T`} 
                />
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-red-50 mt-4">
            <Title order={4} mb="sm">⚠️ Important Notes</Title>
            <List>
              <List.Item>All tensors in an operation must be on the same device</List.Item>
              <List.Item>Moving data between CPU and GPU has overhead - minimize transfers</List.Item>
              <List.Item>GPU memory is limited - monitor usage with nvidia-smi</List.Item>
            </List>
          </Paper>
        </div>

        {/* Automatic Differentiation */}
        <div id="autograd">
          <Title order={2} mb="xl">Automatic Differentiation (Autograd)</Title>
          
          <Paper className="p-6 bg-purple-50 mb-6">
            <Text size="lg">
              PyTorch's autograd engine enables automatic computation of gradients,
              which is essential for training neural networks using backpropagation.
            </Text>
            
            <Text className="mt-4">
              For a scalar function <InlineMath>{`f: \\mathbb{R}^n \\rightarrow \\mathbb{R}`}</InlineMath>, autograd computes:
            </Text>
            <BlockMath>{`\\nabla f(\\mathbf{x}) = \\left(\\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, \\ldots, \\frac{\\partial f}{\\partial x_n}\\right)^T`}</BlockMath>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} mb="sm">Basic Autograd Example</Title>
                <Text className="mb-3">
                  <strong>Chain Rule Application:</strong> For <InlineMath>{`f(g(x))`}</InlineMath>, autograd computes <InlineMath>{`\\frac{df}{dx} = \\frac{df}{dg} \\cdot \\frac{dg}{dx}`}</InlineMath>
                </Text>
                <CodeBlock 
                  language="python" 
                  code={`# Enable gradient computation
x = torch.randn(3, requires_grad=True)
y = x * 2          # y = 2x
z = y * y * 3      # z = 3y² = 3(2x)² = 12x²
out = z.mean()     # out = (1/3)Σ(12x_i²) = 4Σx_i²

print(f"x: {x}")
print(f"out: {out}")

# Compute gradients using backpropagation
out.backward()

# Access gradients: ∂out/∂x = 8x/3
print(f"Gradient of x: {x.grad}")
# Mathematical derivation: ∂/∂x[4Σx_i²] = 8x`} 
                />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Gradient Control</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Detach from computation graph
x = torch.randn(3, requires_grad=True)
y = x.detach()  # y has no gradient

# Disable gradient temporarily
with torch.no_grad():
    y = x * 2  # No gradient computed

# Or use decorator
@torch.no_grad()
def inference(x):
    return x * 2`} 
                />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Gradient Accumulation</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Gradients accumulate by default
x = torch.ones(2, requires_grad=True)

# First backward pass
y = x + 2
y.backward(torch.ones_like(x))
print(x.grad)  # tensor([1., 1.])

# Second backward (accumulates)
y = x * 3
y.backward(torch.ones_like(x))
print(x.grad)  # tensor([4., 4.])

# Clear gradients
x.grad.zero_()`} 
                />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Broadcasting */}
        <div id="broadcasting">
          <Title order={2} mb="xl">Tensor Broadcasting</Title>
          
          <Paper className="p-6 bg-indigo-50 mb-6">
            <Text size="lg" mb="md">
              Broadcasting allows operations between tensors of different shapes by automatically
              expanding the smaller tensor to match the larger one's shape.
            </Text>
            
            <Text className="mb-4">
              <strong>Mathematical Foundation:</strong> For tensors <InlineMath>{`\\mathbf{A} \\in \\mathbb{R}^{m \\times n}`}</InlineMath> and <InlineMath>{`\\mathbf{b} \\in \\mathbb{R}^n`}</InlineMath>, 
              broadcasting computes <InlineMath>{`\\mathbf{A} + \\mathbf{b}`}</InlineMath> by treating <InlineMath>{`\\mathbf{b}`}</InlineMath> as <InlineMath>{`\\mathbf{1}_m \\otimes \\mathbf{b}^T`}</InlineMath>.
            </Text>
            <Text className="font-semibold">Broadcasting Rules:</Text>
            <List>
              <List.Item>Start with the rightmost dimension and work left</List.Item>
              <List.Item>Dimensions are compatible if they are equal or one is 1</List.Item>
              <List.Item>Missing dimensions are treated as 1</List.Item>
              <List.Item>Result shape: <InlineMath>{`\\max(\\text{shape}_A, \\text{shape}_B)`}</InlineMath> element-wise</List.Item>
            </List>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} mb="sm">Broadcasting Examples</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Scalar and tensor
x = torch.randn(3, 4)
y = 2.0
z = x * y  # y broadcasts to (3, 4)

# Vector and matrix
x = torch.randn(3, 4)
y = torch.randn(4)  # Shape: (4,)
z = x + y  # y broadcasts to (3, 4)

# Different dimensions
x = torch.randn(1, 3, 1)
y = torch.randn(2, 1, 4)
z = x + y  # Result shape: (2, 3, 4)`} 
                />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} mb="sm">Common Broadcasting Patterns</Title>
                <CodeBlock 
                  language="python" 
                  code={`# Normalize by mean and std
data = torch.randn(100, 3, 224, 224)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# Reshape for broadcasting
mean = mean.view(1, 3, 1, 1)
std = std.view(1, 3, 1, 1)

normalized = (data - mean) / std

# Batch operations
batch = torch.randn(32, 10)
weights = torch.randn(10)
result = batch * weights  # Applied to each sample`} 
                />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Summary */}
        <div>
          <Title order={2} className="mb-8">Summary: Tensor Fundamentals</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} mb="md">Key Concepts Covered</Title>
                <List spacing="md">
                  <List.Item>Tensor dimensions and shapes</List.Item>
                  <List.Item>Creating and initializing tensors</List.Item>
                  <List.Item>Essential tensor operations</List.Item>
                  <List.Item>GPU acceleration with CUDA</List.Item>
                  <List.Item>Automatic differentiation</List.Item>
                  <List.Item>Broadcasting mechanisms</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} mb="md">Next Steps</Title>
                <List spacing="md">
                  <List.Item>Practice tensor manipulations</List.Item>
                  <List.Item>Build simple neural network layers</List.Item>
                  <List.Item>Explore PyTorch nn.Module</List.Item>
                  <List.Item>Learn about optimizers</List.Item>
                  <List.Item>Implement backpropagation manually</List.Item>
                  <List.Item>Work with real datasets</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6 text-center">
            <Text size="lg" className="font-semibold">
              Remember: Tensors are just multi-dimensional arrays with automatic differentiation!
            </Text>
            <Text className="mt-2">
              Master these fundamentals, and you'll have a solid foundation for deep learning.
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default TensorFundamentals;