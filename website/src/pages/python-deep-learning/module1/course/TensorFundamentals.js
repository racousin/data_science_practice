import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, Table, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const TensorFundamentals = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Tensor Fundamentals in PyTorch
          </Title>
          <Text size="xl" className="text-center mb-6">
            Understanding the Building Blocks of Deep Learning
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" className="mb-4">
                Tensors are the fundamental data structure in deep learning frameworks.
                They generalize matrices to arbitrary dimensions and enable efficient computation on GPUs.
              </Text>
              <List>
                <List.Item>Scalar: 0-dimensional tensor (single number)</List.Item>
                <List.Item>Vector: 1-dimensional tensor (array of numbers)</List.Item>
                <List.Item>Matrix: 2-dimensional tensor (2D array)</List.Item>
                <List.Item>3D+ Tensor: Higher dimensional arrays</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Tensor Dimensions */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">Understanding Tensor Dimensions</Title>
          
          <Grid gutter="xl">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} className="mb-3">Scalar (0D)</Title>
                <Code block language="python">{`import torch

# Scalar - single value
scalar = torch.tensor(3.14)
print(scalar.shape)  # torch.Size([])
print(scalar.ndim)   # 0`}</Code>
              </Paper>
              
              <Paper className="p-4 bg-gray-50 mt-4">
                <Title order={4} className="mb-3">Vector (1D)</Title>
                <Code block language="python">{`# Vector - array of values
vector = torch.tensor([1, 2, 3, 4])
print(vector.shape)  # torch.Size([4])
print(vector.ndim)   # 1`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} className="mb-3">Matrix (2D)</Title>
                <Code block language="python">{`# Matrix - 2D array
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(matrix.shape)  # torch.Size([2, 3])
print(matrix.ndim)   # 2`}</Code>
              </Paper>
              
              <Paper className="p-4 bg-gray-50 mt-4">
                <Title order={4} className="mb-3">3D Tensor</Title>
                <Code block language="python">{`# 3D Tensor - e.g., RGB image
tensor_3d = torch.randn(3, 224, 224)
# [channels, height, width]
print(tensor_3d.shape)  # torch.Size([3, 224, 224])
print(tensor_3d.ndim)   # 3`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Creating Tensors */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">Creating Tensors in PyTorch</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Stack spacing="md">
                <Paper className="p-4 bg-green-50">
                  <Title order={4} className="mb-3">From Python Lists</Title>
                  <Code block language="python">{`# From list
tensor = torch.tensor([1, 2, 3])

# From nested lists (2D)
matrix = torch.tensor([[1, 2], 
                       [3, 4]])`}</Code>
                </Paper>
                
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} className="mb-3">Random Initialization</Title>
                  <Code block language="python">{`# Uniform random [0, 1)
rand_tensor = torch.rand(3, 4)

# Normal distribution (μ=0, σ=1)
randn_tensor = torch.randn(3, 4)

# Random integers
randint_tensor = torch.randint(0, 10, (3, 4))`}</Code>
                </Paper>
              </Stack>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Stack spacing="md">
                <Paper className="p-4 bg-purple-50">
                  <Title order={4} className="mb-3">Special Tensors</Title>
                  <Code block language="python">{`# Zeros
zeros = torch.zeros(3, 4)

# Ones  
ones = torch.ones(3, 4)

# Identity matrix
eye = torch.eye(3)

# Filled with specific value
full = torch.full((3, 4), 7.0)`}</Code>
                </Paper>
                
                <Paper className="p-4 bg-orange-50">
                  <Title order={4} className="mb-3">From NumPy</Title>
                  <Code block language="python">{`import numpy as np

# NumPy to Tensor
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# Tensor to NumPy
tensor = torch.tensor([1, 2, 3])
np_array = tensor.numpy()`}</Code>
                </Paper>
              </Stack>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Tensor Operations */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">Essential Tensor Operations</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-gray-50 mb-4">
                <Title order={4} className="mb-3">Arithmetic Operations</Title>
                <Code block language="python">{`# Element-wise operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
c = a + b  # or torch.add(a, b)

# Multiplication (element-wise)
d = a * b  # or torch.mul(a, b)

# Matrix multiplication
x = torch.randn(2, 3)
y = torch.randn(3, 4)
z = torch.mm(x, y)  # or x @ y  # Result: (2, 4)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Reshaping Operations</Title>
                <Code block language="python">{`# Reshape
x = torch.randn(4, 6)
y = x.view(2, 12)  # Must be compatible
z = x.view(-1, 8)  # -1 infers dimension

# Squeeze & Unsqueeze
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()  # Remove dims of size 1
z = y.unsqueeze(0)  # Add dim at position 0`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Aggregation Operations</Title>
                <Code block language="python">{`# Aggregations
x = torch.randn(3, 4)

mean_val = x.mean()
sum_val = x.sum()
max_val = x.max()
min_val = x.min()

# Along specific dimension
mean_rows = x.mean(dim=0)  # Mean of each column
sum_cols = x.sum(dim=1)    # Sum of each row`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 5: GPU Acceleration */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">GPU Acceleration with CUDA</Title>
          
          <Paper className="p-6 bg-yellow-50 mb-6">
            <Text size="lg" className="mb-4">
              PyTorch tensors can leverage GPU acceleration for massive speedups in computation.
              Moving tensors to GPU memory enables parallel processing of operations.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} className="mb-3">Device Management</Title>
                <Code block language="python">{`# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Get device count
if torch.cuda.is_available():
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} className="mb-3">Moving Tensors to GPU</Title>
                <Code block language="python">{`# Create tensor on GPU
gpu_tensor = torch.randn(3, 4, device='cuda')

# Move existing tensor to GPU
cpu_tensor = torch.randn(3, 4)
gpu_tensor = cpu_tensor.to('cuda')
# or
gpu_tensor = cpu_tensor.cuda()

# Move back to CPU
cpu_tensor = gpu_tensor.cpu()

# Operations must be on same device
result = gpu_tensor @ gpu_tensor.T`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-red-50 mt-4">
            <Title order={4} className="mb-3">⚠️ Important Notes</Title>
            <List>
              <List.Item>All tensors in an operation must be on the same device</List.Item>
              <List.Item>Moving data between CPU and GPU has overhead - minimize transfers</List.Item>
              <List.Item>GPU memory is limited - monitor usage with nvidia-smi</List.Item>
            </List>
          </Paper>
        </div>

        {/* Slide 6: Automatic Differentiation */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">Automatic Differentiation (Autograd)</Title>
          
          <Paper className="p-6 bg-purple-50 mb-6">
            <Text size="lg">
              PyTorch's autograd engine enables automatic computation of gradients,
              which is essential for training neural networks using backpropagation.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} className="mb-3">Basic Autograd Example</Title>
                <Code block language="python">{`# Enable gradient computation
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y * y * 3
out = z.mean()

print(f"x: {x}")
print(f"out: {out}")

# Compute gradients
out.backward()

# Access gradients
print(f"Gradient of x: {x.grad}")
# ∂out/∂x = ∂(3*(2x)²/3)/∂x = 4x`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Gradient Control</Title>
                <Code block language="python">{`# Detach from computation graph
x = torch.randn(3, requires_grad=True)
y = x.detach()  # y has no gradient

# Disable gradient temporarily
with torch.no_grad():
    y = x * 2  # No gradient computed

# Or use decorator
@torch.no_grad()
def inference(x):
    return x * 2`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Gradient Accumulation</Title>
                <Code block language="python">{`# Gradients accumulate by default
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
x.grad.zero_()`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 7: Broadcasting */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">Tensor Broadcasting</Title>
          
          <Paper className="p-6 bg-indigo-50 mb-6">
            <Text size="lg" className="mb-4">
              Broadcasting allows operations between tensors of different shapes by automatically
              expanding the smaller tensor to match the larger one's shape.
            </Text>
            <Text className="font-semibold">Broadcasting Rules:</Text>
            <List>
              <List.Item>Start with the rightmost dimension and work left</List.Item>
              <List.Item>Dimensions are compatible if they are equal or one is 1</List.Item>
              <List.Item>Missing dimensions are treated as 1</List.Item>
            </List>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} className="mb-3">Broadcasting Examples</Title>
                <Code block language="python">{`# Scalar and tensor
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
z = x + y  # Result shape: (2, 3, 4)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50">
                <Title order={4} className="mb-3">Common Broadcasting Patterns</Title>
                <Code block language="python">{`# Normalize by mean and std
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
result = batch * weights  # Applied to each sample`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 8: Practical Example - Linear Layer */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">Practical Example: Implementing a Linear Layer</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg" className="mb-4">
              Let's implement a simple linear layer from scratch using tensor operations:
            </Text>
            <BlockMath>{`y = Wx + b`}</BlockMath>
          </Paper>
          
          <Code block language="python">{`import torch
import torch.nn.functional as F

class LinearLayer:
    def __init__(self, in_features, out_features):
        # Initialize weights with Xavier initialization
        self.weight = torch.randn(out_features, in_features) * (2 / in_features)**0.5
        self.bias = torch.zeros(out_features)
        
        # Enable gradients
        self.weight.requires_grad = True
        self.bias.requires_grad = True
    
    def forward(self, x):
        # x shape: (batch_size, in_features)
        # weight shape: (out_features, in_features)
        # output shape: (batch_size, out_features)
        return x @ self.weight.T + self.bias

# Example usage
layer = LinearLayer(784, 128)  # 784 inputs, 128 outputs
batch = torch.randn(32, 784)   # Batch of 32 samples

# Forward pass
output = layer.forward(batch)
print(f"Output shape: {output.shape}")  # (32, 128)

# Compute loss and gradients
target = torch.randn(32, 128)
loss = F.mse_loss(output, target)
loss.backward()

print(f"Weight gradient shape: {layer.weight.grad.shape}")
print(f"Bias gradient shape: {layer.bias.grad.shape}")`}</Code>
        </div>

        {/* Slide 9: Best Practices */}
        <div data-slide className="min-h-[500px]">
          <Title order={2} className="mb-6">Best Practices and Tips</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">✅ Do's</Title>
                <List>
                  <List.Item>Use vectorized operations instead of loops</List.Item>
                  <List.Item>Keep tensors on the same device</List.Item>
                  <List.Item>Use torch.no_grad() during inference</List.Item>
                  <List.Item>Clear gradients between training steps</List.Item>
                  <List.Item>Use appropriate data types (float32 vs float16)</List.Item>
                  <List.Item>Profile your code to find bottlenecks</List.Item>
                </List>
              </Paper>
              
              <Paper className="p-4 bg-blue-50 mt-4">
                <Title order={4} className="mb-3">Memory Management</Title>
                <Code block language="python">{`# Free GPU memory
torch.cuda.empty_cache()

# Check memory usage
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())

# Delete unused tensors
del large_tensor
torch.cuda.empty_cache()`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} className="mb-3">❌ Don'ts</Title>
                <List>
                  <List.Item>Don't use Python loops for tensor operations</List.Item>
                  <List.Item>Don't mix CPU and GPU tensors in operations</List.Item>
                  <List.Item>Don't forget to zero gradients</List.Item>
                  <List.Item>Don't keep unnecessary computation graphs</List.Item>
                  <List.Item>Don't ignore out-of-memory errors</List.Item>
                </List>
              </Paper>
              
              <Paper className="p-4 bg-yellow-50 mt-4">
                <Title order={4} className="mb-3">Performance Tips</Title>
                <Code block language="python">{`# Use in-place operations when possible
x.add_(1)  # In-place addition
x.mul_(2)  # In-place multiplication

# Batch operations
# Bad: Loop through samples
for i in range(batch_size):
    output[i] = model(input[i])

# Good: Process entire batch
output = model(input)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 10: Summary */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={2} className="text-center mb-8">Summary: Tensor Fundamentals</Title>
          
          <div className="max-w-4xl mx-auto">
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                  <Title order={3} className="mb-4">Key Concepts Covered</Title>
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
                  <Title order={3} className="mb-4">Next Steps</Title>
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
        </div>

      </Stack>
    </Container>
  );
};

export default TensorFundamentals;