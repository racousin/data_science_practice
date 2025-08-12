import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Table, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const PyTorchBasics = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        {/* Introduction */}
        <div id="installation">
          <Title order={1} className="mb-6">
            PyTorch Basics
          </Title>
          <Text size="xl" className="mb-6">
            Getting Started with PyTorch Framework
          </Text>
          <Paper className="p-6 bg-blue-50 mb-6">
            <Text size="lg" className="mb-4">
              PyTorch is a dynamic, flexible deep learning framework that provides:
            </Text>
            <List>
              <List.Item>Dynamic computation graphs (define-by-run)</List.Item>
              <List.Item>Pythonic and intuitive API</List.Item>
              <List.Item>Excellent debugging capabilities</List.Item>
              <List.Item>Strong GPU acceleration support</List.Item>
              <List.Item>Rich ecosystem of tools and libraries</List.Item>
            </List>
          </Paper>
        </div>

        {/* Installation and Setup */}
        <div id="installation">
          <Title order={2} className="mb-6">Installation and Setup</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50 mb-4">
                <Title order={4} className="mb-3">CPU Version</Title>
                <CodeBlock language="bash" code={`# Using pip
pip install torch torchvision torchaudio

# Using conda
conda install pytorch torchvision torchaudio -c pytorch`} />
              </Paper>
              
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Verify Installation</Title>
                <CodeBlock language="python" code={`import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50 mb-4">
                <Title order={4} className="mb-3">GPU Version (CUDA)</Title>
                <CodeBlock language="bash" code={`# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check compatibility at: https://pytorch.org/get-started/locally/`} />
              </Paper>
              
              <Paper className="p-4 bg-purple-50">
                <Title order={4} className="mb-3">Environment Setup</Title>
                <CodeBlock language="python" code={`# Recommended imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)`} />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Tensors vs NumPy Arrays */}
        <div id="tensors-vs-numpy">
          <Title order={2} className="mb-6">Tensors vs NumPy Arrays</Title>
          
          <Paper className="p-4 bg-gray-50 mb-4">
            <Title order={4} className="mb-3">Key Differences</Title>
            <Table>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Aspect</Table.Th>
                  <Table.Th>NumPy Arrays</Table.Th>
                  <Table.Th>PyTorch Tensors</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                <Table.Tr>
                  <Table.Td>Device Support</Table.Td>
                  <Table.Td>CPU only</Table.Td>
                  <Table.Td>CPU and GPU</Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Automatic Differentiation</Table.Td>
                  <Table.Td>Not available</Table.Td>
                  <Table.Td>Built-in (autograd)</Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Deep Learning</Table.Td>
                  <Table.Td>Not optimized</Table.Td>
                  <Table.Td>Native support</Table.Td>
                </Table.Tr>
                <Table.Tr>
                  <Table.Td>Memory Sharing</Table.Td>
                  <Table.Td>N/A</Table.Td>
                  <Table.Td>Shares memory with NumPy</Table.Td>
                </Table.Tr>
              </Table.Tbody>
            </Table>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">NumPy to Tensor</Title>
                <CodeBlock language="python" code={`import numpy as np
import torch

# NumPy array to tensor
np_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.from_numpy(np_array)
print(f"Original: {np_array}")
print(f"Tensor: {tensor}")
print(f"Data type: {tensor.dtype}")

# Convert data type
tensor_float = tensor.float()
print(f"Float tensor: {tensor_float.dtype}")

# Memory is shared!
np_array[0, 0] = 999
print(f"After modifying NumPy: {tensor}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Tensor to NumPy</Title>
                <CodeBlock language="python" code={`# Tensor to NumPy array
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
np_array = tensor.numpy()
print(f"Tensor: {tensor}")
print(f"NumPy: {np_array}")

# GPU tensors must move to CPU first
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1, 2, 3]).cuda()
    # This would fail: gpu_tensor.numpy()
    cpu_array = gpu_tensor.cpu().numpy()
    print(f"GPU to NumPy: {cpu_array}")
    
# Detached tensors (no gradients)
tensor.requires_grad = True
detached_array = tensor.detach().numpy()`} />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Tensor Attributes */}
        <div id="tensor-attributes">
          <Title order={2} className="mb-6">Tensor Attributes</Title>
          
          <Paper className="p-4 bg-gray-50 mb-4">
            <Title order={4} className="mb-3">Understanding Tensor Properties</Title>
            <CodeBlock language="python" code={`import torch

# Create a sample tensor
tensor = torch.randn(2, 3, 4, dtype=torch.float32, device='cpu')

print(f"Tensor: {tensor.shape}")
print(f"Shape/Size: {tensor.shape} / {tensor.size()}")
print(f"Number of dimensions: {tensor.ndim} / {tensor.dim()}")
print(f"Data type: {tensor.dtype}")
print(f"Device: {tensor.device}")
print(f"Requires gradient: {tensor.requires_grad}")
print(f"Memory layout: {tensor.layout}")
print(f"Total elements: {tensor.numel()}")
print(f"Element size (bytes): {tensor.element_size()}")`} />
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Data Types</Title>
                <CodeBlock language="python" code={`# Common data types
float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
double_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
int_tensor = torch.tensor([1, 2], dtype=torch.int32)
long_tensor = torch.tensor([1, 2], dtype=torch.int64)
bool_tensor = torch.tensor([True, False], dtype=torch.bool)

# Type conversion
x = torch.randn(3, 4)
x_int = x.int()          # to integer
x_float = x.float()      # to float32
x_double = x.double()    # to float64
x_bool = x.bool()        # to boolean

# Check and change dtype
print(f"Original dtype: {x.dtype}")
x = x.type(torch.float64)
print(f"New dtype: {x.dtype}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Device Management</Title>
                <CodeBlock language="python" code={`# Device specification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensor on specific device
cpu_tensor = torch.randn(3, 4, device='cpu')
if torch.cuda.is_available():
    gpu_tensor = torch.randn(3, 4, device='cuda')
    # or
    gpu_tensor = torch.randn(3, 4).cuda()

# Move tensors between devices
tensor = torch.randn(3, 4)
tensor = tensor.to(device)  # Move to device
tensor = tensor.cpu()       # Move to CPU
if torch.cuda.is_available():
    tensor = tensor.cuda()  # Move to GPU

# Check device
print(f"Tensor device: {tensor.device}")`} />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Indexing and Slicing */}
        <div id="indexing-slicing">
          <Title order={2} className="mb-6">Indexing and Slicing</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-gray-50 mb-4">
                <Title order={4} className="mb-3">Basic Indexing</Title>
                <CodeBlock language="python" code={`# Create sample tensor
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
print(f"Original tensor:\\n{x}")

# Single element
print(f"Element [1,2]: {x[1, 2]}")

# Single row
print(f"Row 1: {x[1]}")
print(f"Row 1 (explicit): {x[1, :]}")

# Single column
print(f"Column 2: {x[:, 2]}")

# Multiple elements
print(f"First 2 rows: {x[:2]}")
print(f"Last 2 columns: {x[:, -2:]}")`} />
              </Paper>
              
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Advanced Indexing</Title>
                <CodeBlock language="python" code={`# Boolean indexing
x = torch.randn(3, 4)
mask = x > 0
positive_values = x[mask]
print(f"Positive values: {positive_values}")

# Modify using boolean indexing
x[x < 0] = 0  # Set negative values to zero
print(f"After setting negatives to 0:\\n{x}")

# Fancy indexing
indices = torch.tensor([0, 2])
selected_rows = x[indices]
print(f"Selected rows:\\n{selected_rows}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50 mb-4">
                <Title order={4} className="mb-3">Slicing Patterns</Title>
                <CodeBlock language="python" code={`x = torch.arange(24).reshape(4, 6)
print(f"Original:\\n{x}")

# Step slicing
print(f"Every 2nd row: {x[::2]}")
print(f"Every 2nd column: {x[:, ::2]}")

# Reverse slicing
print(f"Reversed rows: {x[::-1]}")
print(f"Reversed columns: {x[:, ::-1]}")

# Complex slicing
print(f"Submatrix: {x[1:3, 2:5]}")

# Ellipsis (...) for unknown dimensions
tensor_3d = torch.randn(2, 3, 4)
print(f"Using ellipsis: {tensor_3d[..., 0].shape}")  # Last dim index 0
print(f"Equivalent to: {tensor_3d[:, :, 0].shape}")`} />
              </Paper>
              
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">In-place Operations</Title>
                <CodeBlock language="python" code={`x = torch.tensor([[1, 2], [3, 4]])

# In-place operations (modify original tensor)
x[0, 0] = 99  # Direct assignment
print(f"After assignment: {x}")

# In-place arithmetic
x.add_(10)    # Add 10 to all elements
x.mul_(2)     # Multiply all elements by 2
print(f"After in-place ops: {x}")

# Copy vs view
y = x.clone()  # Creates a copy
z = x.view(-1) # Creates a view (shares memory)
x[0, 0] = 999
print(f"Original: {x}")
print(f"Copy: {y}")      # Unchanged
print(f"View: {z}")      # Changed!`} />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Mathematical Operations */}
        <div id="tensor-math">
          <Title order={2} className="mb-6">Mathematical Operations</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={4}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Element-wise Operations</Title>
                <CodeBlock language="python" code={`a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Arithmetic
add = a + b  # or torch.add(a, b)
sub = a - b  # or torch.sub(a, b)  
mul = a * b  # or torch.mul(a, b)
div = a / b  # or torch.div(a, b)
pow = a ** 2 # or torch.pow(a, 2)

# Mathematical functions
sqrt = torch.sqrt(a.float())
exp = torch.exp(a.float())
log = torch.log(a.float())
sin = torch.sin(a.float())
cos = torch.cos(a.float())

print(f"Addition: {add}")
print(f"Square root: {sqrt}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={4}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Linear Algebra</Title>
                <CodeBlock language="python" code={`# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Matrix multiplication
C = torch.mm(A, B)      # 2D matrices
C = A @ B               # Alternative syntax
C = torch.matmul(A, B)  # General matrix multiply

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
batch_C = torch.bmm(batch_A, batch_B)  # (10, 3, 5)

# Vector operations
v1 = torch.randn(5)
v2 = torch.randn(5)
dot_product = torch.dot(v1, v2)
cross_product = torch.cross(v1[:3], v2[:3])

print(f"Matrix shape: {C.shape}")
print(f"Dot product: {dot_product}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={4}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">Reductions & Statistics</Title>
                <CodeBlock language="python" code={`x = torch.randn(3, 4)

# Reduction operations
total_sum = torch.sum(x)        # All elements
sum_dim0 = torch.sum(x, dim=0)  # Sum along rows
sum_dim1 = torch.sum(x, dim=1)  # Sum along columns

# Statistics
mean_val = torch.mean(x)
std_val = torch.std(x)
var_val = torch.var(x)
max_val = torch.max(x)
min_val = torch.min(x)

# Argmax/Argmin
max_idx = torch.argmax(x)
max_idx_dim0 = torch.argmax(x, dim=0)

# Other reductions
norm_l2 = torch.norm(x)         # L2 norm
norm_l1 = torch.norm(x, p=1)    # L1 norm
trace = torch.trace(x[:3, :3])  # Matrix trace

print(f"Sum along dim 0: {sum_dim0}")
print(f"Mean: {mean_val:.3f}")`} />
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-purple-50 mt-4">
            <Title order={4} className="mb-3">Advanced Mathematical Operations</Title>
            <CodeBlock language="python" code={`# Eigenvalues and eigenvectors
matrix = torch.randn(3, 3)
eigenvals, eigenvecs = torch.linalg.eig(matrix)

# SVD decomposition  
U, S, V = torch.linalg.svd(matrix)

# QR decomposition
Q, R = torch.linalg.qr(matrix)

# Determinant and inverse
det = torch.linalg.det(matrix)
inv_matrix = torch.linalg.inv(matrix)

# Solving linear systems: Ax = b
A = torch.randn(3, 3)
b = torch.randn(3)
x = torch.linalg.solve(A, b)

print(f"Determinant: {det:.3f}")
print(f"Solution shape: {x.shape}")`} />
          </Paper>
        </div>

        {/* Summary */}
        <div>
          <Title order={2} className="mb-8">Summary: PyTorch Basics</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} className="mb-4">Key Concepts Covered</Title>
                <List spacing="md">
                  <List.Item>PyTorch installation and setup</List.Item>
                  <List.Item>Tensors vs NumPy arrays</List.Item>
                  <List.Item>Tensor attributes and properties</List.Item>
                  <List.Item>Indexing and slicing operations</List.Item>
                  <List.Item>Mathematical operations</List.Item>
                  <List.Item>Device management (CPU/GPU)</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} className="mb-4">Best Practices</Title>
                <List spacing="md">
                  <List.Item>Use appropriate data types for memory efficiency</List.Item>
                  <List.Item>Keep tensors on same device for operations</List.Item>
                  <List.Item>Use vectorized operations over loops</List.Item>
                  <List.Item>Understand memory sharing between tensors</List.Item>
                  <List.Item>Use in-place operations carefully</List.Item>
                  <List.Item>Set random seeds for reproducibility</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6 text-center">
            <Text size="lg" className="font-semibold">
              PyTorch provides a seamless transition from NumPy with GPU acceleration and autograd!
            </Text>
            <Text className="mt-2">
              Master these basics to build a solid foundation for deep learning development.
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default PyTorchBasics;