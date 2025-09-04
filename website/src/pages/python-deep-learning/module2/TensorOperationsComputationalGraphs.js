import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const TensorOperationsComputationalGraphs = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        {/* Tensor Algebra */}
        <div id="tensor-algebra">
          <Title order={1} mb="xl">
            Tensor Operations & Computational Graphs
          </Title>
          <Text size="xl" className="mb-6">
            Tensor Algebra and Operations
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={2} mb="lg">Mathematical Foundation of Tensors</Title>
            <Text className="mb-4">
              A tensor is a multi-dimensional array that generalizes scalars (0D), vectors (1D), and matrices (2D) to higher dimensions. 
              Tensors follow specific algebraic rules and transformations.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Title order={4} mb="sm">Tensor Ranks and Notation</Title>
                <List>
                  <List.Item><strong>Scalar (Rank 0):</strong> a ∈ ℝ</List.Item>
                  <List.Item><strong>Vector (Rank 1):</strong> v ∈ ℝⁿ</List.Item>
                  <List.Item><strong>Matrix (Rank 2):</strong> M ∈ ℝᵐˣⁿ</List.Item>
                  <List.Item><strong>Tensor (Rank ≥ 3):</strong> T ∈ ℝⁿ¹ˣⁿ²ˣ...ˣⁿᵈ</List.Item>
                </List>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <CodeBlock language="python" code={`import torch

# Different tensor ranks
scalar = torch.tensor(3.14)           # Rank 0: shape []
vector = torch.tensor([1, 2, 3])      # Rank 1: shape [3]
matrix = torch.tensor([[1, 2],        # Rank 2: shape [2, 2]
                       [3, 4]])
tensor_3d = torch.randn(2, 3, 4)      # Rank 3: shape [2, 3, 4]

print(f"Scalar: {scalar.shape}, ndim: {scalar.ndim}")
print(f"Vector: {vector.shape}, ndim: {vector.ndim}")
print(f"Matrix: {matrix.shape}, ndim: {matrix.ndim}")
print(f"3D Tensor: {tensor_3d.shape}, ndim: {tensor_3d.ndim}")`} />
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-green-50 mb-6">
            <Title order={4} mb="sm">Tensor Operations Taxonomy</Title>
            <Grid gutter="lg">
              <Grid.Col span={4}>
                <Paper className="p-3 bg-white">
                  <Title order={5} className="mb-2">Element-wise Operations</Title>
                  <CodeBlock language="python" code={`a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Element-wise arithmetic
add = a + b          # Broadcasting rules apply
sub = a - b
mul = a * b          # Hadamard product
div = a / b
pow = a ** 2

# Element-wise functions
relu = torch.relu(a)
sigmoid = torch.sigmoid(a.float())
exp = torch.exp(a.float())

print(f"Element-wise multiply:\\n{mul}")
print(f"ReLU activation:\\n{relu}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-3 bg-white">
                  <Title order={5} className="mb-2">Contraction Operations</Title>
                  <CodeBlock language="python" code={`# Matrix multiplication (Einstein notation: ij,jk->ik)
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.mm(A, B)    # Result: 3x5

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
batch_C = torch.bmm(batch_A, batch_B)  # Result: 10x3x5

# General tensor contraction
# Einstein summation convention
tensor1 = torch.randn(2, 3, 4)
tensor2 = torch.randn(4, 5)
result = torch.einsum('ijk,kl->ijl', tensor1, tensor2)

print(f"Matrix mult shape: {C.shape}")
print(f"Batch mult shape: {batch_C.shape}")
print(f"Einsum result shape: {result.shape}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-3 bg-white">
                  <Title order={5} className="mb-2">Reduction Operations</Title>
                  <CodeBlock language="python" code={`x = torch.randn(2, 3, 4)

# Sum reductions
sum_all = torch.sum(x)           # Scalar result
sum_dim0 = torch.sum(x, dim=0)   # Shape: [3, 4]
sum_dim12 = torch.sum(x, dim=[1, 2])  # Shape: [2]

# Other reductions
mean_val = torch.mean(x)
max_val = torch.max(x)
min_val = torch.min(x)
std_val = torch.std(x)

# Norms
l1_norm = torch.norm(x, p=1)
l2_norm = torch.norm(x, p=2)
frobenius = torch.norm(x, p='fro')

print(f"Original shape: {x.shape}")
print(f"Sum along dim 0: {sum_dim0.shape}")
print(f"L2 norm: {l2_norm:.4f}")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Broadcasting */}
        <div id="broadcasting">
          <Title order={2} mb="lg">Broadcasting Mechanics & Memory</Title>
          
            <Paper mb="xl">
            <Title order={2} mb="lg">Broadcasting Rules</Title>
            <Text className="mb-4">
              Broadcasting allows operations between tensors of different shapes by virtually expanding smaller tensors. 
              Rules: dimensions are aligned from the right, and dimensions of size 1 can be broadcast.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} mb="sm">Broadcasting Examples</Title>
                  <CodeBlock language="python" code={`import torch

# Example 1: Vector + Scalar
vector = torch.tensor([1, 2, 3])      # Shape: [3]
scalar = torch.tensor(10)             # Shape: []
result1 = vector + scalar             # Shape: [3]
print(f"Vector + Scalar: {result1}")

# Example 2: Matrix + Vector (column-wise)
matrix = torch.tensor([[1, 2, 3],     # Shape: [2, 3]
                       [4, 5, 6]])
vector = torch.tensor([10, 20, 30])   # Shape: [3]
result2 = matrix + vector             # Shape: [2, 3]
print(f"Matrix + Vector:\\n{result2}")

# Example 3: Matrix + Vector (row-wise)
matrix = torch.tensor([[1, 2, 3],     # Shape: [2, 3]
                       [4, 5, 6]])
vector = torch.tensor([[10], [20]])   # Shape: [2, 1]
result3 = matrix + vector             # Shape: [2, 3]
print(f"Matrix + Row Vector:\\n{result3}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper p="md">
                  <Title order={4} mb="sm">Broadcasting Step-by-Step</Title>
                  <CodeBlock language="python" code={`# Step-by-step broadcasting analysis
def analyze_broadcasting(a_shape, b_shape):
    """Analyze if two shapes can be broadcast together"""
    print(f"Shape A: {a_shape}")
    print(f"Shape B: {b_shape}")
    
    # Align from right
    max_dim = max(len(a_shape), len(b_shape))
    a_aligned = [1] * (max_dim - len(a_shape)) + list(a_shape)
    b_aligned = [1] * (max_dim - len(b_shape)) + list(b_shape)
    
    print(f"A aligned: {a_aligned}")
    print(f"B aligned: {b_aligned}")
    
    # Check compatibility
    result_shape = []
    compatible = True
    
    for i, (a_dim, b_dim) in enumerate(zip(a_aligned, b_aligned)):
        if a_dim == 1:
            result_shape.append(b_dim)
        elif b_dim == 1:
            result_shape.append(a_dim)
        elif a_dim == b_dim:
            result_shape.append(a_dim)
        else:
            compatible = False
            break
    
    print(f"Compatible: {compatible}")
    if compatible:
        print(f"Result shape: {result_shape}")
    print("-" * 40)

# Test cases
analyze_broadcasting([3], [2, 3])      # Compatible
analyze_broadcasting([1, 4], [3, 1])   # Compatible  
analyze_broadcasting([2, 3], [3, 2])   # Incompatible`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-purple-50">
            <Title order={4} mb="sm">Memory Implications of Broadcasting</Title>
            <CodeBlock language="python" code={`# Memory analysis of broadcasting
def memory_analysis():
    # Large matrix
    large_matrix = torch.randn(1000, 1000)  # 4MB (float32)
    small_vector = torch.randn(1000)        # 4KB
    
    print(f"Large matrix memory: {large_matrix.element_size() * large_matrix.numel() / 1024**2:.2f} MB")
    print(f"Small vector memory: {small_vector.element_size() * small_vector.numel() / 1024:.2f} KB")
    
    # Broadcasting doesn't create copies in memory!
    result = large_matrix + small_vector
    print(f"Result memory: {result.element_size() * result.numel() / 1024**2:.2f} MB")
    
    # But be careful with assignments that require expansion
    expanded_vector = small_vector.expand(1000, 1000)  # View, no copy
    print(f"Expanded vector is view: {expanded_vector.data_ptr() == small_vector.data_ptr()}")
    
    # Explicit expansion creates copies
    copied_vector = small_vector.expand(1000, 1000).contiguous()
    print(f"Copied vector memory: {copied_vector.element_size() * copied_vector.numel() / 1024**2:.2f} MB")

memory_analysis()`} />
          </Paper>
        </div>

        {/* Storage and Views */}
        <div id="storage-views">
          <Title order={2} mb="lg">Storage, Views & Memory Layout</Title>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={2} mb="lg">PyTorch Memory Model</Title>
            <Text className="mb-4">
              PyTorch tensors have a two-level memory model: Storage (actual data) and Tensor (metadata including shape, strides, offset).
              Multiple tensors can share the same storage with different views.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Storage and Views</Title>
                  <CodeBlock language="python" code={`import torch

# Create original tensor
x = torch.arange(12).reshape(3, 4)
print(f"Original tensor:\\n{x}")
print(f"Storage: {x.storage()}")
print(f"Data pointer: {x.data_ptr()}")

# Create view (shares storage)
y = x.view(2, 6)
print(f"\\nView tensor:\\n{y}")
print(f"Same storage: {x.storage().data_ptr() == y.storage().data_ptr()}")

# Modify original, view changes too
x[0, 0] = 999
print(f"\\nAfter modifying x[0,0]:")
print(f"x:\\n{x}")
print(f"y (view):\\n{y}")

# Transpose creates a view with different strides
z = x.t()
print(f"\\nTranspose strides: {z.stride()}")
print(f"Original strides: {x.stride()}")
print(f"Same storage: {x.data_ptr() == z.data_ptr()}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Contiguous vs Non-contiguous</Title>
                  <CodeBlock language="python" code={`# Contiguous tensor
x = torch.arange(12).reshape(3, 4)
print(f"x is contiguous: {x.is_contiguous()}")
print(f"x strides: {x.stride()}")

# Non-contiguous after transpose
y = x.t()
print(f"\\ny is contiguous: {y.is_contiguous()}")
print(f"y strides: {y.stride()}")

# Make contiguous (creates copy)
z = y.contiguous()
print(f"\\nz is contiguous: {z.is_contiguous()}")
print(f"z shares storage with y: {y.data_ptr() == z.data_ptr()}")

# Performance implications
import time

# Large tensor operations
large_x = torch.randn(1000, 1000)
large_y = large_x.t()  # Non-contiguous

# Time contiguous vs non-contiguous operations
start = time.time()
for _ in range(100):
    result1 = large_x.sum()
print(f"Contiguous sum time: {time.time() - start:.4f}s")

start = time.time()
for _ in range(100):
    result2 = large_y.sum()
print(f"Non-contiguous sum time: {time.time() - start:.4f}s")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-green-50">
            <Title order={4} mb="sm">Memory Layout and Strides</Title>
            <CodeBlock language="python" code={`# Understanding strides
def explain_strides(tensor):
    """Explain how strides work for a tensor"""
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor strides: {tensor.stride()}")
    print(f"Storage size: {tensor.storage().size()}")
    
    # Calculate memory offset for element [i, j]
    if tensor.ndim == 2:
        print("\\nMemory layout:")
        for i in range(min(tensor.shape[0], 3)):
            for j in range(min(tensor.shape[1], 3)):
                offset = i * tensor.stride(0) + j * tensor.stride(1)
                print(f"  [{i},{j}] -> storage[{offset}] = {tensor.storage()[offset]}")

# Row-major (C-style) layout
x = torch.arange(6).reshape(2, 3)
print("Row-major layout:")
explain_strides(x)

print("\\n" + "="*50 + "\\n")

# Column-major after transpose
y = x.t()
print("After transpose (column-major access):")
explain_strides(y)

print("\\n" + "="*50 + "\\n")

# Custom strides with as_strided
z = torch.as_strided(torch.arange(12), (3, 4), (4, 1))
print("Custom strided tensor:")
explain_strides(z)`} />
          </Paper>
        </div>

        {/* Computational Graphs */}
        <div id="computational-graphs">
          <Title order={2} mb="lg">Introduction to Computational Graphs</Title>
          
            <Paper mb="xl">
            <Title order={2} mb="lg">Computational Graph Concepts</Title>
            <Text className="mb-4">
              A computational graph represents the flow of operations in a computation. Nodes represent variables or operations, 
              and edges represent dependencies. PyTorch builds these graphs dynamically during forward passes.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} mb="sm">Basic Graph Construction</Title>
                  <CodeBlock language="python" code={`import torch
import torch.nn.functional as F

# Enable gradient computation
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Build computation graph
# z = x² + 2xy + y²
z1 = x ** 2           # Node: pow
z2 = 2 * x * y        # Nodes: mul, mul, mul
z3 = y ** 2           # Node: pow
z = z1 + z2 + z3      # Nodes: add, add

print(f"Result: z = {z}")
print(f"x.grad_fn: {x.grad_fn}")  # None (leaf node)
print(f"z.grad_fn: {z.grad_fn}")  # AddBackward

# Visualize graph structure
print(f"z.grad_fn.next_functions: {z.grad_fn.next_functions}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper p="md">
                  <Title order={4} mb="sm">Graph Properties</Title>
                  <CodeBlock language="python" code={`# Analyze graph properties
def analyze_graph(tensor):
    """Analyze computational graph properties"""
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Is leaf: {tensor.is_leaf}")
    print(f"Gradient function: {tensor.grad_fn}")
    print(f"Retains grad: {tensor.retains_grad if hasattr(tensor, 'retains_grad') else 'N/A'}")

x = torch.tensor(1.0, requires_grad=True)
print("Input tensor x:")
analyze_graph(x)

y = x ** 2
print("\\nIntermediate tensor y = x²:")
analyze_graph(y)

# Detached tensor (breaks graph)
z = y.detach()
print("\\nDetached tensor z:")
analyze_graph(z)

# In-place operations affect graph
w = x.clone()
w.add_(1)  # In-place addition
print("\\nAfter in-place operation:")
analyze_graph(w)`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-purple-50">
            <Title order={4} mb="sm">Advanced Graph Operations</Title>
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-3 bg-white">
                  <Title order={5} className="mb-2">Graph Inspection</Title>
                  <CodeBlock language="python" code={`# Deep graph inspection
def trace_graph(tensor, depth=0):
    """Recursively trace computational graph"""
    indent = "  " * depth
    print(f"{indent}{type(tensor.grad_fn).__name__ if tensor.grad_fn else 'Leaf'}")
    
    if tensor.grad_fn and depth < 3:  # Limit recursion
        for next_func, _ in tensor.grad_fn.next_functions:
            if next_func is not None:
                # Find tensor associated with this function
                for var in [tensor]:  # Simplified
                    if hasattr(var, 'grad_fn') and var.grad_fn == next_func:
                        trace_graph(var, depth + 1)

# Complex computation
x = torch.tensor(2.0, requires_grad=True)
y = torch.sin(x) + torch.cos(x ** 2) + torch.exp(-x)
print("Graph structure for complex function:")
trace_graph(y)`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-3 bg-white">
                  <Title order={5} className="mb-2">Graph Manipulation</Title>
                  <CodeBlock language="python" code={`# Graph manipulation techniques
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 1. Retain gradients for intermediate tensors
y = x ** 2
y.retain_grad()  # Keep gradient after backward pass

z = y.sum()
z.backward()

print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")  # Available due to retain_grad()

# 2. Hook functions for intermediate gradients
def gradient_hook(grad):
    print(f"Gradient hook called with: {grad}")
    return grad  # Can modify gradient here

x.grad.zero_()
y = x ** 3
handle = y.register_hook(gradient_hook)
z = y.sum()
z.backward()

# 3. Context managers for graph control
with torch.no_grad():
    # Operations here don't build graph
    w = x ** 4
    print(f"w requires grad: {w.requires_grad}")

# Clean up
handle.remove()`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Summary */}
        <div>
          <Title order={2} className="mb-8">Summary: Tensor Operations & Computational Graphs</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={2} mb="lg"> Key Tensor Concepts</Title>
                <List spacing="md">
                  <List.Item>Tensors are multi-dimensional arrays with algebraic properties</List.Item>
                  <List.Item>Broadcasting enables operations between different tensor shapes</List.Item>
                  <List.Item>Memory views and strides optimize storage and computation</List.Item>
                  <List.Item>Contiguous layout affects performance of operations</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={2} mb="lg">Computational Graph Insights</Title>
                <List spacing="md">
                  <List.Item>Dynamic graphs built during forward pass enable flexibility</List.Item>
                  <List.Item>Gradient functions track operations for automatic differentiation</List.Item>
                  <List.Item>Memory sharing between tensors requires careful consideration</List.Item>
                  <List.Item>Graph manipulation tools enable advanced optimization techniques</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6 text-center">
            <Text size="lg" className="font-semibold">
              Understanding tensor operations and computational graphs is fundamental to PyTorch mastery
            </Text>
            <Text className="mt-2">
              These concepts enable efficient implementation of complex deep learning algorithms
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default TensorOperationsComputationalGraphs;