import React from 'react';
import { Title, Text, Stack, Grid, Timeline, Box, List, Table, Alert } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import { BookOpen, Cpu, Network, Brain, AlertCircle } from 'lucide-react';
import Architecture from './Introduction/Architecture';

const Introduction = () => {
  return (
    <Stack spacing="xl" className="w-full">
      <Title order={1} id="introduction">Introduction</Title>
      
      <Title order={2} id="historical-context" className="mt-6">
        Historical Context and Evolution
      </Title>
      <Text>
        Deep Learning emerged from the broader field of Machine Learning, revolutionizing artificial intelligence through its ability to learn hierarchical representations of data. The field has evolved from the Perceptron (1957) to modern architectures capable of solving complex tasks across various domains.
      </Text>
      
      <Timeline active={3} bulletSize={24} lineWidth={2}>
        <Timeline.Item bullet={<BookOpen size={12} />} title="1943-1957: Early Foundations">
          <Text size="sm">McCulloch-Pitts neuron (1943) and Perceptron (1957) established the foundational concepts of artificial neurons</Text>
        </Timeline.Item>
        <Timeline.Item bullet={<Cpu size={12} />} title="1960-1980: First AI Winter">
          <Text size="sm">Limited computing power and the XOR problem highlighted limitations of single-layer networks</Text>
        </Timeline.Item>
        <Timeline.Item bullet={<Network size={12} />} title="1986-2006: Backpropagation Era">
          <Text size="sm">Introduction of backpropagation algorithm and multi-layer networks</Text>
        </Timeline.Item>
        <Timeline.Item bullet={<Brain size={12} />} title="2012-Present: Deep Learning Revolution">
          <Text size="sm">AlexNet (2012) demonstrated the power of deep networks, leading to breakthroughs across various domains</Text>
        </Timeline.Item>
      </Timeline>

      <Title order={2} id="frameworks" className="mt-6">
        Deep Learning Frameworks
      </Title>
      <Text>
        Modern deep learning development relies heavily on specialized frameworks that provide efficient implementations of automatic differentiation and GPU acceleration.
      </Text>
      
      <Grid grow className="mt-4">
        <Grid.Col span={4}>
          <Box className="p-4 border rounded">
            <Title order={4}>PyTorch</Title>
            <List>
              <List.Item>Dynamic computational graphs</List.Item>
              <List.Item>Python-first approach</List.Item>
              <List.Item>TorchScript for production</List.Item>
            </List>
          </Box>
        </Grid.Col>
        <Grid.Col span={4}>
          <Box className="p-4 border rounded">
            <Title order={4}>TensorFlow</Title>
            <List>
              <List.Item>Static computational graphs</List.Item>
              <List.Item>Production-ready deployment</List.Item>
              <List.Item>TensorBoard visualization</List.Item>
            </List>
          </Box>
        </Grid.Col>
        <Grid.Col span={4}>
          <Box className="p-4 border rounded">
            <Title order={4}>JAX</Title>
            <List>
              <List.Item>Functional approach</List.Item>
              <List.Item>XLA compilation</List.Item>
              <List.Item>Advanced automatic differentiation</List.Item>
            </List>
          </Box>
        </Grid.Col>
      </Grid>


      <Title order={2} id="pytorch-fundamentals" className="mt-8">
        PyTorch Fundamentals
      </Title>
      
      <Title order={3} id="tensors" className="mt-6">
        Tensors: The Building Blocks
      </Title>
      <Text>
        Tensors are the fundamental data structure in PyTorch, representing multi-dimensional arrays with uniform data type. They support automatic differentiation and GPU acceleration.
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch

# Creating tensors
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # From list
y = torch.zeros(2, 3)                      # Zeros tensor
z = torch.randn(2, 3)                      # Random normal distribution
ones = torch.ones(2, 3)                    # Ones tensor
arange = torch.arange(0, 10, step=2)       # Range tensor

# Basic properties
print(f"Shape: {x.shape}")         # Size of each dimension
print(f"Dtype: {x.dtype}")         # Data type
print(f"Device: {x.device}")       # CPU/GPU location

# Moving to GPU (if available)
if torch.cuda.is_available():
    x_gpu = x.cuda()  # or x.to('cuda')
`}
      />

      <Title order={3} id="tensor-operations" className="mt-6">
        Essential Tensor Operations
      </Title>
      
      <Grid grow gutter="md">
        <Grid.Col span={6}>
            <Title order={4}>Arithmetic Operations</Title>
            <CodeBlock
              language="python"
              code={`
# Basic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

add = a + b  # or torch.add(a, b)
sub = a - b  # or torch.sub(a, b)
mul = a * b  # or torch.mul(a, b)
div = a / b  # or torch.div(a, b)

# Matrix operations
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 2)
matmul = torch.matmul(mat1, mat2)  # Matrix multiplication
`}
            />
        </Grid.Col>
        
        <Grid.Col span={6}>
            <Title order={4}>Reshaping & Indexing</Title>
            <CodeBlock
              language="python"
              code={`
# Reshaping operations
x = torch.randn(4, 4)
reshaped = x.view(16)      # Reshape to 1D
reshaped = x.view(-1, 8)   # Auto-infer first dimension
permuted = x.permute(1, 0) # Transpose dimensions

# Indexing and slicing
first_row = x[0]           # First row
slice_2d = x[1:3, 1:3]    # 2D slice
boolean_idx = x[x > 0]     # Boolean indexing
`}
            />
        </Grid.Col>

      <Grid.Col span={6}>
      <Title order={4}>AutoGrad computes partial derivatives</Title>
        <List>
          <List.Item>Each operation node maintains a reference to its inputs (<code>grad_fn</code>)</List.Item>
          <List.Item>Gradients are computed only for leaf nodes with <code>requires_grad=True</code></List.Item>
          <List.Item>The backward pass is triggered by calling <code>backward()</code> on a scalar output</List.Item>
          <List.Item>For vector outputs, you must specify a gradient vector in <code>backward(gradient)</code></List.Item>
        </List>
      <Text>
  For a function <InlineMath>{"f(x, y)"}</InlineMath>, AutoGrad computes partial derivatives <InlineMath>{"\\frac{\\partial f}{\\partial x}"}</InlineMath> and <InlineMath>{"\\frac{\\partial f}{\\partial y}"}</InlineMath> through the computational graph.
</Text>

      <CodeBlock
        language="python"
        code={`
# Example of gradient computation
def f(x, y):
    return x**2 * y + y**3

# Create tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Compute function
z = f(x, y)

# Compute gradients
z.backward()

# Access computed gradients
print(f"df/dx: {x.grad}")  # 2 * x * y = 12.0
print(f"df/dy: {y.grad}")  # x^2 + 3 * y^2 = 31.0
`}
      />
              </Grid.Col>
      </Grid>
      <Architecture/>
    </Stack>
    
  );
};

export default Introduction;