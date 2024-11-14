import React from 'react';
import { Title, Text, Stack, Grid, Timeline, Box, List, Table, Alert } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import { BookOpen, Cpu, Network, Brain, AlertCircle } from 'lucide-react';

const Introduction = () => {
  return (
    <Stack spacing="xl" className="w-full">
      <Title order={1} id="introduction">Introduction to Deep Learning</Title>
      
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
            <Title order={4}>PyTorch 2.5</Title>
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
        PyTorch 2.5 Fundamentals
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
          <Box className="p-4 border rounded">
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
          </Box>
        </Grid.Col>
        
        <Grid.Col span={6}>
          <Box className="p-4 border rounded">
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
          </Box>
        </Grid.Col>
      </Grid>

      <Title order={3} id="autograd" className="mt-6">
        Automatic Differentiation
      </Title>
      <Text>
        PyTorch's autograd system enables automatic computation of gradients for all operations on tensors.
      </Text>

      <CodeBlock
        language="python"
        code={`
# Creating tensors with gradients
x = torch.randn(2, 2, requires_grad=True)
y = torch.randn(2, 2, requires_grad=True)

# Forward pass
z = x * 2 + y ** 2

# Compute gradients
loss = z.mean()
loss.backward()

# Access gradients
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

# Zero gradients for next iteration
x.grad.zero_()
y.grad.zero_()
`}
      />

      <Alert 
        icon={<AlertCircle size={16} />} 
        title="Best Practices" 
        color="blue"
        className="mt-4"
      >
        <List>
          <List.Item>Always check tensor device location before operations</List.Item>
          <List.Item>Use in-place operations (methods with trailing underscore) carefully</List.Item>
          <List.Item>Remember to zero gradients before each backward pass in training loops</List.Item>
          <List.Item>Use torch.no_grad() for inference to save memory</List.Item>
        </List>
      </Alert>

      <Table className="mt-6">
        <thead>
          <tr>
            <th>Operation Category</th>
            <th>Common Methods</th>
            <th>Use Cases</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Creation</td>
            <td>zeros(), ones(), randn(), arange()</td>
            <td>Initializing tensors, creating masks</td>
          </tr>
          <tr>
            <td>Manipulation</td>
            <td>view(), reshape(), permute(), transpose()</td>
            <td>Changing tensor dimensions, preparing data</td>
          </tr>
          <tr>
            <td>Math Operations</td>
            <td>add(), mul(), matmul(), sum()</td>
            <td>Neural network computations</td>
          </tr>
          <tr>
            <td>Indexing</td>
            <td>index_select(), masked_select()</td>
            <td>Batch processing, attention mechanisms</td>
          </tr>
        </tbody>
      </Table>


    </Stack>
  );
};

export default Introduction;