import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const PytorchIntroduction = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
        {/* Introduction */}
        <div data-slide>
          <Title order={1} mb="xl">
            PyTorch Introduction
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-orange-50 to-red-50 mb-6">
            <Title order={2} className="mb-4">Why PyTorch?</Title>
            <Text size="lg" className="mb-4">
              PyTorch has become the dominant framework in deep learning research due to its 
              intuitive design, dynamic computation graphs, and seamless Python integration. 
              It provides the perfect balance between ease of use and performance.
            </Text>
            
            <Flex direction="column" align="center" className="mb-4">
              <Image
                src="/assets/python-deep-learning/module1/pytorch_ecosystem.png"
                alt="PyTorch Ecosystem"
                w={{ base: 400, sm: 600, md: 800 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              The PyTorch ecosystem for deep learning
            </Text>
          </Paper>
        </div>

        {/* Philosophy */}
        <div data-slide>
          <Title order={2} className="mb-6" id="philosophy">
            PyTorch Philosophy
          </Title>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Design Principles</Title>
            <Text size="lg" className="mb-4">
              PyTorch was designed with specific philosophical principles that make it 
              particularly well-suited for research and experimentation in deep learning.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Pythonic Design</Title>
                  <List>
                    <List.Item><strong>Native Python Feel:</strong> Follows Python conventions and idioms</List.Item>
                    <List.Item><strong>Easy Debugging:</strong> Standard Python debugging tools work seamlessly</List.Item>
                    <List.Item><strong>Interactive Development:</strong> Works naturally in Jupyter notebooks</List.Item>
                    <List.Item><strong>Familiar Syntax:</strong> Looks and feels like NumPy</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Pythonic tensor operations
import torch

# Creating tensors feels natural
x = torch.tensor([1, 2, 3, 4])
y = torch.zeros_like(x)

# Standard Python iteration works
for i, val in enumerate(x):
    y[i] = val * 2

print(y)  # tensor([2, 4, 6, 8])`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Dynamic Computation Graphs</Title>
                  <List>
                    <List.Item><strong>Define-by-Run:</strong> Graph built dynamically during execution</List.Item>
                    <List.Item><strong>Control Flow:</strong> Natural Python control structures</List.Item>
                    <List.Item><strong>Variable Length:</strong> Dynamic sequence lengths and structures</List.Item>
                    <List.Item><strong>Debugging Friendly:</strong> Inspect intermediate values easily</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Dynamic control flow in computation graph
def dynamic_network(x, condition):
    if condition:
        # Different computation path based on runtime condition
        return torch.relu(x @ W1 + b1)
    else:
        return torch.tanh(x @ W2 + b2)

# Graph structure determined at runtime
result = dynamic_network(input_data, some_condition)`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-6 bg-green-50 mb-6">
            <Title order={3} className="mb-4">Research-First Approach</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={4}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Eager Execution</Title>
                  <Text size="sm" className="mb-3">
                    Operations execute immediately, making development interactive and intuitive.
                  </Text>
                  <CodeBlock language="python" code={`# Operations execute immediately
x = torch.randn(3, 4)
print(x.mean())  # Computed right away

# No need to build graphs first
y = x.sum()
print(y.item())  # 2.1453...`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Flexible Experimentation</Title>
                  <Text size="sm" className="mb-3">
                    Easy to modify architectures, try new ideas, and prototype quickly.
                  </Text>
                  <CodeBlock language="python" code={`# Easy to experiment with architectures
for num_layers in [2, 4, 6]:
    model = create_model(num_layers)
    performance = train_and_evaluate(model)
    print(f"{num_layers} layers: {performance}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Production Ready</Title>
                  <Text size="sm" className="mb-3">
                    TorchScript enables deployment while maintaining development flexibility.
                  </Text>
                  <CodeBlock language="python" code={`# Convert to production format
@torch.jit.script
def optimized_model(x):
    return torch.relu(x @ weight + bias)

# Save for deployment
torch.jit.save(optimized_model, "model.pt")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-6 bg-yellow-50">
            <Title order={3} className="mb-4">Framework Comparison</Title>
            
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8f9fa' }}>
                    <th style={{ border: '1px solid #dee2e6', padding: '12px' }}>Aspect</th>
                    <th style={{ border: '1px solid #dee2e6', padding: '12px' }}>PyTorch</th>
                    <th style={{ border: '1px solid #dee2e6', padding: '12px' }}>TensorFlow 2.x</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Execution Model</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Dynamic (define-by-run)</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Eager + Graph mode</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Learning Curve</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Gentle, Pythonic</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Steeper, more concepts</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Debugging</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Standard Python tools</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>TensorBoard, specialized tools</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Research Adoption</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Dominant in research</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Strong in industry</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Paper>
        </div>

        {/* Tensors Foundation */}
        <div data-slide>
          <Title order={2} className="mb-6" id="tensors">
            Tensors: The Foundation
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mb-6">
            <Title order={3} className="mb-4">Understanding Tensors</Title>
            <Text size="lg" className="mb-4">
              Tensors are the fundamental data structure in PyTorch - multi-dimensional arrays 
              that can run on GPUs and support automatic differentiation. They are similar to 
              NumPy arrays but with additional capabilities for deep learning.
            </Text>
            
            <Flex direction="column" align="center" className="mb-4">
              <Image
                src="/assets/python-deep-learning/module1/tensor_dimensions.png"
                alt="Tensor Dimensions"
                w={{ base: 400, sm: 600, md: 700 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              Tensor dimensions: from scalars to multi-dimensional arrays
            </Text>
            
            <Grid gutter="lg" className="mt-6">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Tensor Properties</Title>
                  <List size="sm">
                    <List.Item><strong>dtype:</strong> Data type (float32, int64, bool, etc.)</List.Item>
                    <List.Item><strong>device:</strong> CPU or GPU location</List.Item>
                    <List.Item><strong>shape:</strong> Dimensions of the tensor</List.Item>
                    <List.Item><strong>requires_grad:</strong> Track gradients for autograd</List.Item>
                    <List.Item><strong>grad:</strong> Stores computed gradients</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Common Dimensions</Title>
                  <List size="sm">
                    <List.Item><strong>0D (Scalar):</strong> Single number - <InlineMath>{`\\mathbb{R}`}</InlineMath></List.Item>
                    <List.Item><strong>1D (Vector):</strong> Array of numbers - <InlineMath>{`\\mathbb{R}^n`}</InlineMath></List.Item>
                    <List.Item><strong>2D (Matrix):</strong> Table of numbers - <InlineMath>{`\\mathbb{R}^{m \\times n}`}</InlineMath></List.Item>
                    <List.Item><strong>3D:</strong> Cube (RGB images, time series)</List.Item>
                    <List.Item><strong>4D:</strong> Batch of images (B×C×H×W)</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Tensor Creation and Basic Operations</Title>
            
            <CodeBlock language="python" code={`import torch
import numpy as np

# ============ Tensor Creation ============
# From Python data structures
tensor_from_list = torch.tensor([1, 2, 3, 4])
tensor_from_nested = torch.tensor([[1, 2], [3, 4]])

# From NumPy arrays
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)

# Special constructors
zeros = torch.zeros(3, 4)           # All zeros
ones = torch.ones(2, 3)             # All ones
eye = torch.eye(3)                  # Identity matrix
arange = torch.arange(0, 10, 2)     # Range: [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # Evenly spaced: [0.0, 0.25, 0.5, 0.75, 1.0]

# Random tensors
uniform = torch.rand(3, 4)          # Uniform [0, 1)
normal = torch.randn(3, 4)          # Standard normal N(0,1)
randint = torch.randint(0, 10, (3, 4))  # Random integers [0, 10)

# ============ Tensor Attributes ============
x = torch.randn(2, 3, 4)
print(f"Shape: {x.shape}")         # torch.Size([2, 3, 4])
print(f"Data type: {x.dtype}")     # torch.float32
print(f"Device: {x.device}")       # cpu
print(f"Number of dimensions: {x.ndim}")  # 3
print(f"Total elements: {x.numel()}")     # 24

# ============ Basic Operations ============
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Element-wise operations
c = a + b                          # [5, 7, 9]
c = a * b                          # [4, 10, 18]
c = a ** 2                         # [1, 4, 9]

# Reductions
mean_val = a.float().mean()        # 2.0
sum_val = a.sum()                  # 6
max_val = a.max()                  # 3

print(f"Mean: {mean_val}, Sum: {sum_val}, Max: {max_val}")`} />
          </Paper>

          <Paper className="p-6 bg-teal-50">
            <Title order={3} className="mb-4">Tensor Memory and Storage</Title>
            <Text className="mb-4">
              Understanding PyTorch's tensor memory model is crucial for writing efficient code.
              PyTorch uses a sophisticated storage system that allows multiple tensors to share 
              the same underlying memory.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Storage and Strides</Title>
                  <Text size="sm" className="mb-3">
                    Tensors consist of data (storage) and metadata (strides, shape, offset). 
                    Strides determine how logical indices map to physical memory locations.
                  </Text>
                  
                  <CodeBlock language="python" code={`# Understanding tensor storage
x = torch.arange(6).view(2, 3)
print(f"Shape: {x.shape}")        # [2, 3]
print(f"Strides: {x.stride()}")   # (3, 1)

# Storage is shared between views
y = x.t()  # Transpose
print(f"Y shape: {y.shape}")      # [3, 2]
print(f"Y strides: {y.stride()}") # (1, 3)

# Both tensors share same storage
print(f"Same storage: {x.storage().data_ptr() == y.storage().data_ptr()}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Views vs Copies</Title>
                  <Text size="sm" className="mb-3">
                    Views provide different interpretations of the same memory, while copies 
                    create new storage. Understanding this distinction prevents memory issues.
                  </Text>
                  
                  <CodeBlock language="python" code={`x = torch.arange(12).view(3, 4)

# View operations (share storage)
y = x.view(4, 3)     # Reshape
z = x.transpose(0, 1) # Transpose
w = x[1:, :]         # Slice

# Clone creates a copy
x_copy = x.clone()

# Modifying original affects views, not copies
x[0, 0] = 999
print(f"View affected: {y[0, 0]}")    # 999
print(f"Copy unaffected: {x_copy[0, 0]}")  # 0`} />
                </Paper>
              </Grid.Col>
            </Grid>
            
            <Paper className="p-4 bg-blue-50 mt-4">
              <Title order={4} className="mb-3">Memory Efficiency</Title>
              <CodeBlock language="python" code={`# Memory-efficient operations
x = torch.randn(1000, 1000)

# In-place operations (memory efficient)
x.add_(1)        # Add 1 to all elements in-place
x.mul_(2)        # Multiply by 2 in-place
x.relu_()        # Apply ReLU in-place

# Avoid unnecessary copies
y = x[500:, 500:]  # Slice (view, not copy)

# Use contiguous() when necessary
if not y.is_contiguous():
    y = y.contiguous()  # Make memory layout contiguous

# Check memory usage
print(f"Memory usage: {x.element_size() * x.numel() / 1024**2:.2f} MB")`} />
            </Paper>
          </Paper>
        </div>

        {/* Device Management */}
        <div data-slide>
          <Title order={2} className="mb-6" id="device-management">
            Device Management and GPU Acceleration
          </Title>
          
          <Paper className="p-6 bg-indigo-50 mb-6">
            <Title order={3} className="mb-4">Working with GPUs</Title>
            <Text size="lg" className="mb-4">
              PyTorch provides seamless GPU acceleration for deep learning computations. 
              Understanding device management is crucial for performance optimization.
            </Text>
            
            <Flex direction="column" align="center" className="mb-4">
              <Image
                src="/assets/python-deep-learning/module1/gpu_acceleration.png"
                alt="GPU Acceleration"
                w={{ base: 400, sm: 600, md: 700 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              GPU acceleration dramatically speeds up tensor computations
            </Text>
            
            <CodeBlock language="python" code={`import torch

# ============ Device Detection ============
# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# ============ Device-Agnostic Code ============
# Best practice: automatic device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============ Moving Tensors to GPU ============
# Create tensor on CPU
x_cpu = torch.randn(1000, 1000)

# Move to GPU
x_gpu = x_cpu.to(device)
# or equivalently:
x_gpu = x_cpu.cuda() if cuda_available else x_cpu

# Create tensor directly on GPU
y_gpu = torch.randn(1000, 1000, device=device)

# ============ GPU Operations ============
if cuda_available:
    # Matrix multiplication on GPU
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    result = torch.matmul(x_gpu, y_gpu)
    end_time.record()
    
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = start_time.elapsed_time(end_time)
    print(f"GPU computation time: {gpu_time:.2f} ms")

# ============ Memory Management ============
if cuda_available:
    # Check GPU memory
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Clear cache when needed
    torch.cuda.empty_cache()`} />
            
            <Grid gutter="lg" className="mt-4">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Best Practices</Title>
                  <List size="sm">
                    <List.Item>Always use device-agnostic code</List.Item>
                    <List.Item>Move models and data to same device</List.Item>
                    <List.Item>Monitor GPU memory usage</List.Item>
                    <List.Item>Use torch.cuda.empty_cache() for memory cleanup</List.Item>
                    <List.Item>Batch operations for GPU efficiency</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Common Pitfalls</Title>
                  <List size="sm">
                    <List.Item>Tensors on different devices cause errors</List.Item>
                    <List.Item>Moving tensors unnecessarily hurts performance</List.Item>
                    <List.Item>Small tensors may be faster on CPU</List.Item>
                    <List.Item>GPU memory leaks from unreleased tensors</List.Item>
                    <List.Item>Forgetting to move models to GPU</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Autograd Introduction */}
        <div data-slide>
          <Title order={2} className="mb-6" id="autograd-intro">
            Introduction to Automatic Differentiation
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-emerald-50 to-teal-50 mb-6">
            <Title order={3} className="mb-4">The Magic Behind Deep Learning</Title>
            <Text size="lg" className="mb-4">
              Automatic differentiation (autograd) is what makes training neural networks practical. 
              PyTorch's autograd engine automatically computes gradients by tracking operations 
              and building a dynamic computational graph.
            </Text>
            
            <Flex direction="column" align="center" className="mb-4">
              <Image
                src="/assets/python-deep-learning/module1/computation_graph.png"
                alt="Computation Graph"
                w={{ base: 400, sm: 600, md: 800 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              Dynamic computation graph for automatic differentiation
            </Text>
            
            <Paper className="p-4 bg-white mt-4">
              <Text className="mb-3">
                <strong>Note:</strong> This is just an introduction to autograd. We'll cover automatic 
                differentiation in much more detail in the next section, including the mathematical 
                foundations, chain rule implementation, and advanced gradient computation techniques.
              </Text>
              
              <CodeBlock language="python" code={`# Simple autograd example
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass: compute function
z = x**2 + y**3
print(f"z = {z.item()}")  # z = 31.0

# Backward pass: compute gradients automatically
z.backward()

# Gradients computed via chain rule
print(f"dz/dx = {x.grad}")  # dz/dx = 4.0 (derivative of x^2 is 2x)
print(f"dz/dy = {y.grad}")  # dz/dy = 27.0 (derivative of y^3 is 3y^2)`} />
            </Paper>
          </Paper>

          <Paper className="p-6 bg-amber-50">
            <Title order={3} className="mb-4">Why Autograd Matters</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Before Autograd</Title>
                  <List size="sm">
                    <List.Item>Manual gradient computation</List.Item>
                    <List.Item>Error-prone derivative calculations</List.Item>
                    <List.Item>Limited to simple architectures</List.Item>
                    <List.Item>Difficult to experiment with new models</List.Item>
                    <List.Item>Time-consuming development</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">With Autograd</Title>
                  <List size="sm">
                    <List.Item>Automatic gradient computation</List.Item>
                    <List.Item>Correct derivatives guaranteed</List.Item>
                    <List.Item>Complex architectures possible</List.Item>
                    <List.Item>Easy to try new ideas</List.Item>
                    <List.Item>Focus on model design, not math</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
            
            <Text className="mt-4" style={{ fontStyle: 'italic' }}>
              Next up: We'll dive deep into how autograd works mathematically, explore the 
              computational graph mechanics, and learn advanced techniques for gradient computation 
              and debugging.
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default PytorchIntroduction;