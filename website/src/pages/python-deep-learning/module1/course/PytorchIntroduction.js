import React from 'react';
import { Container, Title, Text, Space, List, Flex, Image,Grid, Paper } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const PytorchIntroduction = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">PyTorch Fundamentals</Title>

      <Title order={2} mt="xl">1. PyTorch: Deep Learning Computing Context</Title>
      
      <Title order={3} mt="md">Why Deep Learning Needs Specialized Computing</Title>
      
      <Text>
        Machine learning and deep learning require immense computational power and memory. Even simple operations like matrix multiplication 
        involve millions of calculations. For example, multiplying two 1000×1000 matrices requires 1 billion multiply-add operations. 
        Additionally, training neural networks requires computing gradients for every parameter through backpropagation, effectively 
        doubling the computational requirements.
      </Text>
      
      <Space h="md" />
      
      <Text><strong>PyTorch</strong> addresses these challenges by providing:</Text>
      <List>
        <List.Item>Efficient tensor operations optimized for parallel hardware</List.Item>
        <List.Item>Automatic gradient computation (autograd)</List.Item>
        <List.Item>Seamless GPU acceleration</List.Item>
        <List.Item>Integration with highly optimized linear algebra libraries</List.Item>
      </List>
      
      <Text mt="md"><strong>Alternatives</strong>: TensorFlow, JAX</Text>
      
      <Title order={3} mt="xl">Understanding Performance</Title>
            <BlockMath>
        {`\\text{Performance} \\propto \\frac{\\text{Parallelism} \\times \\text{Memory Bandwidth}}{\\text{Data Transfer Overhead}}`}
      </BlockMath>
      <Text><strong>Parallelism</strong>: Executing multiple operations simultaneously</Text>
      <List>
        <List.Item><strong>Data Parallelism</strong>: Same operation on different data elements</List.Item>
        <List.Item><strong>Task Parallelism</strong>: Different operations executed concurrently</List.Item>
      </List>
      
      <Text mt="md"><strong>Bandwidth</strong>: Rate of data transfer between processor and memory</Text>
      <List>
        <List.Item>CPU-RAM: ~100 GB/s</List.Item>
        <List.Item>GPU-VRAM: ~1000 GB/s (10× faster)</List.Item>
        <List.Item>PCIe (CPU↔GPU): ~30 GB/s (bottleneck)</List.Item>
      </List>
      
      <Text mt="md"><strong>Data Transfer</strong>: Movement of data between different memory spaces</Text>
      <List>
        <List.Item>Minimize transfers between CPU and GPU</List.Item>
        <List.Item>Batch operations to amortize transfer costs</List.Item>
      </List>
      
      <Title order={3} mt="xl">Hardware Architecture</Title>
      
      <Text><strong>CPU (Central Processing Unit):</strong></Text>
      <List>
        <List.Item>4-64 powerful cores optimized for sequential processing</List.Item>
        <List.Item>Large cache (MB) for complex branching logic</List.Item>
        <List.Item>Optimized for latency (fast individual operations)</List.Item>
        <List.Item>Better for: Control flow, small tensors, irregular access patterns</List.Item>
      </List>
      
      <Text mt="md"><strong>GPU (Graphics Processing Unit):</strong></Text>
      <List>
        <List.Item>1000s-10000s of simple cores for parallel processing</List.Item>
        <List.Item>Small cache per core, optimized for throughput</List.Item>
        <List.Item>SIMD architecture (Single Instruction, Multiple Data)</List.Item>
        <List.Item>Better for: Large matrix operations, convolutions, element-wise ops</List.Item>
      </List>

      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/hardware-comparison.png"
          alt="CPU vs GPU Architecture"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          CPU vs GPU Architecture Comparison
        </Text>
      </Flex>
      <Title order={3} mt="xl">Linear Algebra</Title>
      
      <Text>
        Matrix operations naturally decompose into independent calculations, making them ideal for parallel processing:
      </Text>
      
      <BlockMath>
        {`C_{ij} = \\sum_{k=1}^{n} A_{ik} \\times B_{kj}`}
      </BlockMath>
      
      <Text>
        Each element <InlineMath>{`C_{ij}`}</InlineMath> can be computed independently. For a 1000×1000 matrix multiplication:
      </Text>
      
      <List>
        <List.Item><strong>Total operations</strong>: 1,000,000 dot products (one per output element)</List.Item>
        <List.Item><strong>Operations per dot product</strong>: 1,000 multiply-adds</List.Item>
        <List.Item><strong>Total FLOPs</strong>: 2 billion (2×10⁹) floating-point operations</List.Item>
        <List.Item><strong>Memory needed</strong>: ~12 MB for float32 (3 matrices × 1M elements × 4 bytes)</List.Item>
      </List>
      
      <Text mt="md">
        <strong>Parallelization potential:</strong> All 1,000,000 dot products can theoretically execute simultaneously:
      </Text>
      
      <List>
        <List.Item><strong>Sequential (1 core)</strong>: 2 billion operations in sequence</List.Item>
        <List.Item><strong>CPU (16 cores)</strong>: ~125 million operations per core</List.Item>
        <List.Item><strong>GPU (10,000 cores)</strong>: ~200,000 operations per core</List.Item>
      </List>
      
      <Text mt="md">
        <strong>Real-world execution time estimates:</strong>
      </Text>
      
      <List>
        <List.Item>
          <strong>Intel i9-13900K (24 cores, 5.8 GHz, ~2 TFLOPS)</strong>: 
          <InlineMath>{`\\frac{2 \\times 10^9 \\text{ FLOPs}}{2 \\times 10^{12} \\text{ FLOPS}} = 1 \\text{ ms}`}</InlineMath> (theoretical)
          → ~50 ms (actual with memory overhead)
        </List.Item>
        <List.Item>
          <strong>NVIDIA RTX 4090 (16,384 cores, 82.6 TFLOPS)</strong>: 
          <InlineMath>{`\\frac{2 \\times 10^9 \\text{ FLOPs}}{82.6 \\times 10^{12} \\text{ FLOPS}} = 0.024 \\text{ ms}`}</InlineMath> (theoretical)
          → ~0.5 ms (actual with memory overhead)
        </List.Item>
      </List>
      
      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/matrix-parallelization.png"
          alt="Matrix Multiplication Parallelization"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Visualization of parallel matrix multiplication decomposition
        </Text>
      </Flex>
      
      
      <Title order={3} mt="xl">Python vs Compiled Languages: The Performance Gap</Title>
      
      <Text>
        Python is an interpreted language with significant overhead for numerical computations:
      </Text>
      
      <List>
        <List.Item><strong>Python loop</strong>: ~1000× slower than C++ for numerical operations</List.Item>
        <List.Item><strong>Dynamic typing</strong>: Type checking at runtime adds overhead</List.Item>
        <List.Item><strong>GIL (Global Interpreter Lock)</strong>: Prevents true multi-threading</List.Item>
      </List>
      
      <Title order={3} mt="xl">Optimized Linear Algebra Libraries</Title>
      
      <Text>
        Linear algebra optimization remains an active research field. Modern libraries provide state-of-the-art implementations:
      </Text>
      
      <List>
        <List.Item>
          <strong>BLAS (Basic Linear Algebra Subprograms)</strong>: Standard API for basic operations
          <List withPadding>
            <List.Item>Level 1: Vector operations (O(n))</List.Item>
            <List.Item>Level 2: Matrix-vector operations (O(n²))</List.Item>
            <List.Item>Level 3: Matrix-matrix operations (O(n³))</List.Item>
          </List>
        </List.Item>
        
        <List.Item>
          <strong>LAPACK</strong>: Higher-level operations (eigenvalues, decompositions)
          <Text size="sm" c="dimmed">Reference: <a href="https://www.netlib.org/lapack/" target="_blank" rel="noopener noreferrer">netlib.org/lapack</a></Text>
        </List.Item>
        
        <List.Item>
          <strong>Intel MKL (Math Kernel Library)</strong>: CPU optimizations for Intel processors
          <Text size="sm" c="dimmed">Uses AVX-512 instructions, multi-threading, cache optimization</Text>
          <Text size="sm" c="dimmed">Reference: <a href="https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" target="_blank" rel="noopener noreferrer">Intel oneAPI MKL</a></Text>
        </List.Item>
        
        <List.Item>
          <strong>cuBLAS</strong>: NVIDIA GPU-accelerated BLAS
          <Text size="sm" c="dimmed">Optimized for different GPU architectures (Ampere, Hopper)</Text>
          <Text size="sm" c="dimmed">Reference: <a href="https://developer.nvidia.com/cublas" target="_blank" rel="noopener noreferrer">NVIDIA cuBLAS</a></Text>
        </List.Item>
        
        <List.Item>
          <strong>cuDNN</strong>: Deep learning primitives (convolutions, RNNs)
          <Text size="sm" c="dimmed">Reference: <a href="https://developer.nvidia.com/cudnn" target="_blank" rel="noopener noreferrer">NVIDIA cuDNN</a></Text>
        </List.Item>
      </List>
      
      <Text mt="md">
        <strong>Recent Research:</strong> Algorithms continue to improve. For example, matrix multiplication complexity 
        has been reduced from O(n³) to O(n^2.373) theoretically, though practical implementations still use optimized O(n³) algorithms.
      </Text>
      
      <Title order={2} mt="xl">2. Tensor Definition</Title>
      
      <Text>
        A tensor is a multi-dimensional array that generalizes scalars, vectors, and matrices to arbitrary dimensions.
      </Text>
      
      <Title order={3} mt="md">Core Attributes</Title>
      
      <List>
        <List.Item><strong>shape</strong>: Dimensions of the tensor (e.g., [3, 4] for a 3×4 matrix)</List.Item>
        <List.Item><strong>dtype</strong>: Data type of elements (float32, int64, etc.)</List.Item>
        <List.Item><strong>device</strong>: Computation location (CPU, CUDA GPU, MPS)</List.Item>
        <List.Item><strong>layout</strong>: Memory arrangement (strided or sparse)</List.Item>
        <List.Item><strong>storage</strong>: Underlying 1D memory buffer containing the data</List.Item>
        <List.Item><strong>strides</strong>: Number of elements to skip in storage for each dimension</List.Item>
      </List>
      
      <Title order={3} mt="md">Constructors</Title>
      
      <CodeBlock language="python" code={`import torch

# Basic tensor creation
x = torch.tensor([[1, 2], [3, 4]])  # From data
zeros = torch.zeros(3, 4)           # All zeros
ones = torch.ones(2, 5)             # All ones
rand = torch.rand(2, 3, 4)          # Uniform [0, 1)
randn = torch.randn(3, 3)           # Normal N(0, 1)
arange = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
eye = torch.eye(3)                  # 3×3 identity matrix

# Tensor attributes
print(f"Shape: {x.shape}")      # torch.Size([2, 2])
print(f"Device: {x.device}")    # cpu
print(f"Dtype: {x.dtype}")      # torch.int64
print(f"Strides: {x.stride()}") # (2, 1)
print(f"Layout: {x.layout}")    # torch.strided`} />
      
      <Title order={3} mt="md">Linear Algebra</Title>
      
      <Container fluid px={0}>
        <Grid gutter="lg">
          <Grid.Col span={6}>
            <Paper className="p-4 bg-blue-50">
              <Title order={4} mb="sm">Matrix Multiplication</Title>
              <BlockMath>{`C = AB \\text{ where } C_{ij} = \\sum_k A_{ik}B_{kj}`}</BlockMath>
              <Text size="sm" className="mb-2">Dimensions: <InlineMath>{`(m \\times n) \\cdot (n \\times p) = (m \\times p)`}</InlineMath></Text>
              
              <CodeBlock language="python" code={`# Matrix multiplication
A = torch.randn(10, 5)
B = torch.randn(5, 3)
C = torch.matmul(A, B)  # or A @ B
print(C.shape)  # torch.Size([10, 3])`} />
            </Paper>
          </Grid.Col>
          
          <Grid.Col span={6}>
            <Paper className="p-4 bg-green-50">
              <Title order={4} mb="sm">Element-wise Operations</Title>
              <BlockMath>{`C = A \\odot B \\text{ where } C_{ij} = A_{ij} \\cdot B_{ij}`}</BlockMath>
              <Text size="sm" className="mb-2">Hadamard product (element-wise multiplication)</Text>
              
              <CodeBlock language="python" code={`# Element-wise operations
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Element-wise multiplication
C = A * B  # [[5, 12], [21, 32]]

# Broadcasting example
v = torch.tensor([1, 2])
E = A + v  # [[2, 4], [4, 6]]`} />
            </Paper>
          </Grid.Col>
        </Grid>
      </Container>

      <Title order={2} mt="xl">3. Data Types</Title>
      
      <Text><strong>Float</strong>: float16 (2B), float32 (4B), float64 (8B), bfloat16</Text>
      <Text><strong>Int</strong>: int8, int32, int64, uint8, bool</Text>
      <Text><strong>Memory</strong>: Higher precision = more memory + slower</Text>
      <Text><strong>Casting</strong>: <InlineMath>.to(dtype)</InlineMath> explicit, automatic promotion in mixed ops</Text>
      <Text><strong>Priority</strong>: bool &lt; int8 &lt; int32 &lt; int64 &lt; float16 &lt; float32 &lt; float64</Text>
      
      <CodeBlock language="python" code={`# Data type operations
x = torch.tensor([1, 2, 3])  # int64 by default
y = x.to(torch.float32)      # Explicit casting
z = x * 1.5                   # Automatic promotion to float

# Memory usage comparison
float16_tensor = torch.rand(1000, 1000, dtype=torch.float16)  # 2MB
float32_tensor = torch.rand(1000, 1000, dtype=torch.float32)  # 4MB
float64_tensor = torch.rand(1000, 1000, dtype=torch.float64)  # 8MB`} />

      <Title order={2} mt="xl">4. Memory Layout</Title>
      
      <Text><strong>Strides</strong>: Step size between elements <InlineMath>tensor.stride()</InlineMath></Text>
      <Text><strong>Views</strong>: <InlineMath>reshape()</InlineMath>, <InlineMath>transpose()</InlineMath> - no copy, share storage</Text>
      <Text><strong>Storage</strong>: Underlying 1D memory buffer <InlineMath>tensor.storage()</InlineMath></Text>
      <Text><strong>Contiguous</strong>: Sequential memory layout, faster operations</Text>

      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/memory-layout.png"
          alt="Tensor Memory Layout"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Tensor Memory Layout and Strides
        </Text>
      </Flex>
      
      <CodeBlock language="python" code={`# Memory layout examples
x = torch.rand(3, 4)
print(f"Strides: {x.stride()}")  # (4, 1)

# View vs Copy
y = x.reshape(12)       # View - shares storage
z = x.transpose(0, 1)   # View - different strides
w = x.clone()           # Copy - new storage

# Contiguous check
print(f"x is contiguous: {x.is_contiguous()}")  # True
print(f"z is contiguous: {z.is_contiguous()}")  # False
z_cont = z.contiguous()  # Make contiguous copy`} />

      <Title order={2} mt="xl">5. Devices & Performance</Title>
      
      <Text><strong>Types</strong>: CPU, CUDA, MPS (Apple)</Text>
      <Text><strong>Transfer</strong>: <InlineMath>.to(device)</InlineMath>, <InlineMath>.cuda()</InlineMath>, <InlineMath>.cpu()</InlineMath></Text>
      <Text><strong>Performance</strong>: GPU &gt;&gt; CPU for parallel ops, CPU better for small tensors</Text>
      <Text><strong>Bottleneck</strong>: Host-device transfer ~10-30 GB/s</Text>

      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/performance-comparison.png"
          alt="Device Performance Comparison"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Performance Comparison: CPU vs GPU
        </Text>
      </Flex>
      
      <CodeBlock language="python" code={`# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create tensor on device
x = torch.rand(1000, 1000, device=device)

# Transfer between devices
cpu_tensor = torch.rand(100, 100)
gpu_tensor = cpu_tensor.cuda()  # CPU -> GPU
back_to_cpu = gpu_tensor.cpu()  # GPU -> CPU

# Performance comparison
import time

size = (10000, 10000)
cpu_a = torch.rand(size)
cpu_b = torch.rand(size)

gpu_a = cpu_a.cuda()
gpu_b = cpu_b.cuda()

# CPU multiplication
start = time.time()
cpu_c = cpu_a @ cpu_b
print(f"CPU time: {time.time() - start:.4f}s")

# GPU multiplication
start = time.time()
gpu_c = gpu_a @ gpu_b
torch.cuda.synchronize()  # Wait for completion
print(f"GPU time: {time.time() - start:.4f}s")`} />

      <Title order={2} mt="xl">6. Functions & NumPy</Title>
      
      <Text><strong>Math</strong>: <InlineMath>sin()</InlineMath>, <InlineMath>cos()</InlineMath>, <InlineMath>exp()</InlineMath>, <InlineMath>log()</InlineMath>, <InlineMath>sum()</InlineMath>, <InlineMath>mean()</InlineMath></Text>
      <Text><strong>NumPy</strong>: <InlineMath>torch.from\_numpy()</InlineMath>, <InlineMath>tensor.numpy()</InlineMath> (zero-copy)</Text>
      <Text><strong>Compatibility</strong>: Most np.func → torch.func, device-aware</Text>
      
      <CodeBlock language="python" code={`# Mathematical operations
x = torch.linspace(0, 2*torch.pi, 100)
y = torch.sin(x)
z = torch.exp(-x)

# Reduction operations
tensor = torch.rand(3, 4, 5)
sum_all = tensor.sum()
mean_dim = tensor.mean(dim=1)
max_val, max_idx = tensor.max(dim=2)

# NumPy interoperability
import numpy as np

# NumPy to PyTorch (zero-copy)
np_array = np.array([1, 2, 3, 4])
torch_tensor = torch.from_numpy(np_array)

# PyTorch to NumPy (zero-copy if on CPU)
torch_cpu = torch.rand(3, 4)
np_view = torch_cpu.numpy()

# Modifying one affects the other
np_view[0, 0] = 999
print(torch_cpu[0, 0])  # 999`} />
    </Container>
  );
};

export default PytorchIntroduction;