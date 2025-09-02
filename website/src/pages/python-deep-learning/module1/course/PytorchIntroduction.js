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
      
      <Title order={3} mt="md">Understanding Data Types in Memory</Title>
      
      <Text>
        A <strong>data type</strong> defines how binary data in memory is interpreted as numbers. Each type specifies:
      </Text>
      
      <List>
        <List.Item><strong>Size</strong>: Number of bits/bytes used</List.Item>
        <List.Item><strong>Interpretation</strong>: How bits represent values (integer vs floating-point)</List.Item>
        <List.Item><strong>Range</strong>: Minimum and maximum representable values</List.Item>
      </List>
      
      <Title order={4} mt="md">Integer Representation</Title>
      
      <Text>
        Integers use binary representation with optional sign bit. For example, <strong>int8</strong> uses 8 bits:
      </Text>
      
      <List>
        <List.Item><strong>Unsigned int8</strong>: 0 to 255 (2⁸ - 1)</List.Item>
        <List.Item><strong>Signed int8</strong>: -128 to 127 (using two's complement)</List.Item>
      </List>
      
      <Text mt="sm">
        <strong>Example:</strong> The number 5 in int8 is stored as <code>00000101</code>, while -5 is <code>11111011</code>
      </Text>
      
      <Title order={4} mt="md">Floating-Point Representation</Title>
      
      <Text>
        Floating-point numbers use scientific notation in binary: <InlineMath>{`\\text{value} = \\text{sign} \\times \\text{mantissa} \\times 2^{\\text{exponent}}`}</InlineMath>
      </Text>
      
      <Text mt="sm">
        For <strong>float32</strong> (single precision):
      </Text>
      
      <List>
        <List.Item>1 bit: Sign (0 = positive, 1 = negative)</List.Item>
        <List.Item>8 bits: Exponent (biased by 127)</List.Item>
        <List.Item>23 bits: Mantissa (fractional part)</List.Item>
      </List>
      
      <Text mt="sm">
        <strong>Example:</strong> The number 3.14 ≈ 1.57 × 2¹ is stored with mantissa ≈ 1.57 and exponent = 1
      </Text>
      
      <Title order={3} mt="xl">PyTorch Data Types</Title>
      
      <Text><strong>Floating-point types:</strong></Text>
      <List>
        <List.Item><strong>float16</strong>: Half precision (2 bytes) - Range: ±65,504</List.Item>
        <List.Item><strong>float32</strong>: Single precision (4 bytes) - Range: ±3.4×10³⁸</List.Item>
        <List.Item><strong>float64</strong>: Double precision (8 bytes) - Range: ±1.8×10³⁰⁸</List.Item>
        <List.Item><strong>bfloat16</strong>: Brain float (2 bytes) - Wider range, less precision</List.Item>
      </List>
      
      <Text mt="md"><strong>Integer types:</strong></Text>
      <List>
        <List.Item><strong>int8</strong>: 1 byte - Range: -128 to 127</List.Item>
        <List.Item><strong>int32</strong>: 4 bytes - Range: ±2.1×10⁹</List.Item>
        <List.Item><strong>int64</strong>: 8 bytes - Range: ±9.2×10¹⁸</List.Item>
        <List.Item><strong>uint8</strong>: Unsigned 1 byte - Range: 0 to 255</List.Item>
        <List.Item><strong>bool</strong>: 1 byte - Values: True/False</List.Item>
      </List>
      
      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/memory-usage.png"
          alt="Memory Usage Comparison"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Memory usage for different data types (1M element tensor)
        </Text>
      </Flex>
      
      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/time-operation.png"
          alt="Operation Time Comparison"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Computation time for matrix multiplication with different data types
        </Text>
      </Flex>
      
      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/precision-comparison.png"
          alt="Precision Comparison"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Precision vs range tradeoffs for different floating-point types
        </Text>
      </Flex>
      
      <Title order={3} mt="xl">Type Casting</Title>
      
      <Text>
        <strong>Type casting</strong> converts data from one type to another. This can be explicit (you specify) or implicit (automatic).
      </Text>
      
      <CodeBlock language="python" code={`# Explicit casting example
x = torch.tensor([1, 2, 3])          # Default: int64
y = x.to(torch.float32)              # Explicitly cast to float32
z = x.float()                         # Alternative syntax

# Check types
print(f"x dtype: {x.dtype}")         # torch.int64
print(f"y dtype: {y.dtype}")         # torch.float32

# Casting with precision loss
high_precision = torch.tensor([3.141592653589793], dtype=torch.float64)
low_precision = high_precision.to(torch.float16)
print(f"float64: {high_precision[0]:.15f}")  # 3.141592653589793
print(f"float16: {low_precision[0]:.15f}")   # 3.140625000000000 (precision lost)`} />
      
      <Title order={3} mt="xl">Type Priority and Automatic Promotion</Title>
      
      <Text>
        <strong>Type priority</strong> determines the result type when mixing different data types in operations. 
        PyTorch automatically promotes to the higher-priority type to prevent data loss.
      </Text>
      
      <Text mt="sm">
        <strong>Priority order:</strong> bool &lt; int8 &lt; int32 &lt; int64 &lt; float16 &lt; float32 &lt; float64
      </Text>
      
      <CodeBlock language="python" code={`# Automatic type promotion example
int_tensor = torch.tensor([1, 2, 3])           # int64
float_tensor = torch.tensor([1.5, 2.5, 3.5])  # float32

# Mixed operation - automatically promotes to float32
result = int_tensor + float_tensor
print(f"Result dtype: {result.dtype}")         # torch.float32
print(f"Result: {result}")                     # tensor([2.5, 4.5, 6.5])

# Priority demonstration
bool_t = torch.tensor([True, False])           # bool
int8_t = torch.tensor([1, 2], dtype=torch.int8)
float32_t = torch.tensor([1.0, 2.0])

# bool + int8 -> int8
r1 = bool_t + int8_t
print(f"bool + int8 = {r1.dtype}")            # torch.int8

# int8 + float32 -> float32
r2 = int8_t + float32_t  
print(f"int8 + float32 = {r2.dtype}")         # torch.float32`} />
      
      <Title order={3} mt="xl">Memory and Performance Implications</Title>
      
      <CodeBlock language="python" code={`# Memory usage comparison
size = (1000, 1000)  # 1M elements

# Different precisions
float16_tensor = torch.rand(size, dtype=torch.float16)  # 2MB
float32_tensor = torch.rand(size, dtype=torch.float32)  # 4MB
float64_tensor = torch.rand(size, dtype=torch.float64)  # 8MB

print(f"float16 memory: {float16_tensor.element_size() * float16_tensor.nelement() / 1e6:.1f} MB")
print(f"float32 memory: {float32_tensor.element_size() * float32_tensor.nelement() / 1e6:.1f} MB")
print(f"float64 memory: {float64_tensor.element_size() * float64_tensor.nelement() / 1e6:.1f} MB")

# Performance comparison (matrix multiplication)
import time

a16 = torch.rand(1000, 1000, dtype=torch.float16, device='cuda')
b16 = torch.rand(1000, 1000, dtype=torch.float16, device='cuda')

a32 = a16.to(torch.float32)
b32 = b16.to(torch.float32)

# float16 multiplication
start = time.time()
c16 = a16 @ b16
torch.cuda.synchronize()
print(f"float16 time: {(time.time() - start)*1000:.2f} ms")

# float32 multiplication  
start = time.time()
c32 = a32 @ b32
torch.cuda.synchronize()
print(f"float32 time: {(time.time() - start)*1000:.2f} ms")`} />

      <Title order={2} mt="xl">4. Memory Layout</Title>
      
      <Title order={3} mt="md">Understanding Tensor Storage: Logical vs Physical</Title>
      
      <Text>
        PyTorch tensors have two distinct aspects that are crucial to understand:
      </Text>
      
      <List>
        <List.Item>
          <strong>Logical Tensor</strong>: The multi-dimensional array you manipulate in code (e.g., a 3×4 matrix)
        </List.Item>
        <List.Item>
          <strong>Physical Storage</strong>: The actual 1D contiguous memory buffer where data is stored
        </List.Item>
      </List>
      
      <Text mt="md">
        The connection between these two is managed through <strong>metadata</strong> that describes how to interpret the 1D storage as a multi-dimensional tensor. This separation enables efficient operations without copying data.
      </Text>
      
      <Title order={3} mt="xl">Strides: The Bridge Between Logical and Physical</Title>
      
      <Text>
        <strong>Strides</strong> define how many elements to skip in storage when moving along each dimension. They map the logical n-dimensional indices to the physical 1D storage position.
      </Text>
      
      <Text mt="md">
        For a position <InlineMath>{`(i, j, k)`}</InlineMath> in a 3D tensor, the storage offset is:
      </Text>
      <Text ta="center">
        <InlineMath>{`\\text{offset} = i \\times \\text{stride}[0] + j \\times \\text{stride}[1] + k \\times \\text{stride}[2]`}</InlineMath>
      </Text>
      
      <Text mt="md">Let's see how a 3×4 matrix is stored in memory:</Text>
      
      <CodeBlock language="python" code={`x = torch.arange(12).reshape(3, 4)
print(x)  # [[ 0,  1,  2,  3],
          #  [ 4,  5,  6,  7],
          #  [ 8,  9, 10, 11]]`} />
      
      <Text mt="md">
        The underlying storage is always a flat 1D array:
      </Text>
      
      <CodeBlock language="python" code={`print(list(x.storage()))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print(x.stride())          # (4, 1)`} />
      
      <Text mt="md">
        Stride (4, 1) means: to move one row down, skip 4 elements; to move one column right, skip 1 element.
        For example, element x[1, 2] is at storage position: 1×4 + 2×1 = 6, which contains value 6.
      </Text>
      
      <Text mt="md">
        <strong>Transpose changes strides without copying data:</strong>
      </Text>
      
      <CodeBlock language="python" code={`y = x.t()  # Transpose
print(y.stride())  # (1, 4) - strides are swapped!
print(x.storage().data_ptr() == y.storage().data_ptr())  # True - same memory!`} />
      
      <Title order={3} mt="xl">Views: Efficient Tensor Manipulation</Title>
      
      <Text>
        A <strong>view</strong> is a tensor that shares the same underlying storage with another tensor but has different metadata (shape, strides, offset). Views enable efficient reshaping without copying data.
      </Text>
      
      <Text mt="md">
        <strong>Operations that create views (no data copy):</strong>
      </Text>
      
      <CodeBlock language="python" code={`x = torch.arange(12).reshape(3, 4)
view1 = x.reshape(2, 6)     # Reshape
view2 = x.transpose(0, 1)   # Transpose  
view3 = x[1:]               # Slicing
view4 = x.flatten()         # Flatten`} />
      
      <Text mt="md">
        All these views share the same memory. Modifying a view affects the original:
      </Text>
      
      <CodeBlock language="python" code={`view1[0, 0] = 100
print(x[0, 0])  # 100 - original is modified!`} />
      
      <Text mt="md">
        <strong>Operations that create copies (new storage):</strong>
      </Text>
      
      <CodeBlock language="python" code={`copy1 = x.clone()           # Explicit copy
copy2 = x.numpy().copy()    # NumPy copy
copy1[0, 0] = 999
print(x[0, 0])  # 100 - original unchanged`} />
      
      <Title order={3} mt="xl">Contiguous Memory Layout</Title>
      
      <Text>
        A tensor is <strong>contiguous</strong> when its elements are stored in the order you would expect when iterating through dimensions from last to first. Non-contiguous tensors can be less efficient for certain operations.
      </Text>
      
      <CodeBlock language="python" code={`x = torch.arange(12).reshape(3, 4)
print(x.is_contiguous())  # True
print(x.stride())         # (4, 1) - decreasing`} />
      
      <Text mt="md">
        Transpose creates a non-contiguous view (strides not in decreasing order):
      </Text>
      
      <CodeBlock language="python" code={`y = x.t()
print(y.is_contiguous())  # False  
print(y.stride())         # (1, 4) - not decreasing!`} />
      
      <Text mt="md">
        Some operations require contiguous tensors. Use <code>.contiguous()</code> to create a contiguous copy:
      </Text>
      
      <CodeBlock language="python" code={`# y.view(-1)  # Error! Non-contiguous
z = y.contiguous()   # Make contiguous copy
z.view(-1)          # Now works!`} />
      
      <Title order={3} mt="xl">Sparse Tensors: Efficient Storage for Mostly-Zero Data</Title>
      
      <Text>
        <strong>Sparse tensors</strong> store only non-zero values and their indices, dramatically reducing memory usage for tensors with many zeros (common in NLP, graphs, and scientific computing).
      </Text>
      
      <Text mt="md">
        Create a sparse tensor with only 3 non-zero values in a 1000×1000 matrix:
      </Text>
      
      <CodeBlock language="python" code={`dense = torch.zeros(1000, 1000)
dense[10, 20] = 5.0
dense[100, 200] = 3.0  
dense[500, 750] = 7.0`} />
      
      <Text mt="md">
        Convert to sparse COO (Coordinate) format - stores only indices and values:
      </Text>
      
      <CodeBlock language="python" code={`sparse = dense.to_sparse()
print(sparse._indices().shape)  # [2, 3] - positions
print(sparse._values())         # [5.0, 3.0, 7.0]`} />
      
      <Text mt="md">
        Memory savings are dramatic for sparse data:
      </Text>
      
      <CodeBlock language="python" code={`# Dense: 1M floats = 4 MB
# Sparse: 3 values + 6 indices = ~72 bytes
# Compression ratio: ~55,000x`} />
      
      <Text mt="md">
        Create sparse tensors directly:
      </Text>
      
      <CodeBlock language="python" code={`sparse = torch.sparse_coo_tensor(
    indices=[[0, 1], [2, 0]],  # positions
    values=[3.0, 4.0],          # values
    size=(3, 3)
)`} />
      
      <Title order={2} mt="xl">5. Devices & Performance</Title>
      
      <Title order={3} mt="md">Understanding Computing Devices</Title>
      
      <Text>
        <strong>CPU (Central Processing Unit)</strong>: General-purpose processor optimized for sequential tasks and complex logic. 
        Good for small tensors and operations requiring high precision or complex branching.
      </Text>
      
      <Text mt="sm">
        <strong>CUDA (NVIDIA GPUs)</strong>: Parallel computing platform for NVIDIA graphics cards. 
        Excels at matrix operations and parallel computations with thousands of cores processing data simultaneously.
      </Text>
      
      <Text mt="sm">
        <strong>MPS (Metal Performance Shaders)</strong>: Apple's GPU acceleration framework for M1/M2 chips. 
        Provides GPU acceleration on Apple Silicon devices.
      </Text>
      
      <Title order={3} mt="md">Checking Available Devices</Title>
      
      <CodeBlock language="python" code={`# Check available devices
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Get device automatically
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
print(f"Using device: {device}")`} />
      
      <Title order={3} mt="md">Device Transfers</Title>
      
      <Text>
        The <code>.to(device)</code> method creates a copy of the tensor on the specified device. 
        This involves physically copying data between CPU RAM and GPU VRAM, which can be expensive.
      </Text>
      
      <CodeBlock language="python" code={`# Create tensor on CPU
cpu_tensor = torch.rand(1000, 1000)
print(f"Tensor device: {cpu_tensor.device}")  # cpu

# Transfer to GPU (creates a copy in GPU memory)
gpu_tensor = cpu_tensor.to('cuda')  
print(f"GPU tensor device: {gpu_tensor.device}")  # cuda:0

# Alternative syntax
gpu_tensor = cpu_tensor.cuda()  # Shorthand for .to('cuda')
cpu_back = gpu_tensor.cpu()     # Shorthand for .to('cpu')`} />
      
      <Text mt="md">
        <strong>Important:</strong> Device transfers are not in-place operations. They return new tensors:
      </Text>
      
      <CodeBlock language="python" code={`x = torch.rand(100, 100)
y = x.to('cuda')  # x stays on CPU, y is new tensor on GPU
print(x.device)   # cpu
print(y.device)   # cuda:0`} />
      
      <Title order={3} mt="md">GPU Synchronization</Title>
      
      <Text>
        GPU operations are <strong>asynchronous</strong> - they return immediately while computation happens in the background.
        <code>torch.cuda.synchronize()</code> blocks until all GPU operations complete.
      </Text>
      
      <CodeBlock language="python" code={`import time

# Without synchronization - misleading timing
start = time.time()
gpu_result = gpu_tensor @ gpu_tensor  # Returns immediately
print(f"Time: {time.time() - start:.6f}s")  # Too fast! Operation still running

# With synchronization - accurate timing
start = time.time()
gpu_result = gpu_tensor @ gpu_tensor
torch.cuda.synchronize()  # Wait for GPU to finish
print(f"Actual time: {time.time() - start:.6f}s")`} />
      
      <Title order={3} mt="md">Performance Comparison</Title>
      
      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/performance-comparison.png"
          alt="Device Performance Comparison"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Performance Comparison: CPU vs GPU for Matrix Operations
        </Text>
      </Flex>
      
      <CodeBlock language="python" code={`# Complete performance comparison
import time

size = (5000, 5000)

# Create matrices on CPU
cpu_a = torch.rand(size)
cpu_b = torch.rand(size)

# Transfer to GPU (this has a cost!)
transfer_start = time.time()
gpu_a = cpu_a.cuda()
gpu_b = cpu_b.cuda()
torch.cuda.synchronize()
print(f"Transfer time: {time.time() - transfer_start:.4f}s")

# CPU multiplication
start = time.time()
cpu_c = cpu_a @ cpu_b
cpu_time = time.time() - start

# GPU multiplication
start = time.time()
gpu_c = gpu_a @ gpu_b
torch.cuda.synchronize()  # Essential for accurate timing
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f}s")
print(f"GPU time: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")`} />
      
      <Text mt="md">
        <strong>Key Points:</strong> GPUs excel at large parallel operations but have transfer overhead. 
        Use GPU for large matrix operations, training neural networks. Keep small operations on CPU to avoid transfer costs.
      </Text>

      <Title order={2} mt="xl">6. Functions & NumPy</Title>
      
      <Title order={3} mt="md">Mathematical Functions</Title>
      
      <Text>
        PyTorch provides a comprehensive set of mathematical functions that operate element-wise on tensors. 
        These functions are optimized for both CPU and GPU computation.
      </Text>
      
      <Title order={4} mt="md">Trigonometric Functions</Title>
      
      <Text>
        Trigonometric functions operate element-wise, computing the function for each element independently:
      </Text>
      
      <CodeBlock language="python" code={`# Create angles from 0 to 2π
angles = torch.linspace(0, 2*torch.pi, 5)
print(angles)  # [0.0000, 1.5708, 3.1416, 4.7124, 6.2832]`} />
      
      <Text mt="sm">
        Apply sine function: <InlineMath>{`\\sin(x)`}</InlineMath> maps each angle to its sine value:
      </Text>
      
      <CodeBlock language="python" code={`sines = torch.sin(angles)
print(sines)   # [0.0000, 1.0000, 0.0000, -1.0000, 0.0000]`} />
      
      <Text mt="sm">
        Other trigonometric functions work similarly:
      </Text>
      
      <CodeBlock language="python" code={`cosines = torch.cos(angles)  # Cosine values
tangents = torch.tan(angles)  # Tangent values
arcsin = torch.asin(sines)    # Inverse sine (returns angles)`} />
      
      <Title order={4} mt="md">Exponential and Logarithmic Functions</Title>
      
      <Text>
        The exponential function <InlineMath>{`e^x`}</InlineMath> and logarithm <InlineMath>{`\\ln(x)`}</InlineMath> are fundamental in deep learning:
      </Text>
      
      <CodeBlock language="python" code={`x = torch.tensor([1.0, 2.0, 3.0])
exp_x = torch.exp(x)  # e^x for each element
print(exp_x)  # [2.7183, 7.3891, 20.0855]`} />
      
      <Text mt="sm">
        Natural logarithm is the inverse of exponential:
      </Text>
      
      <CodeBlock language="python" code={`log_exp_x = torch.log(exp_x)  # ln(e^x) = x
print(log_exp_x)  # [1.0, 2.0, 3.0] - recovers original`} />
      
      <Text mt="sm">
        For numerical stability in deep learning, use special functions:
      </Text>
      
      <CodeBlock language="python" code={`# log(1 + e^x) computed stably for large x
x = torch.tensor([100.0, -100.0, 0.0])
stable = torch.log1p(torch.exp(x))  # More stable than log(1 + exp(x))`} />
      
      <Title order={4} mt="md">Power and Root Functions</Title>
      
      <CodeBlock language="python" code={`x = torch.tensor([1.0, 4.0, 9.0, 16.0])
sqrt_x = torch.sqrt(x)     # Square root
print(sqrt_x)  # [1.0, 2.0, 3.0, 4.0]`} />
      
      <Text mt="sm">
        General power function: <InlineMath>{`x^n`}</InlineMath>
      </Text>
      
      <CodeBlock language="python" code={`squared = torch.pow(x, 2)   # x^2
cubed = torch.pow(x, 3)     # x^3
reciprocal = torch.pow(x, -1)  # 1/x`} />
      
      <Title order={3} mt="xl">Reduction Operations</Title>
      
      <Text>
        Reduction operations aggregate tensor values along specified dimensions, reducing the tensor's dimensionality.
      </Text>
      
      <Title order={4} mt="md">Sum and Mean</Title>
      
      <Text>
        Sum aggregates values: <InlineMath>{`\\sum_{i} x_i`}</InlineMath>
      </Text>
      
      <CodeBlock language="python" code={`x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# Sum all elements
total = x.sum()
print(total)  # 21`} />
      
      <Text mt="sm">
        Reduce along specific dimensions:
      </Text>
      
      <CodeBlock language="python" code={`# Sum along rows (dim=0)
row_sums = x.sum(dim=0)
print(row_sums)  # [5, 7, 9]  # [1+4, 2+5, 3+6]`} />
      
      <CodeBlock language="python" code={`# Sum along columns (dim=1)
col_sums = x.sum(dim=1)
print(col_sums)  # [6, 15]  # [1+2+3, 4+5+6]`} />
      
      <Text mt="sm">
        Mean computes average: <InlineMath>{`\\frac{1}{n}\\sum_{i} x_i`}</InlineMath>
      </Text>
      
      <CodeBlock language="python" code={`mean_all = x.mean()           # 3.5 (21/6)
mean_rows = x.mean(dim=0)     # [2.5, 3.5, 4.5]
mean_cols = x.mean(dim=1)     # [2.0, 5.0]`} />
      
      <Title order={4} mt="md">Min, Max, and ArgMax</Title>
      
      <CodeBlock language="python" code={`x = torch.tensor([[3, 7, 2],
                  [9, 1, 5]])
# Maximum value
max_val = x.max()
print(max_val)  # 9`} />
      
      <Text mt="sm">
        Get both values and indices along a dimension:
      </Text>
      
      <CodeBlock language="python" code={`# Max along columns (dim=1)
values, indices = x.max(dim=1)
print(values)   # [7, 9] - max values
print(indices)  # [1, 0] - positions of max values`} />
      
      <CodeBlock language="python" code={`# ArgMax returns only indices
argmax = x.argmax(dim=1)
print(argmax)  # [1, 0] - same as indices above`} />
      
      <Title order={4} mt="md">Standard Deviation and Variance</Title>
      
      <Text>
        Variance measures spread: <InlineMath>{`\\sigma^2 = \\frac{1}{n}\\sum_{i}(x_i - \\mu)^2`}</InlineMath>
      </Text>
      
      <CodeBlock language="python" code={`x = torch.randn(100)  # 100 samples from N(0,1)
mean = x.mean()        # Should be close to 0
std = x.std()          # Should be close to 1
var = x.var()          # Should be close to 1`} />
      
      <Text mt="sm">
        Compute along specific dimensions:
      </Text>
      
      <CodeBlock language="python" code={`batch = torch.randn(32, 10, 20)  # Batch of 32 samples
# Compute stats across batch dimension
batch_mean = batch.mean(dim=0)  # Shape: [10, 20]
batch_std = batch.std(dim=0)    # Shape: [10, 20]`} />
      
      <Title order={3} mt="xl">Element-wise Operations</Title>
      
      <Title order={4} mt="md">Arithmetic Operations</Title>
      
      <Text>
        Basic arithmetic operations work element-wise between tensors of the same shape:
      </Text>
      
      <CodeBlock language="python" code={`a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])`} />
      
      <CodeBlock language="python" code={`# Addition: a + b
add = torch.add(a, b)  # [5.0, 7.0, 9.0]`} />
      
      <CodeBlock language="python" code={`# Multiplication: a * b (element-wise, not matrix!)
mul = torch.mul(a, b)  # [4.0, 10.0, 18.0]`} />
      
      <CodeBlock language="python" code={`# Division: a / b
div = torch.div(a, b)  # [0.25, 0.40, 0.50]`} />
      
      <Title order={4} mt="md">Comparison Operations</Title>
      
      <Text>
        Comparisons return boolean tensors:
      </Text>
      
      <CodeBlock language="python" code={`x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([5, 4, 3, 2, 1])`} />
      
      <CodeBlock language="python" code={`# Element-wise comparison
greater = x > 3        # [False, False, False, True, True]
equal = x == y         # [False, False, True, False, False]
less_equal = x <= y    # [True, True, True, False, False]`} />
      
      <Title order={4} mt="md">Activation Functions</Title>
      
      <Text>
        Common activation functions used in neural networks:
      </Text>
      
      <Text mt="sm">
        ReLU (Rectified Linear Unit): <InlineMath>{`\\text{ReLU}(x) = \\max(0, x)`}</InlineMath>
      </Text>
      
      <CodeBlock language="python" code={`x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)
relu = torch.relu(x)
print(relu)  # [0, 0, 0, 1, 2]`} />
      
      <Text mt="sm">
        Sigmoid (Logistic): <InlineMath>{`\\sigma(x) = \\frac{1}{1 + e^{-x}}`}</InlineMath>
      </Text>
      
      <CodeBlock language="python" code={`sigmoid = torch.sigmoid(x)
print(sigmoid)  # [0.119, 0.269, 0.500, 0.731, 0.881]`} />
      
      <Text mt="sm">
        Tanh (Hyperbolic Tangent): <InlineMath>{`\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}`}</InlineMath>
      </Text>
      
      <CodeBlock language="python" code={`tanh = torch.tanh(x)
print(tanh)  # [-0.964, -0.762, 0.000, 0.762, 0.964]`} />
      
      <Title order={3} mt="xl">Broadcasting</Title>
      
      <Text>
        Broadcasting allows operations between tensors of different shapes by automatically expanding smaller tensors.
      </Text>
      
      <Text mt="sm">
        <strong>Broadcasting Rules:</strong>
      </Text>
      <List>
        <List.Item>Compare shapes element-wise from right to left</List.Item>
        <List.Item>Dimensions are compatible if they are equal or one is 1</List.Item>
        <List.Item>Missing dimensions are treated as 1</List.Item>
      </List>
      
      <Text mt="md">
        Example: Adding a scalar to a matrix
      </Text>
      
      <CodeBlock language="python" code={`matrix = torch.tensor([[1, 2], 
                       [3, 4]])
scalar = 10
result = matrix + scalar  # Scalar broadcasts to [2, 2]
print(result)  # [[11, 12], [13, 14]]`} />
      
      <Text mt="sm">
        Example: Adding a vector to each row of a matrix
      </Text>
      
      <CodeBlock language="python" code={`matrix = torch.ones(3, 4)  # Shape: [3, 4]
vector = torch.tensor([1, 2, 3, 4])  # Shape: [4]
# Vector broadcasts from [4] to [1, 4] to [3, 4]
result = matrix + vector
print(result.shape)  # [3, 4]`} />
      
      <Text mt="sm">
        Example: Outer product via broadcasting
      </Text>
      
      <CodeBlock language="python" code={`a = torch.tensor([[1], [2], [3]])  # Shape: [3, 1]
b = torch.tensor([4, 5, 6])        # Shape: [3]
# Broadcasting: [3, 1] * [3] → [3, 1] * [1, 3] → [3, 3]
outer = a * b
print(outer)  # [[4, 5, 6], [8, 10, 12], [12, 15, 18]]`} />
      
      <Title order={3} mt="xl">NumPy Interoperability</Title>
      
      <Text>
        PyTorch tensors can seamlessly convert to/from NumPy arrays, often without copying data.
      </Text>
      
      <Title order={4} mt="md">NumPy to PyTorch</Title>
      
      <CodeBlock language="python" code={`import numpy as np

# Create NumPy array
np_array = np.array([1, 2, 3, 4, 5])
print(f"NumPy dtype: {np_array.dtype}")  # int64`} />
      
      <Text mt="sm">
        Convert to PyTorch tensor (shares memory on CPU):
      </Text>
      
      <CodeBlock language="python" code={`# Zero-copy conversion
torch_tensor = torch.from_numpy(np_array)
print(f"PyTorch dtype: {torch_tensor.dtype}")  # torch.int64`} />
      
      <Text mt="sm">
        Modifying one affects the other (they share memory):
      </Text>
      
      <CodeBlock language="python" code={`np_array[0] = 100
print(torch_tensor[0])  # 100 - changed!`} />
      
      <Title order={4} mt="md">PyTorch to NumPy</Title>
      
      <CodeBlock language="python" code={`# Create PyTorch tensor on CPU
cpu_tensor = torch.rand(2, 3)
# Zero-copy conversion to NumPy
np_view = cpu_tensor.numpy()
print(type(np_view))  # <class 'numpy.ndarray'>`} />
      
      <Text mt="sm">
        GPU tensors must be moved to CPU first:
      </Text>
      
      <CodeBlock language="python" code={`if torch.cuda.is_available():
    gpu_tensor = torch.rand(2, 3, device='cuda')
    # gpu_tensor.numpy()  # Error! Can't convert GPU tensor
    np_array = gpu_tensor.cpu().numpy()  # Move to CPU first`} />
      
      <Title order={4} mt="md">Deep Copy vs Shallow Copy</Title>
      
      <Text>
        To avoid shared memory, create explicit copies:
      </Text>
      
      <CodeBlock language="python" code={`# Deep copy from NumPy
np_array = np.array([1, 2, 3])
torch_copy = torch.tensor(np_array)  # New memory allocated
np_array[0] = 999
print(torch_copy[0])  # 1 - unchanged!`} />
      
      <CodeBlock language="python" code={`# Deep copy from PyTorch
torch_tensor = torch.rand(2, 3)
np_copy = torch_tensor.numpy().copy()  # Explicit copy
torch_tensor[0, 0] = 999
print(np_copy[0, 0])  # Original value - unchanged!`} />
      
      <Title order={3} mt="xl">Function Compatibility</Title>
      
      <Text>
        Most NumPy functions have PyTorch equivalents with similar names and behavior:
      </Text>
      
      <CodeBlock language="python" code={`# NumPy
np_arr = np.array([1, 2, 3, 4])
np_mean = np.mean(np_arr)
np_std = np.std(np_arr)
np_max = np.max(np_arr)`} />
      
      <CodeBlock language="python" code={`# PyTorch equivalents
torch_tensor = torch.tensor([1, 2, 3, 4])
torch_mean = torch.mean(torch_tensor.float())
torch_std = torch.std(torch_tensor.float())
torch_max = torch.max(torch_tensor)`} />
      
      <Text mt="md">
        <strong>Key differences from NumPy:</strong>
      </Text>
      
      <List>
        <List.Item>PyTorch is device-aware (CPU/GPU)</List.Item>
        <List.Item>PyTorch tracks gradients for automatic differentiation</List.Item>
        <List.Item>Some functions have different default behaviors (e.g., random seed)</List.Item>
        <List.Item>PyTorch uses float32 by default, NumPy uses float64</List.Item>
      </List>
      
      <CodeBlock language="python" code={`# Example: Different default dtypes
np_ones = np.ones(3)        # float64 by default
torch_ones = torch.ones(3)  # float32 by default
print(f"NumPy: {np_ones.dtype}")    # float64
print(f"PyTorch: {torch_ones.dtype}")  # torch.float32`} />
    </Container>
  );
};

export default PytorchIntroduction;