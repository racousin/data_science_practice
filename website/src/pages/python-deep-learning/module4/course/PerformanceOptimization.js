import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Badge, List, Grid, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';

const PerformanceOptimization = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <Title order={1}>Performance Optimization Techniques</Title>
        
        <Text>
          Modern deep learning frameworks provide numerous optimization techniques to accelerate training 
          and inference. We'll explore key strategies to maximize performance on available hardware.
        </Text>

        
          <Title id="torch-compile" order={2} mt="xl">Torch Compile</Title>
          
          <Text>
            PyTorch 2.0 introduced torch.compile, which optimizes models through graph compilation
            and fusion of operations, significantly improving performance.
          </Text>

            <Title order={4}>How Torch Compile Works</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Graph Capture:</strong> Traces Python execution to build computation graph</List.Item>
              <List.Item><strong>Graph Optimization:</strong> Fuses operations, eliminates redundancy</List.Item>
              <List.Item><strong>Code Generation:</strong> Generates optimized kernels for target hardware</List.Item>
              <List.Item><strong>Caching:</strong> Reuses compiled graphs for repeated calls</List.Item>
            </List>


          <CodeBlock language="python" code={`def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10))) #same output`} />
Arbitrary Python functions can be optimized by passing the callable to torch.compile. We can then call the returned optimized function in place of the original function.


          <CodeBlock language="python" code={`def simple_function(x, y):
    # Multiple element-wise operations
    z = torch.sin(x) + torch.cos(y)
    z = z * torch.exp(-z)
    z = torch.tanh(z) + torch.sigmoid(z)
    z = z ** 2 + torch.sqrt(torch.abs(z) + 1e-8)
    return z
    
        # Create compiled version
    compiled_function = torch.compile(simple_function)
    
    # Test tensors - make them large enough to see benefits
    size = (1000, 1000)
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
        print("Benchmarking original function...")
    start = time.perf_counter()
    for _ in range(100):
        result = simple_function(x, y)
        torch.cuda.synchronize()
    original_time = time.perf_counter() - start
    
    # Benchmark compiled function
    print("Benchmarking compiled function...")
    start = time.perf_counter()
    for _ in range(100):
        result = compiled_function(x, y)
        torch.cuda.synchronize()
    compiled_time = time.perf_counter() - start`} />
<CodeBlock code={`Function Results:
Original function: 0.019s
Compiled function: 0.007s
Speedup: 2.89x`} />

Behavior of torch.compile with Nested Modules and Function Calls

When you use torch.compile, the compiler will try to recursively compile every function call inside the target function or module inside the target function or module that is not in a skip list (such as built-ins, some functions in the torch.* namespace).

Best Practices:

1. Top-Level Compilation: One approach is to compile at the highest level possible (i.e., when the top-level module is initialized/called) and selectively disable compilation when encountering excessive graph breaks or errors. If there are still many compile issues, compile individual subcomponents instead.

2. Modular Testing: Test individual functions and modules with torch.compile before integrating them into larger models to isolate potential issues.


            <List size="sm">
              <List.Item>First call will be slower due to compilation overhead</List.Item>
              <List.Item>Best for static input shapes - dynamic shapes may cause recompilation</List.Item>
            </List>

        

        
          <Title id="mixed-precision" order={2} mt="xl">Mixed Precision Training</Title>
          
          <Text>
            While we can simply cast models and data to lower precision to reduce memory usage,
            this naive approach can impact training stability. Mixed precision training provides
            a sophisticated solution that combines the benefits of both precisions.
          </Text>

          <Title order={3} mt="lg">Basic Precision Casting</Title>
          
          <Text>
            You can manually cast models and tensors to 16-bit precision:
          </Text>

          <CodeBlock language="python" code={`# Simple casting to FP16
model = model.to(torch.float16)  # or model.half()
data = data.to(torch.float16)

# This halves memory usage but may cause:
# - Gradient underflow (gradients become zero)
# - Loss of precision in weight updates
# - Training instability`} />

          <Title order={3} mt="lg">How Mixed Precision Works</Title>
          
          <Text>
            Standard training uses 32-bit floating point (FP32) for all computations.
            Mixed precision training strategically combines:
          </Text>
          
          <List>
            <List.Item><strong>FP16:</strong> For most forward and backward pass computations</List.Item>
            <List.Item><strong>FP32:</strong> For operations needing higher precision (loss scaling, weight updates)</List.Item>
          </List>

          <Title order={4} mt="md">What Gets Cast to FP16 vs FP32?</Title>
          

            <Text weight={500} mb="sm">During mixed precision training:</Text>
            
            <Title order={5}>FP16 (Half Precision):</Title>
            <List size="sm">
              <List.Item><strong>Activations:</strong> Intermediate layer outputs during forward pass</List.Item>
              <List.Item><strong>Gradients:</strong> Computed gradients during backward pass</List.Item>
              <List.Item><strong>Forward computations:</strong> Matrix multiplications, convolutions</List.Item>
            </List>
            
            <Title order={5} mt="md">FP32 (Full Precision):</Title>
            <List size="sm">
              <List.Item><strong>Master weights:</strong> The actual model parameters always stay in FP32</List.Item>
              <List.Item><strong>Loss values:</strong> Final loss computation remains in FP32 for stability</List.Item>
              <List.Item><strong>Optimizer states:</strong> Adam moments, SGD momentum buffers</List.Item>
              <List.Item><strong>Weight updates:</strong> Gradient application to parameters</List.Item>
            </List>

          <CodeBlock language="python" code={`# What happens under the hood:
with autocast():
    # Input activations cast to FP16
    x_fp16 = x.to(torch.float16)
    
    # Weights temporarily cast to FP16 for computation
    w_fp16 = model.weight.to(torch.float16)  
    
    # Forward pass in FP16
    output = torch.matmul(x_fp16, w_fp16)  # FP16 computation
    
    # Loss computed in FP32 for stability
    loss = criterion(output.float(), target)  # Cast back to FP32

# Master weights remain in FP32 throughout
print(model.weight.dtype)  # torch.float32
print(optimizer.param_groups[0]['params'][0].dtype)  # torch.float32`} />

          <Title order={3} mt="lg">Key Benefits</Title>
          
          <Paper p="md" withBorder>
            <Grid>
              <Grid.Col span={4}>
                <Badge color="green" size="lg">Memory Efficiency</Badge>
                <Text size="sm" mt="xs">FP16 uses half the memory, enabling larger models or batch sizes</Text>
              </Grid.Col>
              <Grid.Col span={4}>
                <Badge color="blue" size="lg">Speed Improvements</Badge>
                <Text size="sm" mt="xs">Modern GPUs (V100, A100) have Tensor Cores for faster FP16 operations</Text>
              </Grid.Col>
              <Grid.Col span={4}>
                <Badge color="purple" size="lg">Maintained Accuracy</Badge>
                <Text size="sm" mt="xs">Preserves model quality through gradient scaling and FP32 master weights</Text>
              </Grid.Col>
            </Grid>
          </Paper>

          <Title order={3} mt="lg">Implementation Details</Title>

          <Text>The technique involves three key components:</Text>

          <Title order={4} mt="md">1. Automatic Loss Scaling</Title>
          <Text>Prevents gradient underflow by scaling the loss before backpropagation:</Text>
          
          <CodeBlock language="python" code={`from torch.cuda.amp import GradScaler

# Initialize gradient scaler
scaler = GradScaler()

# Scale loss before backward pass
scaled_loss = scaler.scale(loss)
scaled_loss.backward()

# Unscale gradients before optimizer step
scaler.unscale_(optimizer)`} />

          <Title order={4} mt="md">2. Smart Precision Casting</Title>
          <Text>Autocast automatically chooses the right precision for each operation:</Text>
          
          <CodeBlock language="python" code={`from torch.cuda.amp import autocast

# Operations inside autocast use FP16 where safe
with autocast():
    output = model(input)  # FP16 computation
    loss = criterion(output, target)  # Loss stays in FP32
    
# Gradients computed in FP16, master weights stay FP32`} />

          <Title order={4} mt="md">3. Complete Training Loop</Title>
          <Text>Combining all components for mixed precision training:</Text>
          
          <CodeBlock language="python" code={`def train_mixed_precision(model, dataloader, optimizer):
    scaler = GradScaler()
    
    for data, target in dataloader:
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights with unscaled gradients
        scaler.step(optimizer)
        scaler.update()`} />

        

          <Title order={3} mt="lg">Gradient Checkpointing</Title>
          
          <Text>
            During standard backpropagation, all intermediate activations from the forward pass are 
            stored in memory to compute gradients. Gradient checkpointing strategically discards 
            some activations and recomputes them during the backward pass when needed.
          </Text>

            <Title order={4}>How It Works</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Forward Pass:</strong> Only save activations at checkpoint boundaries</List.Item>
              <List.Item><strong>Backward Pass:</strong> Recompute intermediate activations from nearest checkpoint</List.Item>
              <List.Item><strong>Trade-off:</strong> ~30% slower training for ~60% memory reduction</List.Item>
              <List.Item><strong>Best for:</strong> Very deep networks with sequential layers</List.Item>
            </List>

          <CodeBlock language="python" code={`from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Without checkpointing: stores all intermediate values
        # x -> linear_out -> norm_out -> activation_out
        
        # With checkpointing: only stores input and output
        if self.training:
            # Wrap the computation in checkpoint
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)`} />

          <Title order={3} mt="lg">2. CPU Offloading</Title>
          
          <Text>
            Modern optimizers like Adam maintain momentum and variance buffers that double or triple 
            the memory needed for parameters. CPU offloading moves these optimizer states to system RAM, 
            keeping only the model weights on GPU.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Memory Breakdown for Adam Optimizer</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Model Parameters:</strong> Original weights (FP32)</List.Item>
              <List.Item><strong>Gradients:</strong> Same size as parameters</List.Item>
              <List.Item><strong>First Moment (m):</strong> Running average of gradients</List.Item>
              <List.Item><strong>Second Moment (v):</strong> Running average of squared gradients</List.Item>
              <List.Item><strong>Total:</strong> 4× parameter memory for Adam vs 2× for SGD</List.Item>
            </List>
          </Paper>

          <Text mt="md">
            By offloading optimizer states to CPU, we reduce GPU memory from 4× to 2× parameter size:
          </Text>

          <CodeBlock language="python" code={`# Simplified CPU offloading concept
class CPUOffloadAdam:
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
        # Keep optimizer states on CPU instead of GPU
        self.m = {id(p): torch.zeros_like(p, device='cpu') for p in self.params}
        self.v = {id(p): torch.zeros_like(p, device='cpu') for p in self.params}
        self.t = 0
    
    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            
            # Transfer gradient to CPU for computation
            grad_cpu = p.grad.cpu()
            
            # Update moments on CPU (no GPU memory used)
            self.m[id(p)] = 0.9 * self.m[id(p)] + 0.1 * grad_cpu
            self.v[id(p)] = 0.999 * self.v[id(p)] + 0.001 * grad_cpu**2
            
            # Compute update on CPU
            m_hat = self.m[id(p)] / (1 - 0.9**self.t)
            v_hat = self.v[id(p)] / (1 - 0.999**self.t)
            update = self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            
            # Apply update to GPU parameters
            p.data.add_(update.to(p.device), alpha=-1)`} />

          <Title order={3} mt="lg">3. Memory Efficient Attention</Title>
          
          <Text>
            Standard attention mechanisms have O(n²) memory complexity for sequence length n. 
            Memory-efficient implementations like Flash Attention reorganize computations to 
            reduce memory usage while maintaining mathematical equivalence.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Flash Attention Key Concepts</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Tiling:</strong> Process attention in blocks instead of full matrices</List.Item>
              <List.Item><strong>Kernel Fusion:</strong> Combine multiple operations to avoid intermediate storage</List.Item>
              <List.Item><strong>Recomputation:</strong> Recompute softmax normalization instead of storing</List.Item>
              <List.Item><strong>Benefits:</strong> 10-100× less memory, 2-4× faster on long sequences</List.Item>
            </List>
          </Paper>

          <CodeBlock language="python" code={`# Standard attention (high memory usage)
def standard_attention(Q, K, V):
    # Q, K, V: [batch, heads, seq_len, dim]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    # scores: [batch, heads, seq_len, seq_len] - O(n²) memory!
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

# Using Flash Attention (memory efficient)
from flash_attn import flash_attn_func

def efficient_attention(Q, K, V):
    # Flash Attention processes in blocks, never materializing full attention matrix
    # Memory: O(n) instead of O(n²)
    output = flash_attn_func(Q, K, V, dropout_p=0.0, causal=False)
    return output`} />

          <Title order={3} mt="lg">4. Activation Recomputation</Title>
          
          <Text>
            Similar to gradient checkpointing but more fine-grained. Instead of saving all 
            activations or checkpointing entire layers, selectively recompute expensive 
            but memory-light operations while caching memory-heavy but compute-light operations.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Smart Recomputation Strategy</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Recompute:</strong> Activation functions (GELU, ReLU), dropout, normalization</List.Item>
              <List.Item><strong>Cache:</strong> Linear projections, convolutions (memory-heavy)</List.Item>
              <List.Item><strong>Rationale:</strong> Activations are fast to recompute but large to store</List.Item>
            </List>
          </Paper>

          <CodeBlock language="python" code={`class SelectiveRecomputation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)  # Expensive to recompute
        self.linear2 = nn.Linear(dim * 4, dim)  # Expensive to recompute
        self.activation = nn.GELU()  # Cheap to recompute
        self.dropout = nn.Dropout(0.1)  # Cheap to recompute
    
    def forward(self, x):
        # Cache the expensive linear computation
        hidden = self.linear1(x)
        
        if self.training:
            # Don't save activation output - recompute in backward
            hidden = checkpoint(lambda h: self.dropout(self.activation(h)), hidden)
        else:
            hidden = self.dropout(self.activation(hidden))
        
        return self.linear2(hidden)`} />


        
          <Title id="io-optimization" order={2} mt="xl">I/O and DataLoader Optimization</Title>
          
          <Text mb="md">
            <strong>I/O (Input/Output)</strong> refers to reading data from disk/memory and transferring it to the GPU. 
            This can become a bottleneck if the GPU processes data faster than it can be loaded, leaving the GPU idle.
          </Text>

          <Title order={3}>Understanding DataLoader Parameters</Title>

          <Title order={4} mt="md">1. num_workers - Parallel Data Loading</Title>
          <Text mb="sm">
            Controls how many subprocesses load data in parallel. Each worker loads batches independently, 
            allowing data preparation to happen while the GPU processes the current batch.
          </Text>
          <CodeBlock language="python" code={`# Default: single-threaded loading (slow)
dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

# Optimized: parallel loading with multiple workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)`} />

          <Title order={4} mt="md">2. pin_memory - Faster GPU Transfer</Title>
          <Text mb="sm">
            Allocates data in page-locked (pinned) memory, enabling faster transfer from CPU to GPU. 
            This bypasses one memory copy operation during the transfer.
          </Text>
          <CodeBlock language="python" code={`# Without pinned memory: CPU → Pageable → Pinned → GPU
dataloader = DataLoader(dataset, pin_memory=False)

# With pinned memory: CPU → Pinned → GPU (faster)
dataloader = DataLoader(dataset, pin_memory=True)`} />

          <Title order={4} mt="md">3. persistent_workers - Avoid Worker Restart</Title>
          <Text mb="sm">
            Keeps worker processes alive between epochs instead of shutting down and restarting them. 
            This eliminates the overhead of creating new processes each epoch.
          </Text>
          <CodeBlock language="python" code={`# Workers restart each epoch (overhead)
dataloader = DataLoader(dataset, num_workers=4, persistent_workers=False)

# Workers stay alive (faster epoch transitions)
dataloader = DataLoader(dataset, num_workers=4, persistent_workers=True)`} />

          <Title order={4} mt="md">4. prefetch_factor - Batch Prefetching</Title>
          <Text mb="sm">
            Number of batches each worker prefetches. Workers prepare future batches while the model 
            processes the current one, creating a buffer to prevent GPU starvation.
          </Text>
          <CodeBlock language="python" code={`# Each worker prefetches 2 batches (default)
dataloader = DataLoader(dataset, num_workers=4, prefetch_factor=2)

# Increase for more aggressive prefetching (uses more memory)
dataloader = DataLoader(dataset, num_workers=4, prefetch_factor=4)`} />

          <Title order={3} mt="lg">Complete Optimized DataLoader</Title>
          <Text mb="sm">
            Combining all optimizations for maximum throughput:
          </Text>
          <CodeBlock language="python" code={`import torch
from torch.utils.data import DataLoader
import multiprocessing

def create_optimized_dataloader(dataset, batch_size=32, device='cuda'):
    """Create a DataLoader with optimal settings for GPU training"""
    
    # Determine optimal number of workers
    num_cores = multiprocessing.cpu_count()
    num_workers = min(num_cores - 1, 8)  # Leave one core for main process
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,      # Parallel loading
        pin_memory=True,               # Fast GPU transfer
        persistent_workers=True,       # Avoid worker restart
        prefetch_factor=2,            # Prefetch batches
        drop_last=True                # Consistent batch sizes
    )
    
    return dataloader`} />

          <Title order={3} mt="lg">Benchmarking I/O Performance</Title>
          <Text mb="sm">
            Measure and compare loading speeds to identify bottlenecks:
          </Text>
          <CodeBlock language="python" code={`import time

def benchmark_dataloader(dataloader, num_batches=100, device='cuda'):
    """Benchmark data loading and transfer speed"""
    
    # Warmup
    for i, (data, target) in enumerate(dataloader):
        if i >= 5:
            break
    
    # Benchmark
    start_time = time.perf_counter()
    
    for i, (data, target) in enumerate(dataloader):
        # Transfer to GPU with non_blocking for async transfer
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Synchronize to measure actual transfer time
        torch.cuda.synchronize()
        
        if i >= num_batches:
            break
    
    elapsed = time.perf_counter() - start_time
    throughput = num_batches / elapsed
    
    print(f"Loading speed: {throughput:.1f} batches/sec")
    print(f"Time per batch: {elapsed/num_batches*1000:.1f} ms")`} />

          <Title order={3} mt="lg">Comparing Different Configurations</Title>
          <Text mb="sm">
            Test the impact of each optimization parameter:
          </Text>
          <CodeBlock language="python" code={`# Compare configurations
configs = [
    {"num_workers": 0, "pin_memory": False},  # Baseline
    {"num_workers": 4, "pin_memory": False},  # Add workers
    {"num_workers": 4, "pin_memory": True},   # Add pinned memory
    {"num_workers": 4, "pin_memory": True, 
     "persistent_workers": True}              # Full optimization
]

for config in configs:
    loader = DataLoader(dataset, batch_size=128, **config)
    # Benchmark each configuration...`} />

          <Title order={3} mt="lg">Memory-Efficient Dataset Design</Title>
          <Text mb="sm">
            Pre-allocate data when possible to avoid repeated allocations:
          </Text>
          <CodeBlock language="python" code={`class OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, input_dim=1024):
        # Pre-allocate all data in memory once
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, 10, (size,))
    
    def __getitem__(self, idx):
        # Return pre-allocated tensors (no new allocations)
        return self.data[idx], self.targets[idx]`} />


          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/data_pipeline_optimization.png"
              alt="Data Pipeline Optimization Flow"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        
      </Stack>
    </Container>
  );
};

export default PerformanceOptimization;