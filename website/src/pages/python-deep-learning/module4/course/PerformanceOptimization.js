import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Badge, List, Grid, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';

const PerformanceOptimization = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <div data-slide>
        <Title order={1}>Performance Optimization Techniques</Title>
        
        <Text>
          Modern deep learning frameworks provide numerous optimization techniques to accelerate training 
          and inference. We'll explore key strategies to maximize performance on available hardware.
        </Text>
</div>
<div data-slide>
        
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
</div>
<div data-slide>

          <CodeBlock language="python" code={`def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10))) #same output`} />
Arbitrary Python functions can be optimized by passing the callable to torch.compile. We can then call the returned optimized function in place of the original function.
</div>
<div data-slide>

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
</div>
    <div data-slide>
      <Stack spacing="md">
        <Title order={3}>torch.compile with Nested Modules</Title>
        
        <Text>
          torch.compile recursively compiles all nested function calls except built-ins and torch.* functions.
        </Text>

        <List spacing="xs">
          <List.Item>
            <strong>Top-Level Strategy:</strong> Compile at the highest level, disable for problematic components
          </List.Item>
          
          <List.Item>
            <strong>Test Incrementally:</strong> Verify individual modules before full integration
          </List.Item>

          <List.Item>
            <strong>First call slower:</strong> Compilation overhead on initial execution
          </List.Item>
          
          <List.Item>
            <strong>Static shapes preferred:</strong> Dynamic shapes trigger recompilation
          </List.Item>
        </List>
      </Stack>
    </div>
<div data-slide>
        
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


</div>
<div data-slide>
            <Title order={3} mt="lg">How Mixed Precision Works</Title>

          <List>
            <List.Item><strong>FP16:</strong> For most forward and backward pass computations</List.Item>
            <List.Item><strong>FP32:</strong> For operations needing higher precision (loss scaling, weight updates)</List.Item>
          </List>
                <Flex direction="column" align="center" mt="md">
                  <Image
                    src="/assets/python-deep-learning/module4/mixed.jpg"
                    alt="Matrix Multiplication Parallelization"
                    style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                    fluid
                  />
          
                </Flex>

</div>
        <div data-slide>
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

            <CodeBlock language="python" mt="md" code=
{`def forward(self, x):
        # Standard forward (stores all activations)
        # x = self.layer1(x)
        # x = self.layer2(x)
        
        # With checkpointing (recomputes layer1 in backward)
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return self.layer3(x)`}
            />
</div>
<div data-slide>
          <Title order={3} mt="lg">CPU Offloading</Title>
          
          <Text>
            Modern optimizers like Adam maintain momentum and variance buffers that double  
            the memory needed for parameters. CPU offloading moves these optimizer states to system RAM, 
            keeping only the model weights on GPU.
          </Text>

            <Title order={4}>Memory Breakdown for Adam Optimizer</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Model Parameters:</strong> Original weights (FP32)</List.Item>
              <List.Item><strong>Gradients:</strong> Same size as parameters</List.Item>
              <List.Item><strong>First Moment (m):</strong> Running average of gradients</List.Item>
              <List.Item><strong>Second Moment (v):</strong> Running average of squared gradients</List.Item>
              <List.Item><strong>Total:</strong> 4× parameter memory for Adam vs 2× for SGD</List.Item>
            </List>

          <Text mt="md">
            By offloading optimizer states to CPU, we reduce GPU memory from 4× to 2× parameter size:
          </Text>

          <CodeBlock language="python" mt="md" code=
{`# Standard optimizer - everything on GPU
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# CPU offloading with libraries like DeepSpeed or FairScale
from fairscale.optim import OSS

# Offload optimizer states to CPU
optimizer = OSS(
    params=model.parameters(),
    optim=torch.optim.Adam,
    optim_params={"lr": 1e-3},
)`}/>
          </div>
<div data-slide>
          <Title order={3} mt="lg">Memory Efficient Attention</Title>
          
          <Text>
            Standard attention mechanisms have O(n²) memory complexity for sequence length n. 
            Memory-efficient implementations like Flash Attention reorganize computations to 
            reduce memory usage while maintaining mathematical equivalence.
            {' '}<a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noopener noreferrer">
              Read the Flash Attention paper for more details
            </a>.
          </Text>

            <Title order={4}>Flash Attention Key Concepts</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Tiling:</strong> Process attention in blocks instead of full matrices</List.Item>
              <List.Item><strong>Kernel Fusion:</strong> Combine multiple operations to avoid intermediate storage</List.Item>
              <List.Item><strong>Recomputation:</strong> Recompute softmax normalization instead of storing</List.Item>
              <List.Item><strong>Benefits:</strong> 10-100× less memory, 2-4× faster on long sequences</List.Item>
            </List>
</div>

       <div data-slide>
          <Title id="io-optimization" order={2} mt="xl">I/O and DataLoader Optimization</Title>
          
          <Text mb="md">
            <strong>I/O (Input/Output)</strong> refers to reading data from disk/memory and transferring it to the GPU. 
            This can become a bottleneck if the GPU processes data faster than it can be loaded, leaving the GPU idle.
          </Text>

          <Title order={3}>DataLoader Parameters</Title>

          <Title order={4} mt="md">1. num_workers - Parallel Data Loading</Title>
          <Text mb="sm">
            Controls how many subprocesses load data in parallel. Each worker loads batches independently, 
            allowing data preparation to happen while the GPU processes the current batch.
          </Text>

                 
          <CodeBlock language="python" code={`# Default: single-threaded loading (slow)
dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

# Optimized: parallel loading with multiple workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)`} />
          </div> 
<div data-slide>
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
</div>
<div data-slide>
        
        <Title id="pruning-optimization" order={2} mt="xl">Inference optimization : Pruning Optimization</Title>
        
        <Text>
          Neural network pruning removes redundant parameters to create smaller, faster models without 
          significantly impacting accuracy. Most networks are overparameterized - many weights contribute 
          little to the final predictions and can be eliminated.
        </Text>

        <Title order={3} mt="lg">Core Concepts</Title>
        
        <Text>
          Pruning exploits the observation that neural networks contain significant redundancy. 
          By removing less important connections, we can achieve:
        </Text>
        
        <List spacing="sm" mt="md">
          <List.Item><strong>Reduced Model Size:</strong> 50-90% parameter reduction is common</List.Item>
          <List.Item><strong>Faster Inference:</strong> Less computation with sparse operations</List.Item>
          <List.Item><strong>Maintained Accuracy:</strong> Often within 1-2% of original performance</List.Item>
        </List>
</div>
<div data-slide>
        <Title order={3} mt="lg">Main Pruning Strategies</Title>

        <Title order={4} mt="md">1. Magnitude-Based Pruning</Title>
        <Text>
          The simplest approach: remove weights with the smallest absolute values, 
          assuming they contribute least to the output.
        </Text>
        
        <Title order={4} mt="md">2. Structured Pruning</Title>
        <Text>
          Removes entire neurons, channels, or filters instead of individual weights. 
          This creates actual speedup on standard hardware without special sparse operations.
        </Text>
        
        <Title order={4} mt="md">3. Iterative Pruning with Fine-tuning</Title>
        <Text>
          Gradually prune the network in steps, retraining between each pruning iteration 
          to recover accuracy. This typically yields better results than one-shot pruning.
        </Text>
        
</div>

      </Stack>
    </Container>
  );
};

export default PerformanceOptimization;