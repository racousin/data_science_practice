import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Group, Badge, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';

const ResourceProfiling = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <Title order={1}>Resource Profiling & Memory Management</Title>
        
        <Text>
          Understanding how deep learning models utilize system resources is crucial for optimization.
          We'll explore how different model components consume memory and compute resources.
        </Text>

        
          <Title id="model-components" order={2} mt="xl">Model Component Analysis</Title>
          
          <Text>
            A trained neural network consists of several memory-consuming components during training:
          </Text>

            <Title order={4}>Memory Components Breakdown</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Model Parameters:</strong> Weights and biases of the network</List.Item>
              <List.Item><strong>Gradients:</strong> Same size as parameters, stored during backpropagation</List.Item>
              <List.Item><strong>Optimizer States:</strong> Momentum, variance (Adam uses 2x parameter memory)</List.Item>
              <List.Item><strong>Activations:</strong> Intermediate outputs saved for backward pass</List.Item>
              <List.Item><strong>Temporary Buffers:</strong> Workspace for operations like convolutions</List.Item>
            </List>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/distribution.png"
              alt="Model Memory Components Visualization"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        

        
          
            <Title id="memory-breakdown" order={2}>Memory Formula for Training</Title>
            
            <Text>
              During training, the total memory consumption is the sum of all components:
            </Text>
            
            <BlockMath>
              {`\\text{Total Memory} = \\text{Parameters} + \\text{Gradients} + \\text{Optimizer States} + \\text{Activations}`}
            </BlockMath>

            <Title order={3} mt="lg">Parameters Memory</Title>
            <Text>
              Model parameters (weights and biases) are the core storage requirement:
            </Text>
            <BlockMath>
              {`\\text{Parameter Memory} = N_{\\text{params}} \\times \\text{bytes\\_per\\_element}`}
            </BlockMath>
            
            <Text mt="sm">
              <strong>Example:</strong> A model with 1,000,000 parameters using float32 (4 bytes):
            </Text>
            <BlockMath>
              {`\\text{Memory} = 1,000,000 \\times 4\\text{ bytes} = 4\\text{ MB}`}
            </BlockMath>
            
            <Alert color="blue" mt="md">
              <Text size="sm">
                <strong>Note:</strong> During inference, only parameter memory is needed: <InlineMath>{`\\text{Inference Memory} = \\text{Parameters}`}</InlineMath>
              </Text>
            </Alert>

            <Title order={3} mt="lg">Gradients Memory</Title>
            <Text>
              Gradients are computed for all trainable parameters during backpropagation:
            </Text>
            <BlockMath>
              {`\\text{Gradient Memory} = N_{\\text{trainable}} \\times \\text{bytes\\_per\\_element}`}
            </BlockMath>
            
            
            <Text mt="sm">
              <strong>Example:</strong> Same 1M parameter model with all parameters trainable:
            </Text>
            <BlockMath>
              {`\\text{Gradient Memory} = 1,000,000 \\times 4\\text{ bytes} = 4\\text{ MB}`}
            </BlockMath>
            
            <Title order={3} mt="lg">Optimizer Memory</Title>
            <Text>
              Memory requirements vary by optimizer type. Common optimizers:
            </Text>
            
            <List mt="md" spacing="sm">
              <List.Item>
                <strong>SGD (no momentum):</strong> No additional memory
                <BlockMath>{`\\text{Memory}_{\\text{SGD}} = 0`}</BlockMath>
              </List.Item>
              
              <List.Item>
                <strong>SGD with momentum:</strong> Stores velocity for each parameter
                <BlockMath>{`\\text{Memory}_{\\text{SGD+momentum}} = P`}</BlockMath>
              </List.Item>
              
              <List.Item>
                <strong>Adam:</strong> Stores first and second moment estimates
                <BlockMath>{`\\text{Memory}_{\\text{Adam}} = 2P`}</BlockMath>
              </List.Item>
            </List>

            <Text size="sm" c="dimmed">Where P is the parameter memory size</Text>
          <Text mt="sm">
              <strong>Example:</strong> 1M parameter model with adam:
            </Text>
            <BlockMath>
              {`\\text{Gradient Memory} = 1,000,000 \\times 4\\text{ bytes} \\times 2 = 8\\text{ MB}`}
            </BlockMath>
        <Title order={3} mt="lg">Activation Memory</Title>
          
          <Text>
            Activations are intermediate outputs saved during forward pass for use in backpropagation.
            Unlike parameters and gradients, activation memory scales with batch size.
          </Text>

          <Text>For a fully connected layer:</Text>

          <BlockMath>
            {`\\text{Total Activations} = \\sum_{l=1}^{L} B \\times H_l \\times \\text{bytes\\_per\\_element}`}
          </BlockMath>
          
          <Text mt="sm">
            <strong>Example:</strong> 3-layer MLP with batch size 64, float32:
          </Text>
          <List size="xm" mt="sm">
            <List.Item>Layer 1: 64 × 512 × 4 = 131,072 bytes (0.13 MB)</List.Item>
            <List.Item>Layer 2: 64 × 256 × 4 = 65,536 bytes (0.07 MB)</List.Item>
            <List.Item>Total: ~0.2 MB per forward pass</List.Item>
          </List>
          
            <Text size="xm">
              Activation memory can dominate total memory usage with large batch sizes or deep networks!
            </Text>


<Title order={2} mt="xl">Memory Profiling Tools</Title>

          <Text>
            Let's build a memory profiler to track how different components consume memory during training.
          </Text>

          <Title order={3} mt="lg">Basic Memory Tracker</Title>
          
          <Text>
            First, create a simple utility to measure current GPU memory usage:
          </Text>

          <CodeBlock language="python" code={`import torch

def get_memory_stats():
    """Get current GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        return {'allocated': allocated, 'reserved': reserved}
    return {'allocated': 0, 'reserved': 0}`} />

          <Text mt="md">
            Track memory changes during model operations:
          </Text>

          <CodeBlock language="python" code={`def measure_memory_delta(func, *args, **kwargs):
    """Measure memory change from a function call"""
    torch.cuda.empty_cache()
    before = get_memory_stats()
    
    result = func(*args, **kwargs)
    
    after = get_memory_stats()
    delta = after['allocated'] - before['allocated']
    
    print(f"Memory delta: +{delta:.2f} MB")
    return result, delta`} />

          <Title order={3} mt="lg">Component-Wise Memory Analysis</Title>
          
          <Text>
            Profile memory usage of individual model components:
          </Text>

          <CodeBlock language="python" code={`class ComponentProfiler:
    def __init__(self, model):
        self.model = model
        self.stats = {}
    
    def profile_parameters(self):
        """Calculate parameter memory"""
        total = 0
        for name, param in self.model.named_parameters():
            memory = param.numel() * param.element_size() / 1024**2
            self.stats[f'param_{name}'] = memory
            total += memory
        return total`} />

          <Text mt="md">
            Add gradient tracking during backward pass:
          </Text>

          <CodeBlock language="python" code={`    def profile_gradients(self):
        """Calculate gradient memory after backward"""
        total = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                memory = param.grad.numel() * param.grad.element_size() / 1024**2
                self.stats[f'grad_{name}'] = memory
                total += memory
        return total`} />

          <Title order={3} mt="lg">Activation Memory Tracking</Title>
          
          <Text>
            Use hooks to monitor activation memory during forward pass:
          </Text>

          <CodeBlock language="python" code={`class ActivationProfiler:
    def __init__(self):
        self.activations = []
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        """Hook to capture activation sizes"""
        if isinstance(output, torch.Tensor):
            memory_mb = output.numel() * output.element_size() / 1024**2
            self.activations.append({
                'layer': module.__class__.__name__,
                'shape': list(output.shape),
                'memory_mb': memory_mb
            })`} />

          <Text mt="md">
            Register hooks and run profiling:
          </Text>

          <CodeBlock language="python" code={`    def attach_hooks(self, model):
        """Attach hooks to all layers"""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Clean up hooks after profiling"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []`} />

          <Title order={3} mt="lg">Complete Training Memory Profile</Title>
          
          <Text>
            Put it all together to profile a complete training step:
          </Text>

          <CodeBlock language="python" code={`def profile_training_step(model, data, target, optimizer):
    """Profile memory during one training iteration"""
    profiler = ComponentProfiler(model)
    activation_profiler = ActivationProfiler()
    
    print("=== Memory Profile ===")
    
    # Parameters
    param_mem = profiler.profile_parameters()
    print(f"Parameters: {param_mem:.2f} MB")
    
    # Forward pass with activation tracking
    activation_profiler.attach_hooks(model)
    output = model(data)
    activation_profiler.remove_hooks()
    
    act_mem = sum(a['memory_mb'] for a in activation_profiler.activations)
    print(f"Activations: {act_mem:.2f} MB")
    
    # Backward pass
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    
    grad_mem = profiler.profile_gradients()
    print(f"Gradients: {grad_mem:.2f} MB")
    
    # Optimizer step (tracks optimizer state memory)
    before_opt = get_memory_stats()['allocated']
    optimizer.step()
    after_opt = get_memory_stats()['allocated']
    opt_mem = after_opt - before_opt
    print(f"Optimizer states: {opt_mem:.2f} MB")
    
    total = param_mem + act_mem + grad_mem + opt_mem
    print(f"\\nTotal: {total:.2f} MB")
    
    return profiler.stats`} />

          <Title order={3} mt="lg">Usage Example</Title>
          
          <Text>
            Profile a simple model to understand memory distribution:
          </Text>

          <CodeBlock language="python" code={`# Create model and move to GPU
model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).cuda()

# Setup training components
optimizer = torch.optim.Adam(model.parameters())
data = torch.randn(32, 1024).cuda()  # Batch size 32
target = torch.randint(0, 10, (32,)).cuda()

# Profile one training step
stats = profile_training_step(model, data, target, optimizer)

# Analyze results
print("\\nDetailed breakdown:")
for key, value in sorted(stats.items()):
    if value > 0.01:  # Show only significant components
        print(f"  {key}: {value:.3f} MB")`} />

          <Title order={3} mt="lg">Gradient Checkpointing</Title>
          
          <Text>
            Gradient checkpointing trades compute for memory by recomputing activations during backward pass:
          </Text>

          <CodeBlock language="python" code={`from torch.utils.checkpoint import checkpoint

class CheckpointedMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Checkpoint the first layer computation
        x = checkpoint(self._forward_block1, x)
        x = self.fc2(x)
        return x
    
    def _forward_block1(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

# Memory comparison
print("Standard forward: stores all activations")
print("Checkpointed: recomputes fc1 activations during backward")`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/gradient_checkpointing.png"
              alt="Gradient Checkpointing Trade-off"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        

        
          <Title id="gpu-vs-cpu" order={2} mt="xl">GPU vs CPU Performance</Title>
          
          <Text>
            Understanding when to use GPU vs CPU is crucial for efficient training:
          </Text>

            <Title order={4}>Performance Characteristics</Title>
            <Group position="apart" mt="md">
              <div>
                <Badge color="blue" size="lg">CPU</Badge>
                <List size="sm" mt="sm">
                  <List.Item>Low latency for small operations</List.Item>
                  <List.Item>Better for sequential tasks</List.Item>
                  <List.Item>More memory available</List.Item>
                  <List.Item>Efficient for small batch sizes</List.Item>
                </List>
              </div>
              <div>
                <Badge color="green" size="lg">GPU</Badge>
                <List size="sm" mt="sm">
                  <List.Item>High throughput for parallel ops</List.Item>
                  <List.Item>Excellent for matrix operations</List.Item>
                  <List.Item>Limited memory (typically 8-80GB)</List.Item>
                  <List.Item>Efficient for large batch sizes</List.Item>
                </List>
              </div>
            </Group>

          <CodeBlock language="python" code={`import time

def benchmark_device(model, input_tensor, device, num_iterations=100):
    """Benchmark model performance on different devices"""
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Synchronize for GPU timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        output = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    throughput = num_iterations / elapsed_time
    
    print(f"Device: {device}")
    print(f"  Total time: {elapsed_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} iterations/s")
    print(f"  Avg time per iteration: {elapsed_time/num_iterations*1000:.2f}ms")
    
    return throughput

# Compare CPU vs GPU
model = MLP(input_size=1024, hidden_size=2048)
batch_sizes = [1, 16, 64, 256]

for batch_size in batch_sizes:
    print(f"\\nBatch size: {batch_size}")
    input_tensor = torch.randn(batch_size, 1024)
    
    cpu_throughput = benchmark_device(model, input_tensor, torch.device('cpu'))
    if torch.cuda.is_available():
        gpu_throughput = benchmark_device(model, input_tensor, torch.device('cuda'))
        print(f"  Speedup: {gpu_throughput/cpu_throughput:.1f}x")`} />

          <Title order={3} mt="lg">Memory Transfer Overhead</Title>
          
          <Text>
            Moving data between CPU and GPU can be a bottleneck:
          </Text>

          <CodeBlock language="python" code={`def measure_transfer_overhead(size_mb=100):
    """Measure CPU-GPU transfer overhead"""
    size_elements = int(size_mb * 1024 * 1024 / 4)  # 32-bit floats
    
    # Create CPU tensor
    cpu_tensor = torch.randn(size_elements)
    
    # Measure CPU to GPU
    start = time.time()
    gpu_tensor = cpu_tensor.cuda()
    torch.cuda.synchronize()
    cpu_to_gpu_time = time.time() - start
    
    # Measure GPU to CPU
    start = time.time()
    cpu_tensor_back = gpu_tensor.cpu()
    torch.cuda.synchronize()
    gpu_to_cpu_time = time.time() - start
    
    print(f"Transfer {size_mb}MB:")
    print(f"  CPU → GPU: {cpu_to_gpu_time*1000:.2f}ms ({size_mb/cpu_to_gpu_time:.1f} MB/s)")
    print(f"  GPU → CPU: {gpu_to_cpu_time*1000:.2f}ms ({size_mb/gpu_to_cpu_time:.1f} MB/s)")
    
    return cpu_to_gpu_time, gpu_to_cpu_time

# Test different sizes
for size in [10, 100, 500]:
    measure_transfer_overhead(size)
    print()`} />

          <Alert title="Best Practices" color="green" mt="md">
            <List size="sm">
              <List.Item>Use GPU for large batch training and inference</List.Item>
              <List.Item>Minimize CPU-GPU transfers by keeping data on GPU</List.Item>
              <List.Item>Use pinned memory for faster transfers</List.Item>
              <List.Item>Profile your specific workload to find optimal device</List.Item>
            </List>
          </Alert>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/gpu_cpu_comparison.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        
      </Stack>
    </Container>
  );
};

export default ResourceProfiling;