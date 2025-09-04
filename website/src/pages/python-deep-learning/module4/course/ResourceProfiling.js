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

        <section id="model-components">
          <Title order={2} mt="xl">Model Component Analysis</Title>
          
          <Text>
            A trained neural network consists of several memory-consuming components during training:
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Memory Components Breakdown</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Model Parameters:</strong> Weights and biases of the network</List.Item>
              <List.Item><strong>Gradients:</strong> Same size as parameters, stored during backpropagation</List.Item>
              <List.Item><strong>Optimizer States:</strong> Momentum, variance (Adam uses 2x parameter memory)</List.Item>
              <List.Item><strong>Activations:</strong> Intermediate outputs saved for backward pass</List.Item>
              <List.Item><strong>Temporary Buffers:</strong> Workspace for operations like convolutions</List.Item>
            </List>
          </Paper>

          <CodeBlock language="python" code={`import torch
import torch.nn as nn

# Simple MLP for demonstration
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model and analyze memory
model = MLP()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")  # 32-bit floats`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/model_memory_components.png"
              alt="Model Memory Components Visualization"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </section>

        <section id="memory-breakdown">
          <Title order={2} mt="xl">Memory Breakdown: Parameters, Gradients, Optimizer</Title>
          
          <Text>
            Let's calculate the exact memory requirements for training an MLP:
          </Text>

          <Paper p="md" withBorder mt="md">
            <Title order={4}>Memory Formula for Training</Title>
            <BlockMath>
              {`\\text{Total Memory} = \\text{Parameters} + \\text{Gradients} + \\text{Optimizer States} + \\text{Activations}`}
            </BlockMath>
            
            <Text mt="md">For Adam optimizer:</Text>
            <BlockMath>
              {`\\text{Memory}_{\\text{Adam}} = P + P + 2P = 4P`}
            </BlockMath>
            <Text size="sm" c="dimmed">Where P is the parameter memory</Text>
          </Paper>

          <CodeBlock language="python" code={`def calculate_model_memory(model, batch_size=32, optimizer_type='adam'):
    """Calculate memory requirements for training"""
    # Count parameters
    param_memory = 0
    for p in model.parameters():
        param_memory += p.numel() * p.element_size()
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    # Optimizer memory
    if optimizer_type == 'adam':
        optimizer_memory = 2 * param_memory  # Momentum + variance
    elif optimizer_type == 'sgd':
        optimizer_memory = param_memory  # Only momentum
    else:
        optimizer_memory = 0
    
    # Print breakdown
    print(f"Parameter memory: {param_memory / 1024**2:.2f} MB")
    print(f"Gradient memory: {gradient_memory / 1024**2:.2f} MB")
    print(f"Optimizer memory: {optimizer_memory / 1024**2:.2f} MB")
    print(f"Total (excl. activations): {(param_memory + gradient_memory + optimizer_memory) / 1024**2:.2f} MB")
    
    return param_memory + gradient_memory + optimizer_memory

# Example usage
model = MLP(input_size=1024, hidden_size=512)
total_memory = calculate_model_memory(model, optimizer_type='adam')`} />

          <Alert title="Memory Scaling" color="blue" mt="md">
            With Adam optimizer, you need approximately 4x the model parameter memory just for training states,
            not including activations!
          </Alert>
        </section>

        <section id="activation-memory">
          <Title order={2} mt="xl">Activation Memory & Checkpointing</Title>
          
          <Text>
            Activations can consume significant memory, especially with large batch sizes. 
            The memory grows linearly with batch size and network depth.
          </Text>

          <Paper p="md" withBorder mt="md">
            <Title order={4}>Activation Memory Formula</Title>
            <Text>For a fully connected layer:</Text>
            <BlockMath>
              {`\\text{Activation Memory} = \\text{batch\\_size} \\times \\text{hidden\\_size} \\times \\text{bytes\\_per\\_element}`}
            </BlockMath>
            
            <Text mt="md">Total for entire network:</Text>
            <BlockMath>
              {`\\text{Total Activations} = \\sum_{l=1}^{L} B \\times H_l \\times 4`}
            </BlockMath>
            <Text size="sm" c="dimmed">Where B is batch size, H_l is hidden size at layer l</Text>
          </Paper>

          <CodeBlock language="python" code={`class MemoryProfiler:
    def __init__(self):
        self.activations = []
        
    def hook_fn(self, module, input, output):
        """Hook to track activation memory"""
        if isinstance(output, torch.Tensor):
            memory = output.numel() * output.element_size()
            self.activations.append({
                'layer': module.__class__.__name__,
                'shape': list(output.shape),
                'memory_mb': memory / 1024**2
            })
    
    def profile_model(self, model, input_tensor):
        """Profile activation memory usage"""
        hooks = []
        for module in model.modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(self.hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Print results
        total_memory = sum(a['memory_mb'] for a in self.activations)
        print(f"Total activation memory: {total_memory:.2f} MB")
        for act in self.activations:
            print(f"  {act['layer']}: {act['shape']} -> {act['memory_mb']:.3f} MB")
        
        return self.activations

# Profile a model
profiler = MemoryProfiler()
model = MLP(input_size=1024, hidden_size=512)
dummy_input = torch.randn(64, 1024)  # Batch size 64
activations = profiler.profile_model(model, dummy_input)`} />

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
        </section>

        <section id="gpu-vs-cpu">
          <Title order={2} mt="xl">GPU vs CPU Performance</Title>
          
          <Text>
            Understanding when to use GPU vs CPU is crucial for efficient training:
          </Text>

          <Paper p="md" withBorder mt="md">
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
          </Paper>

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
        </section>
      </Stack>
    </Container>
  );
};

export default ResourceProfiling;