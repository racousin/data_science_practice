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


<Title order={2} mt="xl">Memory Profiling with PyTorch Profiler</Title>

          <Text>
            PyTorch provides a powerful built-in profiler that can track CPU, GPU, and memory usage with minimal overhead.
          </Text>

          <Title order={3} mt="lg">Basic Profiler Setup</Title>
          
          <Text>
            Start with a simple profiler configuration to track memory allocation:
          </Text>

          <CodeBlock language="python" code={`import torch
from torch.profiler import profile, ProfilerActivity, record_function

# Basic profiler with memory tracking
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True,
             record_shapes=True) as prof:
    # Your model operations here
    model(input_tensor)

# Print memory summary
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))`} />

          <Title order={3} mt="lg">Memory Timeline Profiling</Title>
          
          <Text>
            Track memory allocation and deallocation over time:
          </Text>

          <CodeBlock language="python" code={`def profile_memory_timeline(model, input_data, steps=3):
    """Profile memory usage over multiple training steps"""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step in range(steps):
            with record_function(f"step_{step}"):
                output = model(input_data)
                loss = output.mean()
                loss.backward()
    
    # Export timeline for visualization
    prof.export_chrome_trace("memory_timeline.json")
    return prof`} />

          <Text mt="md">
            View the timeline in Chrome DevTools (chrome://tracing) for detailed visualization.
          </Text>

          <Title order={3} mt="lg">Detailed Operation Profiling</Title>
          
          <Text>
            Use record_function to annotate and track specific code sections:
          </Text>

          <CodeBlock language="python" code={`def profile_training_components(model, dataloader, optimizer):
    """Profile different components of training loop"""
    
    with profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 2:  # Profile only first 2 batches
                break
                
            with record_function("data_transfer"):
                data, target = data.cuda(), target.cuda()
            
            with record_function("forward"):
                output = model(data)
                
            with record_function("loss"):
                loss = F.cross_entropy(output, target)
            
            with record_function("backward"):
                loss.backward()
                
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
    
    return prof`} />

          <Title order={3} mt="lg">Memory Breakdown Analysis</Title>
          
          <Text>
            Analyze memory usage by operation type and layer:
          </Text>

          <CodeBlock language="python" code={`def analyze_memory_by_operation(prof):
    """Extract and analyze memory usage from profiler"""
    
    # Group by operation name
    events = prof.key_averages(group_by_input_shape=True)
    
    memory_stats = []
    for evt in events:
        if evt.self_cuda_memory_usage > 0:
            memory_stats.append({
                'name': evt.key,
                'calls': evt.count,
                'memory_mb': evt.self_cuda_memory_usage / 1024**2,
                'memory_per_call': evt.self_cuda_memory_usage / evt.count / 1024**2
            })
    
    # Sort by total memory usage
    memory_stats.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    print("Top Memory Consuming Operations:")
    for stat in memory_stats[:10]:
        print(f"  {stat['name']}: {stat['memory_mb']:.2f} MB "
              f"({stat['calls']} calls, {stat['memory_per_call']:.3f} MB/call)")`} />

          <Title order={3} mt="lg">Layer-wise Memory Profiling</Title>
          
          <Text>
            Profile memory consumption for each layer in your model:
          </Text>

          <CodeBlock language="python" code={`class LayerMemoryProfiler:
    def __init__(self, model):
        self.model = model
        self.memory_usage = {}
        
    def profile_layers(self, input_data):
        """Profile each layer's memory consumption"""
        
        def make_hook(name):
            def hook(module, input, output):
                with record_function(f"layer_{name}"):
                    pass  # Profiler will track memory here
            return hook
        
        # Add hooks to all layers
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Run profiling
        with profile(
            activities=[ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                self.model(input_data)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        return prof`} />

          <Title order={3} mt="lg">Memory Optimization Detection</Title>
          
          <Text>
            Identify memory optimization opportunities:
          </Text>

          <CodeBlock language="python" code={`def find_memory_bottlenecks(prof):
    """Identify potential memory optimization targets"""
    
    events = prof.key_averages()
    
    # Find peak memory operations
    peak_memory_ops = []
    for evt in events:
        if evt.self_cuda_memory_usage > 0:
            efficiency = evt.cuda_time_total / evt.self_cuda_memory_usage if evt.cuda_time_total > 0 else 0
            peak_memory_ops.append({
                'op': evt.key,
                'memory_mb': evt.self_cuda_memory_usage / 1024**2,
                'time_ms': evt.cuda_time_total / 1000,
                'efficiency': efficiency
            })
    
    # Sort by memory usage
    peak_memory_ops.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    print("Memory Optimization Candidates:")
    print("(High memory + low efficiency = optimization opportunity)\\n")
    
    for op in peak_memory_ops[:5]:
        print(f"Operation: {op['op']}")
        print(f"  Memory: {op['memory_mb']:.2f} MB")
        print(f"  Time: {op['time_ms']:.2f} ms")
        print(f"  Efficiency: {op['efficiency']:.4f}\\n")`} />

          <Title order={3} mt="lg">Complete Profiling Example</Title>
          
          <Text>
            Full example with visualization and analysis:
          </Text>

          <CodeBlock language="python" code={`# Setup model and data
model = torchvision.models.resnet18().cuda()
optimizer = torch.optim.Adam(model.parameters())
input_batch = torch.randn(16, 3, 224, 224).cuda()
target = torch.randint(0, 1000, (16,)).cuda()

# Profile with all features enabled
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    # Training step
    output = model(input_batch)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

# Generate reports
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage", 
    row_limit=15
))

# Export for visualization
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
prof.export_stacks("profiler_stacks.txt", "self_cuda_memory_usage")

# Tensorboard integration
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/profile')
writer.add_text('profile', prof.key_averages().table())
writer.close()

print("\\nProfile exported. View with:")
print("  - Chrome: chrome://tracing (load trace.json)")
print("  - TensorBoard: tensorboard --logdir=runs")`} />

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