import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Table, Badge, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';

const ModelComplexityAnalysis = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <Title order={1}>Model Complexity Analysis</Title>
        
        <Text>
          Understanding the computational complexity and memory requirements of neural networks 
          is essential for designing efficient models and choosing appropriate hardware.
        </Text>

          <Title id="flops-calculation" order={2} mt="xl">FLOPs Calculation for MLPs</Title>
          
          <Text>
            FLOPs (Floating Point Operations) measure the computational cost of a model.
            For MLPs, we primarily count matrix multiplications and additions.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>FLOPs Formula for Linear Layer</Title>
            <Text>For a linear layer with input size N, output size M, and batch size B:</Text>
            
            <BlockMath>
              {`\\text{FLOPs}_{\\text{forward}} = B \\times (2 \\times N \\times M - M)`}
            </BlockMath>
            
            <Text size="sm" c="dimmed" mt="sm">
              • Matrix multiplication: B × N × M multiplications + B × N × M additions<br/>
              • Bias addition: B × M additions (often negligible)
            </Text>
            
            <Text mt="md">For backward pass (gradients):</Text>
            <BlockMath>
              {`\\text{FLOPs}_{\\text{backward}} \\approx 2 \\times \\text{FLOPs}_{\\text{forward}}`}
            </BlockMath>
          </Paper>

          <CodeBlock language="python" code={`def calculate_mlp_flops(layers, batch_size=1, include_backward=True):
    """
    Calculate FLOPs for an MLP
    layers: list of layer sizes, e.g., [784, 256, 128, 10]
    """
    total_flops = 0
    
    for i in range(len(layers) - 1):
        input_size = layers[i]
        output_size = layers[i + 1]
        
        # Forward pass FLOPs
        # Matrix multiplication: 2 * input * output - output (multiply-add counted as 2 ops)
        forward_flops = batch_size * (2 * input_size * output_size - output_size)
        
        # Backward pass approximately 2x forward
        backward_flops = 2 * forward_flops if include_backward else 0
        
        layer_flops = forward_flops + backward_flops
        total_flops += layer_flops
        
        print(f"Layer {i+1}: {input_size} → {output_size}")
        print(f"  Forward FLOPs: {forward_flops:,}")
        if include_backward:
            print(f"  Backward FLOPs: {backward_flops:,}")
        print(f"  Total: {layer_flops:,}")
    
    print(f"\\nTotal FLOPs: {total_flops:,}")
    print(f"GFLOPs: {total_flops / 1e9:.3f}")
    
    return total_flops

# Example: Calculate FLOPs for a typical MLP
layers = [784, 512, 256, 128, 10]  # MNIST classifier
batch_size = 64
flops = calculate_mlp_flops(layers, batch_size)`} />

          <Title order={3} mt="lg">FLOPs vs MACs</Title>
          
          <Alert color="blue" mt="md">
            <Text size="sm">
              <strong>MACs (Multiply-Accumulate operations):</strong> Count fused multiply-add as 1 operation<br/>
              <strong>FLOPs:</strong> Count multiply and add separately as 2 operations<br/>
              <InlineMath>{`\\text{FLOPs} = 2 \\times \\text{MACs}`}</InlineMath>
            </Text>
          </Alert>

          <CodeBlock language="python" code={`import torch
from torch.profiler import profile, ProfilerActivity

def profile_model_flops(model, input_shape):
    """Use PyTorch profiler to measure actual FLOPs"""
    model.eval()
    inputs = torch.randn(input_shape)
    
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        with torch.no_grad():
            model(inputs)
    
    # Get FLOPs from profiler
    flops = sum([int(evt.flops) for evt in prof.events() if evt.flops])
    
    print(f"Measured FLOPs: {flops:,}")
    print(f"GFLOPs: {flops / 1e9:.3f}")
    
    # Print top operations by FLOPs
    events = sorted([evt for evt in prof.events() if evt.flops], 
                   key=lambda x: x.flops, reverse=True)[:5]
    
    print("\\nTop operations by FLOPs:")
    for evt in events:
        print(f"  {evt.name}: {evt.flops:,} FLOPs")
    
    return flops

# Profile an MLP
model = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

flops = profile_model_flops(model, (64, 784))`} />
        

        
          <Title id="memory-requirements" order={2} mt="xl">Memory Requirements Estimation</Title>
          
          <Text>
            Accurate memory estimation helps prevent out-of-memory errors and optimize batch sizes.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Complete Memory Breakdown</Title>
            
            <Table mt="md">
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Formula</th>
                  <th>When Needed</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Model Parameters</td>
                  <td><InlineMath>{`P \\times 4`}</InlineMath> bytes</td>
                  <td>Always</td>
                </tr>
                <tr>
                  <td>Gradients</td>
                  <td><InlineMath>{`P \\times 4`}</InlineMath> bytes</td>
                  <td>Training</td>
                </tr>
                <tr>
                  <td>Adam Optimizer</td>
                  <td><InlineMath>{`2P \\times 4`}</InlineMath> bytes</td>
                  <td>Training with Adam</td>
                </tr>
                <tr>
                  <td>Activations</td>
                  <td><InlineMath>{`\\sum_l B \\times H_l \\times 4`}</InlineMath> bytes</td>
                  <td>Forward + Backward</td>
                </tr>
                <tr>
                  <td>Batch Data</td>
                  <td><InlineMath>{`B \\times D \\times 4`}</InlineMath> bytes</td>
                  <td>Always</td>
                </tr>
              </tbody>
            </Table>
            
            <Text size="sm" c="dimmed" mt="sm">
              P = number of parameters, B = batch size, H_l = hidden size at layer l, D = data dimension
            </Text>
          </Paper>

          <CodeBlock language="python" code={`class MemoryEstimator:
    def __init__(self, model, input_shape, batch_size, optimizer='adam', mixed_precision=False):
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.dtype_size = 2 if mixed_precision else 4  # FP16 vs FP32
        
    def estimate_memory(self):
        """Estimate total memory requirements"""
        memory_breakdown = {}
        
        # 1. Model parameters
        param_count = sum(p.numel() for p in self.model.parameters())
        memory_breakdown['parameters'] = param_count * self.dtype_size
        
        # 2. Gradients (same size as parameters during training)
        memory_breakdown['gradients'] = param_count * self.dtype_size
        
        # 3. Optimizer states
        if self.optimizer == 'adam':
            # Adam stores momentum and variance
            memory_breakdown['optimizer'] = param_count * self.dtype_size * 2
        elif self.optimizer == 'sgd':
            # SGD with momentum
            memory_breakdown['optimizer'] = param_count * self.dtype_size
        else:
            memory_breakdown['optimizer'] = 0
        
        # 4. Activations (estimated based on forward pass)
        activation_memory = self._estimate_activations()
        memory_breakdown['activations'] = activation_memory
        
        # 5. Input batch
        input_memory = self.batch_size * np.prod(self.input_shape) * self.dtype_size
        memory_breakdown['input_batch'] = input_memory
        
        # Total
        total_memory = sum(memory_breakdown.values())
        
        # Print breakdown
        print("Memory Requirements Breakdown:")
        print("-" * 40)
        for component, memory in memory_breakdown.items():
            print(f"{component:15}: {memory / 1024**2:8.2f} MB ({memory / total_memory * 100:.1f}%)")
        print("-" * 40)
        print(f"{'Total':15}: {total_memory / 1024**2:8.2f} MB")
        
        return memory_breakdown
    
    def _estimate_activations(self):
        """Estimate activation memory using hooks"""
        activation_memory = 0
        
        def hook_fn(module, input, output):
            nonlocal activation_memory
            if isinstance(output, torch.Tensor):
                activation_memory += output.numel() * self.dtype_size
        
        hooks = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Dummy forward pass
        dummy_input = torch.randn(self.batch_size, *self.input_shape)
        with torch.no_grad():
            self.model(dummy_input)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        return activation_memory

# Example usage
import numpy as np

model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048),
    torch.nn.ReLU(),
    torch.nn.Linear(2048, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)
)

estimator = MemoryEstimator(
    model=model,
    input_shape=(1024,),
    batch_size=128,
    optimizer='adam',
    mixed_precision=False
)

memory_breakdown = estimator.estimate_memory()`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/memory_breakdown_chart.png"
              alt="Memory Breakdown Visualization"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        

        
          <Title id="batch-size-optimization" order={2} mt="xl">Batch Size vs Memory Trade-offs</Title>
          
          <Text>
            Batch size significantly impacts both memory usage and training dynamics.
            Finding the optimal batch size requires balancing multiple factors.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Batch Size Impact</Title>
            
            <List spacing="sm" mt="md">
              <List.Item>
                <strong>Linear scaling with activations:</strong> Activation memory scales linearly with batch size
              </List.Item>
              <List.Item>
                <strong>Fixed parameter memory:</strong> Model and optimizer memory independent of batch size
              </List.Item>
              <List.Item>
                <strong>Gradient noise:</strong> Smaller batches → noisier gradients → better generalization
              </List.Item>
              <List.Item>
                <strong>Hardware utilization:</strong> Larger batches → better GPU utilization
              </List.Item>
            </List>
          </Paper>

          <CodeBlock language="python" code={`def find_optimal_batch_size(model, input_shape, max_memory_gb=8, safety_factor=0.9):
    """
    Find the maximum batch size that fits in memory
    """
    max_memory_bytes = max_memory_gb * 1024**3 * safety_factor
    
    # Fixed memory (parameters, gradients, optimizer)
    param_count = sum(p.numel() for p in model.parameters())
    fixed_memory = param_count * 4 * 4  # Assuming Adam optimizer (4x params)
    
    # Memory per sample (activations + data)
    # Estimate with batch size 1
    dummy_input = torch.randn(1, *input_shape)
    
    # Track activation memory
    activation_memory_per_sample = 0
    
    def hook_fn(module, input, output):
        nonlocal activation_memory_per_sample
        if isinstance(output, torch.Tensor):
            activation_memory_per_sample += output.numel() * 4
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.ReLU)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    # Add input memory
    input_memory_per_sample = np.prod(input_shape) * 4
    memory_per_sample = activation_memory_per_sample + input_memory_per_sample
    
    # Calculate maximum batch size
    available_for_batch = max_memory_bytes - fixed_memory
    max_batch_size = int(available_for_batch / memory_per_sample)
    
    print(f"Memory Analysis:")
    print(f"  Fixed memory: {fixed_memory / 1024**3:.2f} GB")
    print(f"  Memory per sample: {memory_per_sample / 1024**2:.2f} MB")
    print(f"  Max batch size: {max_batch_size}")
    
    # Find power of 2 batch size
    optimal_batch_size = 2 ** int(np.log2(max_batch_size))
    print(f"  Recommended batch size (power of 2): {optimal_batch_size}")
    
    return optimal_batch_size

# Example
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048),
    torch.nn.ReLU(),
    torch.nn.Linear(2048, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 10)
)

optimal_bs = find_optimal_batch_size(
    model=model,
    input_shape=(1024,),
    max_memory_gb=16  # GPU memory
)`} />

          <Title order={3} mt="lg">Gradient Accumulation</Title>
          
          <Text>
            When desired batch size exceeds memory limits, use gradient accumulation:
          </Text>

          <CodeBlock language="python" code={`def train_with_gradient_accumulation(model, dataloader, optimizer, 
                                     effective_batch_size=256, 
                                     actual_batch_size=32):
    """
    Train with gradient accumulation to simulate larger batch sizes
    """
    accumulation_steps = effective_batch_size // actual_batch_size
    
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"Step {i+1}: Accumulated {accumulation_steps} mini-batches")
    
    print(f"Simulated batch size: {effective_batch_size}")
    print(f"Actual batch size: {actual_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")`} />
        

        
          <Title id="profiling-tools" order={2} mt="xl">PyTorch Profiling Tools</Title>
          
          <Text>
            PyTorch provides powerful profiling tools to analyze performance bottlenecks:
          </Text>

          <CodeBlock language="python" code={`from torch.profiler import profile, record_function, ProfilerActivity
import torch.utils.benchmark as benchmark

class ModelProfiler:
    def __init__(self, model, input_shape, device='cpu'):
        self.model = model.to(device)
        self.input_shape = input_shape
        self.device = device
    
    def profile_detailed(self, num_iterations=100):
        """Detailed profiling with PyTorch profiler"""
        inputs = torch.randn(self.input_shape).to(self.device)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True
        ) as prof:
            with record_function("model_inference"):
                for _ in range(num_iterations):
                    self.model(inputs)
        
        # Print profiler results
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        # Export to Chrome tracing
        prof.export_chrome_trace("trace.json")
        print("\\nTrace exported to trace.json (view in chrome://tracing)")
        
        return prof
    
    def benchmark_layers(self):
        """Benchmark individual layers"""
        inputs = torch.randn(self.input_shape).to(self.device)
        
        layer_times = []
        x = inputs
        
        for name, layer in self.model.named_children():
            # Time this layer
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            timer = benchmark.Timer(
                stmt='layer(x)',
                globals={'layer': layer, 'x': x}
            )
            
            result = timer.timeit(100)
            layer_times.append({
                'name': name,
                'time_ms': result.mean * 1000,
                'shape_in': list(x.shape),
                'shape_out': list(layer(x).shape)
            })
            
            x = layer(x)
        
        # Print results
        print("\\nLayer-wise Benchmark:")
        print("-" * 60)
        total_time = sum(lt['time_ms'] for lt in layer_times)
        
        for lt in layer_times:
            percentage = (lt['time_ms'] / total_time) * 100
            print(f"{lt['name']:20} {lt['time_ms']:8.3f}ms ({percentage:5.1f}%) "
                  f"{lt['shape_in']} → {lt['shape_out']}")
        
        print("-" * 60)
        print(f"{'Total':20} {total_time:8.3f}ms")
        
        return layer_times
    
    def memory_snapshot(self):
        """Take memory snapshot for analysis"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run model
            inputs = torch.randn(self.input_shape).to(self.device)
            outputs = self.model(inputs)
            
            # Get memory stats
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"\\nGPU Memory Stats:")
            print(f"  Currently allocated: {allocated:.2f} MB")
            print(f"  Currently reserved: {reserved:.2f} MB")
            print(f"  Peak allocated: {peak:.2f} MB")
            
            # Memory summary
            print(torch.cuda.memory_summary())
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'peak': peak
            }

# Example usage
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(2048, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)
)

profiler = ModelProfiler(
    model=model,
    input_shape=(32, 1024),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Run profiling
prof = profiler.profile_detailed(num_iterations=50)
layer_times = profiler.benchmark_layers()
if torch.cuda.is_available():
    memory_stats = profiler.memory_snapshot()`} />

          <Alert title="Profiling Best Practices" color="green" mt="md">
            <List size="sm">
              <List.Item>Always warm up the GPU before profiling (run a few iterations first)</List.Item>
              <List.Item>Use torch.cuda.synchronize() for accurate GPU timing</List.Item>
              <List.Item>Profile both forward and backward passes for training</List.Item>
              <List.Item>Export traces for visualization in Chrome (chrome://tracing)</List.Item>
              <List.Item>Monitor both time and memory to identify bottlenecks</List.Item>
            </List>
          </Alert>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/profiler_visualization.png"
              alt="PyTorch Profiler Visualization"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        
      </Stack>
    </Container>
  );
};

export default ModelComplexityAnalysis;