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

        <section id="torch-compile">
          <Title order={2} mt="xl">Torch Compile & Graph Optimization</Title>
          
          <Text>
            PyTorch 2.0 introduced torch.compile, which optimizes models through graph compilation
            and fusion of operations, significantly improving performance.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>How Torch Compile Works</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Graph Capture:</strong> Traces Python execution to build computation graph</List.Item>
              <List.Item><strong>Graph Optimization:</strong> Fuses operations, eliminates redundancy</List.Item>
              <List.Item><strong>Code Generation:</strong> Generates optimized kernels for target hardware</List.Item>
              <List.Item><strong>Caching:</strong> Reuses compiled graphs for repeated calls</List.Item>
            </List>
          </Paper>

          <CodeBlock language="python" code={`import torch
import torch.nn as nn
import time

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Compare performance with and without compile
def benchmark_compile():
    model = MLP(input_size=1024, hidden_sizes=[2048, 1024, 512])
    compiled_model = torch.compile(model)  # Compile the model
    
    # Test input
    batch_size = 256
    x = torch.randn(batch_size, 1024)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
        _ = compiled_model(x)
    
    # Benchmark original model
    start = time.perf_counter()
    for _ in range(100):
        _ = model(x)
    original_time = time.perf_counter() - start
    
    # Benchmark compiled model
    start = time.perf_counter()
    for _ in range(100):
        _ = compiled_model(x)
    compiled_time = time.perf_counter() - start
    
    print(f"Original model: {original_time:.3f}s")
    print(f"Compiled model: {compiled_time:.3f}s")
    print(f"Speedup: {original_time/compiled_time:.2f}x")
    
benchmark_compile()`} />

          <Title order={3} mt="lg">Compilation Modes and Backends</Title>

          <CodeBlock language="python" code={`# Different compilation modes
model = MLP()

# Mode: default - balanced optimization
compiled_default = torch.compile(model)

# Mode: reduce-overhead - minimize framework overhead
compiled_reduce = torch.compile(model, mode="reduce-overhead")

# Mode: max-autotune - extensive tuning for best performance
compiled_max = torch.compile(model, mode="max-autotune")

# Custom backend selection
compiled_inductor = torch.compile(model, backend="inductor")  # Default, good for GPUs
compiled_onnx = torch.compile(model, backend="onnxrt")  # ONNX Runtime

# Disable certain optimizations
compiled_custom = torch.compile(
    model,
    options={
        "max_autotune": True,
        "triton.cudagraphs": True,  # Enable CUDA graphs
        "trace.enabled": True,       # Enable tracing
        "trace.graph_diagram": True  # Save graph visualization
    }
)

print("Compilation modes configured successfully")`} />

          <Alert title="Compilation Tips" color="blue" mt="md">
            <List size="sm">
              <List.Item>First call will be slower due to compilation overhead</List.Item>
              <List.Item>Best for static input shapes - dynamic shapes may cause recompilation</List.Item>
              <List.Item>Use fullgraph=True for maximum optimization if model has no Python control flow</List.Item>
              <List.Item>Monitor with TORCH_LOGS="+dynamo" for compilation insights</List.Item>
            </List>
          </Alert>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/torch_compile_workflow.png"
              alt="Torch Compile Workflow"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </section>

        <section id="mixed-precision">
          <Title order={2} mt="xl">Mixed Precision Training</Title>
          
          <Text>
            Mixed precision training uses both FP16 and FP32 computations to accelerate training
            while maintaining model accuracy.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Benefits of Mixed Precision</Title>
            <Grid mt="md">
              <Grid.Col span={6}>
                <Badge color="green" size="lg">Memory Savings</Badge>
                <Text size="sm" mt="xs">50% reduction in memory usage for activations and weights</Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Badge color="blue" size="lg">Speed Improvement</Badge>
                <Text size="sm" mt="xs">2-3x speedup on modern GPUs with Tensor Cores</Text>
              </Grid.Col>
            </Grid>
          </Paper>

          <CodeBlock language="python" code={`import torch
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, dataloader, optimizer, num_epochs=5):
    """Training with Automatic Mixed Precision (AMP)"""
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler()
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip if needed
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# Compare memory usage
def compare_precision_memory():
    model = MLP(input_size=1024, hidden_sizes=[2048, 2048, 1024])
    batch_size = 128
    x = torch.randn(batch_size, 1024).cuda()
    
    # FP32 forward pass
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output_fp32 = model.cuda()(x)
    fp32_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    # FP16 forward pass
    model_fp16 = model.half()
    x_fp16 = x.half()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output_fp16 = model_fp16(x_fp16)
    fp16_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"FP32 Peak Memory: {fp32_memory:.2f} MB")
    print(f"FP16 Peak Memory: {fp16_memory:.2f} MB")
    print(f"Memory Reduction: {(1 - fp16_memory/fp32_memory)*100:.1f}%")

if torch.cuda.is_available():
    compare_precision_memory()`} />

          <Title order={3} mt="lg">Loss Scaling and Gradient Management</Title>

          <Paper p="md" withBorder>
            <Title order={4}>Why Loss Scaling?</Title>
            <Text>
              FP16 has limited range (±65,504) compared to FP32 (±3.4×10³⁸).
              Small gradients can underflow to zero in FP16.
            </Text>
            
            <BlockMath>{`\\text{Scaled Loss} = \\text{Loss} \\times \\text{Scale Factor}`}</BlockMath>
            <BlockMath>{`\\text{True Gradient} = \\frac{\\text{Scaled Gradient}}{\\text{Scale Factor}}`}</BlockMath>
          </Paper>

          <CodeBlock language="python" code={`class ManualMixedPrecision:
    """Manual implementation to understand mixed precision mechanics"""
    
    def __init__(self, initial_scale=2**16):
        self.scale = initial_scale
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 2000
        self.steps_since_update = 0
        
    def scale_loss(self, loss):
        """Scale the loss for FP16 training"""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients before optimizer step"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def check_gradients(self, parameters):
        """Check for inf/nan in gradients"""
        for param in parameters:
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    return False
        return True
    
    def update_scale(self, gradients_valid):
        """Update loss scale based on gradient validity"""
        if gradients_valid:
            self.steps_since_update += 1
            if self.steps_since_update >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_update = 0
                print(f"Scale increased to {self.scale}")
        else:
            self.scale *= self.backoff_factor
            self.steps_since_update = 0
            print(f"Scale decreased to {self.scale}")

# Example usage
mp_trainer = ManualMixedPrecision()
model = MLP().cuda().half()  # Convert model to FP16
optimizer = torch.optim.Adam(model.parameters())

for data, target in dataloader:
    data, target = data.cuda().half(), target.cuda()
    
    # Forward pass in FP16
    output = model(data)
    loss = F.cross_entropy(output, target)
    
    # Scale loss and backward
    scaled_loss = mp_trainer.scale_loss(loss)
    scaled_loss.backward()
    
    # Check and unscale gradients
    if mp_trainer.check_gradients(model.parameters()):
        mp_trainer.unscale_gradients(optimizer)
        optimizer.step()
    
    optimizer.zero_grad()
    mp_trainer.update_scale(gradients_valid)`} />
        </section>

        <section id="memory-optimization">
          <Title order={2} mt="xl">Memory Optimization Strategies</Title>
          
          <Text>
            Efficient memory management enables training larger models and batch sizes on limited hardware.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Key Memory Optimization Techniques</Title>
            <List spacing="sm" mt="md">
              <List.Item><strong>Gradient Checkpointing:</strong> Trade compute for memory</List.Item>
              <List.Item><strong>CPU Offloading:</strong> Move optimizer states to CPU</List.Item>
              <List.Item><strong>Memory Efficient Attention:</strong> Flash Attention, xFormers</List.Item>
              <List.Item><strong>Activation Recomputation:</strong> Recompute instead of storing</List.Item>
              <List.Item><strong>Parameter Sharding:</strong> Distribute parameters across devices</List.Item>
            </List>
          </Paper>

          <CodeBlock language="python" code={`from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as F

class MemoryEfficientMLP(nn.Module):
    def __init__(self, input_size=1024, num_layers=8, hidden_size=2048):
        super().__init__()
        
        # Create deep network
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_size, 10)
        
        # Checkpointing configuration
        self.checkpoint_segments = 4  # Checkpoint every N layers
        
    def forward(self, x):
        # Split layers into segments for checkpointing
        segment_size = len(self.layers) // self.checkpoint_segments
        
        for i in range(0, len(self.layers), segment_size):
            segment = self.layers[i:i+segment_size]
            if self.training:
                # Use checkpointing during training
                x = checkpoint_sequential(segment, segment_size, x)
            else:
                # Normal forward during inference
                for layer in segment:
                    x = layer(x)
        
        return self.output(x)

# Memory-efficient optimizer with CPU offloading
class CPUOffloadOptimizer:
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
        # Store optimizer states on CPU
        self.momentum_buffers = {
            id(p): torch.zeros_like(p, device='cpu') 
            for p in self.params
        }
    
    def step(self):
        """Optimizer step with CPU offloading"""
        for param in self.params:
            if param.grad is None:
                continue
            
            # Move gradient to CPU
            grad_cpu = param.grad.cpu()
            
            # Update momentum on CPU
            momentum = self.momentum_buffers[id(param)]
            momentum.mul_(0.9).add_(grad_cpu, alpha=0.1)
            
            # Apply update on GPU
            param.data.add_(momentum.cuda(), alpha=-self.lr)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# Compare memory usage
def compare_memory_techniques():
    torch.cuda.reset_peak_memory_stats()
    
    # Standard model
    model_standard = MLP(input_size=1024, hidden_sizes=[2048]*8)
    model_standard.cuda()
    x = torch.randn(64, 1024).cuda()
    
    # Forward and backward
    loss = model_standard(x).sum()
    loss.backward()
    standard_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    # Memory-efficient model
    torch.cuda.reset_peak_memory_stats()
    model_efficient = MemoryEfficientMLP(input_size=1024, num_layers=8)
    model_efficient.cuda()
    
    loss = model_efficient(x).sum()
    loss.backward()
    efficient_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Standard model memory: {standard_memory:.2f} MB")
    print(f"Efficient model memory: {efficient_memory:.2f} MB")
    print(f"Memory saved: {(1 - efficient_memory/standard_memory)*100:.1f}%")

if torch.cuda.is_available():
    compare_memory_techniques()`} />

          <Alert title="Memory Profiling" color="yellow" mt="md">
            Use torch.cuda.memory_summary() to get detailed memory breakdown and identify optimization opportunities.
          </Alert>
        </section>

        <section id="io-optimization">
          <Title order={2} mt="xl">I/O and DataLoader Optimization</Title>
          
          <Text>
            Data loading can be a significant bottleneck. Optimizing I/O ensures GPU utilization remains high.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>DataLoader Optimization Parameters</Title>
            <Table mt="md">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Impact</th>
                  <th>Recommended Value</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>num_workers</td>
                  <td>Parallel data loading</td>
                  <td>2-4 × num_gpus</td>
                </tr>
                <tr>
                  <td>pin_memory</td>
                  <td>Faster GPU transfer</td>
                  <td>True for GPU training</td>
                </tr>
                <tr>
                  <td>persistent_workers</td>
                  <td>Avoid worker restart</td>
                  <td>True if memory allows</td>
                </tr>
                <tr>
                  <td>prefetch_factor</td>
                  <td>Samples to prefetch</td>
                  <td>2-4</td>
                </tr>
              </tbody>
            </Table>
          </Paper>

          <CodeBlock language="python" code={`import torch.utils.data as data
from torch.utils.data import DataLoader
import multiprocessing

class OptimizedDataLoader:
    def __init__(self, dataset, batch_size=32, device='cuda'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        
        # Optimal number of workers
        self.num_workers = min(multiprocessing.cpu_count(), 8)
        
    def get_dataloader(self):
        """Create optimized DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Pin memory for faster GPU transfer
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Prefetch 2 batches per worker
            drop_last=True  # Drop incomplete batches for consistent size
        )
    
    def benchmark_loading(self, num_batches=100):
        """Benchmark data loading speed"""
        dataloader = self.get_dataloader()
        
        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
        
        # Benchmark
        start_time = time.perf_counter()
        for i, (data, target) in enumerate(dataloader):
            # Transfer to GPU
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            if i >= num_batches:
                break
        
        elapsed = time.perf_counter() - start_time
        print(f"Loaded {num_batches} batches in {elapsed:.2f}s")
        print(f"Average time per batch: {elapsed/num_batches*1000:.2f}ms")
        print(f"Throughput: {num_batches * self.batch_size / elapsed:.0f} samples/s")

# Custom dataset with optimizations
class OptimizedDataset(data.Dataset):
    def __init__(self, size=10000, input_dim=1024):
        # Pre-allocate and cache data in memory
        self.data = torch.randn(size, input_dim, dtype=torch.float32)
        self.targets = torch.randint(0, 10, (size,), dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return pre-allocated tensors (no allocation in workers)
        return self.data[idx], self.targets[idx]

# Compare different DataLoader configurations
def compare_dataloader_configs():
    dataset = OptimizedDataset(size=50000)
    configs = [
        {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        {"num_workers": 4, "pin_memory": False, "persistent_workers": False},
        {"num_workers": 4, "pin_memory": True, "persistent_workers": False},
        {"num_workers": 4, "pin_memory": True, "persistent_workers": True},
    ]
    
    for config in configs:
        print(f"\\nConfig: {config}")
        loader = DataLoader(dataset, batch_size=128, **config)
        
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= 100:
                break
        elapsed = time.perf_counter() - start
        
        print(f"  Time for 100 batches: {elapsed:.2f}s")
        print(f"  Throughput: {100 * 128 / elapsed:.0f} samples/s")

compare_dataloader_configs()`} />

          <Title order={3} mt="lg">Data Pipeline Optimization</Title>

          <CodeBlock language="python" code={`# Advanced data pipeline with preprocessing on GPU
class GPUAugmentationPipeline:
    def __init__(self, device='cuda'):
        self.device = device
        
    def augment_batch(self, batch):
        """Apply augmentations on GPU for better performance"""
        # Move to GPU if not already
        if batch.device.type != 'cuda':
            batch = batch.to(self.device)
        
        # GPU-based augmentations (example)
        batch = batch + torch.randn_like(batch) * 0.1  # Add noise
        batch = F.dropout2d(batch, p=0.1, training=True)  # Random dropout
        
        return batch
    
    def create_data_stream(self, dataloader):
        """Create optimized data stream with GPU preprocessing"""
        for data, target in dataloader:
            # Async transfer to GPU
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # GPU augmentation
            data = self.augment_batch(data)
            
            yield data, target

# Prefetch to GPU for maximum throughput
class DataPrefetcher:
    def __init__(self, loader, device='cuda'):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        """Preload next batch to GPU"""
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)
    
    def next(self):
        """Get next batch and start loading the following one"""
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        target = self.next_target
        
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        
        self.preload()
        return data, target

# Usage example
print("Data pipeline optimization strategies configured")`} />

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/data_pipeline_optimization.png"
              alt="Data Pipeline Optimization Flow"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </section>
      </Stack>
    </Container>
  );
};

export default PerformanceOptimization;