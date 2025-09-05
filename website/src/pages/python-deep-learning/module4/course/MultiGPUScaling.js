import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Badge, List, Grid, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';

const MultiGPUScaling = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <Title order={1}>Multi-GPU Scaling Strategies</Title>
        
        <Text>
          As models grow larger, single GPU training becomes insufficient. We'll explore different 
          parallelization strategies to scale training across multiple GPUs and nodes.
        </Text>

        
          <Title id="parallelization-overview" order={2} mt="xl">Parallelization Strategies Overview</Title>
          
          <Text>
            There are four main types of parallelism in distributed deep learning:
          </Text>

          <Paper p="md" withBorder>
            <Grid>
              <Grid.Col span={6}>
                <Badge color="blue" size="lg">Data Parallelism (DP)</Badge>
                <Text size="sm" mt="xs">
                  • Replicate model on each GPU<br/>
                  • Split batch across GPUs<br/>
                  • Synchronize gradients
                </Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Badge color="green" size="lg">Model Parallelism (MP)</Badge>
                <Text size="sm" mt="xs">
                  • Split model layers across GPUs<br/>
                  • Sequential computation<br/>
                  • For very large models
                </Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Badge color="orange" size="lg">Pipeline Parallelism (PP)</Badge>
                <Text size="sm" mt="xs">
                  • Split model into stages<br/>
                  • Micro-batching for efficiency<br/>
                  • Reduces idle time
                </Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Badge color="purple" size="lg">Tensor Parallelism (TP)</Badge>
                <Text size="sm" mt="xs">
                  • Split tensors across GPUs<br/>
                  • Parallel matrix operations<br/>
                  • For very wide layers
                </Text>
              </Grid.Col>
            </Grid>
          </Paper>

          <Title order={3} mt="lg">Choosing the Right Strategy</Title>

          <Table mt="md">
            <thead>
              <tr>
                <th>Model Size</th>
                <th>Batch Size</th>
                <th>Recommended Strategy</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Small (fits on 1 GPU)</td>
                <td>Large</td>
                <td>Data Parallel</td>
                <td>Simple, efficient</td>
              </tr>
              <tr>
                <td>Medium (fits on 1 GPU)</td>
                <td>Very Large</td>
                <td>Data Parallel + Gradient Accumulation</td>
                <td>Memory efficient</td>
              </tr>
              <tr>
                <td>Large (doesn't fit on 1 GPU)</td>
                <td>Any</td>
                <td>Model/Pipeline Parallel</td>
                <td>Necessary for memory</td>
              </tr>
              <tr>
                <td>Very Large</td>
                <td>Any</td>
                <td>3D Parallelism (DP+MP+PP)</td>
                <td>Maximum scaling</td>
              </tr>
            </tbody>
          </Table>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/parallelization_strategies.png"
              alt="Parallelization Strategies Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>

          <Title order={3} mt="lg">Memory and Compute Distribution</Title>

          <Paper p="md" withBorder>
            <Title order={4}>Resource Distribution Across Strategies</Title>
            
            <Text mt="md">For a model with P parameters, batch size B, and N GPUs:</Text>
            
            <Table mt="md">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Parameters per GPU</th>
                  <th>Activations per GPU</th>
                  <th>Communication</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Data Parallel</td>
                  <td><InlineMath>{`P`}</InlineMath></td>
                  <td><InlineMath>{`\\frac{B}{N} \\times A`}</InlineMath></td>
                  <td>All-reduce gradients</td>
                </tr>
                <tr>
                  <td>Model Parallel</td>
                  <td><InlineMath>{`\\frac{P}{N}`}</InlineMath></td>
                  <td><InlineMath>{`B \\times A`}</InlineMath></td>
                  <td>Activations between layers</td>
                </tr>
                <tr>
                  <td>Pipeline Parallel</td>
                  <td><InlineMath>{`\\frac{P}{N}`}</InlineMath></td>
                  <td><InlineMath>{`\\frac{B}{M} \\times A`}</InlineMath></td>
                  <td>Micro-batch activations</td>
                </tr>
              </tbody>
            </Table>
            
            <Text size="sm" c="dimmed" mt="sm">
              A = activations per sample, M = number of micro-batches
            </Text>
          </Paper>
        

        
          <Title id="data-parallel" order={2} mt="xl">Data Parallelism (DP/DDP)</Title>
          
          <Text>
            Data parallelism is the most common and straightforward parallelization strategy,
            perfect for models that fit on a single GPU.
          </Text>

          <Title order={3} mt="lg">DataParallel (DP) - Single Node</Title>

          <CodeBlock language="python" code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=1024, hidden_sizes=[2048, 1024], num_classes=10):
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

# Simple DataParallel (single machine, multiple GPUs)
def train_with_dp(model, dataloader, num_gpus=2):
    if torch.cuda.device_count() < num_gpus:
        print(f"Warning: Only {torch.cuda.device_count()} GPUs available")
    
    # Wrap model with DataParallel
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(dataloader):
            # DataParallel automatically splits batch across GPUs
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)  # Parallel forward
            loss = F.cross_entropy(output, target)
            loss.backward()  # Parallel backward
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# Note: DataParallel has limitations:
# - Imbalanced GPU memory (GPU 0 stores more)
# - Single-process, GIL-limited
# - Not suitable for multi-node training`} />

          <Title order={3} mt="lg">DistributedDataParallel (DDP) - Multi-Node</Title>

          <Alert color="green" mt="md">
            DDP is recommended over DP for better performance and scalability.
            It uses multiple processes, avoiding Python GIL limitations.
          </Alert>

          <CodeBlock language="python" code={`import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up DDP process group"""
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, dataset):
    """Training function for each DDP process"""
    # Setup
    setup_ddp(rank, world_size)
    
    # Create model and move to GPU
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Distributed sampler ensures each GPU gets different data
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    for epoch in range(5):
        sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # DDP automatically synchronizes gradients across GPUs
            optimizer.step()
            
            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    cleanup_ddp()

# Launch DDP training (typically done with torchrun or torch.multiprocessing)
import torch.multiprocessing as mp

def launch_ddp_training():
    world_size = torch.cuda.device_count()
    model = MLP()
    dataset = YourDataset()  # Your dataset here
    
    # Spawn processes for each GPU
    mp.spawn(
        train_ddp,
        args=(world_size, model, dataset),
        nprocs=world_size,
        join=True
    )

# Command line launch:
# torchrun --nproc_per_node=4 your_script.py
# or
# python -m torch.distributed.launch --nproc_per_node=4 your_script.py`} />

          <Title order={3} mt="lg">Gradient Synchronization</Title>

          <Paper p="md" withBorder>
            <Title order={4}>All-Reduce Algorithm</Title>
            <Text>
              DDP uses all-reduce to efficiently synchronize gradients:
            </Text>
            <BlockMath>{`\\nabla_{\\text{avg}} = \\frac{1}{N} \\sum_{i=1}^{N} \\nabla_i`}</BlockMath>
            
            <Text mt="md">Communication cost:</Text>
            <BlockMath>{`\\text{Time} = \\alpha + \\beta \\times \\frac{2(N-1)}{N} \\times M`}</BlockMath>
            
            <Text size="sm" c="dimmed">
              α = latency, β = inverse bandwidth, M = message size, N = number of GPUs
            </Text>
          </Paper>

          <CodeBlock language="python" code={`# Custom gradient synchronization for understanding
class ManualGradientSync:
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
        
    def sync_gradients(self):
        """Manually synchronize gradients across GPUs"""
        for param in self.model.parameters():
            if param.grad is not None:
                # All-reduce gradient across all processes
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                # Average gradients
                param.grad /= self.world_size
    
    def benchmark_communication(self, size_mb=100):
        """Benchmark gradient synchronization time"""
        import time
        
        # Create dummy tensor
        tensor = torch.randn(int(size_mb * 1024 * 1024 / 4)).cuda()
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(100):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        print(f"All-reduce {size_mb}MB across {self.world_size} GPUs:")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Per iteration: {elapsed/100*1000:.2f}ms")
        print(f"  Bandwidth: {size_mb * 100 / elapsed:.1f} MB/s")`} />
        

        
          <Title id="model-parallel" order={2} mt="xl">Model & Pipeline Parallelism</Title>
          
          <Text>
            When models don't fit on a single GPU, we need to split them across devices.
          </Text>

          <Title order={3}>Model Parallelism</Title>

          <CodeBlock language="python" code={`# Simple model parallelism example
class ModelParallelMLP(nn.Module):
    def __init__(self, input_size=1024, hidden_size=4096, num_classes=10):
        super().__init__()
        
        # Split model across two GPUs
        # First half on GPU 0
        self.layer1 = nn.Linear(input_size, hidden_size).to('cuda:0')
        self.relu1 = nn.ReLU().to('cuda:0')
        
        # Second half on GPU 1
        self.layer2 = nn.Linear(hidden_size, hidden_size).to('cuda:1')
        self.relu2 = nn.ReLU().to('cuda:1')
        self.layer3 = nn.Linear(hidden_size, num_classes).to('cuda:1')
    
    def forward(self, x):
        # Start on GPU 0
        x = x.to('cuda:0')
        x = self.relu1(self.layer1(x))
        
        # Transfer to GPU 1
        x = x.to('cuda:1')
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        
        return x

# Pipeline parallelism for better efficiency
from torch.distributed.pipeline.sync import Pipe

class PipelineParallelMLP(nn.Module):
    def __init__(self, input_size=1024, hidden_size=4096, num_classes=10):
        super().__init__()
        
        # Define sequential layers
        layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Split into pipeline stages
        # balance parameter determines how to split layers
        self.model = Pipe(
            layers,
            balance=[2, 2, 2, 1],  # Layers per GPU
            devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
            chunks=8  # Number of micro-batches
        )
    
    def forward(self, x):
        return self.model(x)

# Memory calculation for model parallelism
def calculate_mp_memory(model_params, num_gpus, strategy='model'):
    """Calculate memory per GPU for different parallelism strategies"""
    
    if strategy == 'data':
        params_per_gpu = model_params
        print(f"Data Parallel: {params_per_gpu:,} params per GPU")
        
    elif strategy == 'model':
        params_per_gpu = model_params // num_gpus
        print(f"Model Parallel: {params_per_gpu:,} params per GPU")
        
    elif strategy == 'pipeline':
        params_per_gpu = model_params // num_gpus
        print(f"Pipeline Parallel: {params_per_gpu:,} params per GPU")
        print(f"  Note: Activations from micro-batches add overhead")
    
    # Memory in MB (assuming FP32)
    memory_mb = params_per_gpu * 4 / (1024**2)
    
    # With optimizer (Adam)
    memory_with_opt = memory_mb * 4  # Params + grads + 2x optimizer states
    
    print(f"  Base memory: {memory_mb:.2f} MB")
    print(f"  With Adam optimizer: {memory_with_opt:.2f} MB")
    
    return params_per_gpu

# Example: 7B parameter model
model_params = 7_000_000_000
num_gpus = 8

print("Memory requirements for 7B parameter model:")
for strategy in ['data', 'model', 'pipeline']:
    calculate_mp_memory(model_params, num_gpus, strategy)
    print()`} />

          <Title order={3} mt="lg">Pipeline Parallelism Scheduling</Title>

          <Paper p="md" withBorder>
            <Title order={4}>Pipeline Bubble Time</Title>
            <Text>
              Pipeline parallelism introduces bubble time (idle GPUs).
              With M micro-batches and P pipeline stages:
            </Text>
            
            <BlockMath>{`\\text{Bubble Fraction} = \\frac{P - 1}{M}`}</BlockMath>
            
            <Text mt="sm">
              To minimize bubble overhead, use M ≫ P (many micro-batches)
            </Text>
          </Paper>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/pipeline_parallel_schedule.png"
              alt="Pipeline Parallelism Schedule"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        

        
          <Title id="distributed-training" order={2} mt="xl">Distributed Training Best Practices</Title>
          
          <Text>
            Effective distributed training requires careful consideration of communication patterns,
            batch sizes, and learning rate scaling.
          </Text>

          <Paper p="md" withBorder>
            <Title order={4}>Key Best Practices</Title>
            <List spacing="sm" mt="md">
              <List.Item>
                <strong>Linear learning rate scaling:</strong> LR = base_lr × num_gpus
              </List.Item>
              <List.Item>
                <strong>Warm-up period:</strong> Gradually increase LR over first epochs
              </List.Item>
              <List.Item>
                <strong>Gradient clipping:</strong> Essential for stability with large batches
              </List.Item>
              <List.Item>
                <strong>Mixed precision:</strong> Reduces communication overhead
              </List.Item>
              <List.Item>
                <strong>Gradient accumulation:</strong> Simulate larger batches
              </List.Item>
            </List>
          </Paper>

          <CodeBlock language="python" code={`class DistributedTrainer:
    def __init__(self, model, world_size, base_lr=0.001, warmup_epochs=5):
        self.model = model
        self.world_size = world_size
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        
        # Scale learning rate
        self.lr = base_lr * world_size
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
    def get_lr(self, epoch):
        """Learning rate schedule with warm-up"""
        if epoch < self.warmup_epochs:
            # Linear warm-up
            return self.lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing after warm-up
            progress = (epoch - self.warmup_epochs) / (100 - self.warmup_epochs)
            return self.lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch with distributed best practices"""
        self.model.train()
        
        # Adjust learning rate
        current_lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = F.cross_entropy(output, target)
            
            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Logging (only on rank 0)
            if batch_idx % 50 == 0 and dist.get_rank() == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
    
    def benchmark_training(self, input_shape, num_iterations=100):
        """Benchmark distributed training performance"""
        import time
        
        dummy_data = torch.randn(input_shape).cuda()
        dummy_target = torch.randint(0, 10, (input_shape[0],)).cuda()
        
        # Warmup
        for _ in range(10):
            output = self.model(dummy_data)
            loss = F.cross_entropy(output, dummy_target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iterations):
            output = self.model(dummy_data)
            loss = F.cross_entropy(output, dummy_target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate throughput
        samples_per_sec = (input_shape[0] * num_iterations) / elapsed
        
        # All-reduce to get total throughput
        throughput_tensor = torch.tensor([samples_per_sec]).cuda()
        dist.all_reduce(throughput_tensor, op=dist.ReduceOp.SUM)
        
        if dist.get_rank() == 0:
            total_throughput = throughput_tensor.item()
            print(f"\\nDistributed Training Performance:")
            print(f"  World size: {self.world_size}")
            print(f"  Per-GPU throughput: {samples_per_sec:.1f} samples/s")
            print(f"  Total throughput: {total_throughput:.1f} samples/s")
            print(f"  Scaling efficiency: {total_throughput / (samples_per_sec * self.world_size) * 100:.1f}%")

# Example launch script
def main():
    # This would be called by each process in distributed training
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Setup distributed training
    setup_ddp(rank, world_size)
    
    # Create model and trainer
    model = DDP(MLP().to(rank), device_ids=[rank])
    trainer = DistributedTrainer(model, world_size)
    
    # Create distributed dataloader
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=256, sampler=sampler)
    
    # Train
    for epoch in range(100):
        trainer.train_epoch(dataloader, epoch)
    
    # Cleanup
    cleanup_ddp()

if __name__ == "__main__":
    # Launch with: torchrun --nproc_per_node=8 script.py
    main()`} />

          <Title order={3} mt="lg">Advanced Scaling Strategies</Title>

          <Paper p="md" withBorder>
            <Title order={4}>3D Parallelism (DP + MP + PP)</Title>
            <Text>
              For maximum scale, combine all parallelism strategies:
            </Text>
            
            <List size="sm" mt="md">
              <List.Item>Data parallel across nodes</List.Item>
              <List.Item>Pipeline parallel within nodes</List.Item>
              <List.Item>Tensor parallel within layers</List.Item>
            </List>
            
            <Text mt="md">Total GPUs = DP_size × PP_size × TP_size</Text>
          </Paper>

          <Alert title="Scaling Checklist" color="green" mt="md">
            <List size="sm">
              <List.Item>✓ Profile single-GPU performance first</List.Item>
              <List.Item>✓ Choose appropriate parallelism strategy based on model size</List.Item>
              <List.Item>✓ Optimize batch size for GPU memory utilization</List.Item>
              <List.Item>✓ Use mixed precision to reduce communication</List.Item>
              <List.Item>✓ Implement gradient accumulation for large effective batches</List.Item>
              <List.Item>✓ Monitor scaling efficiency and communication overhead</List.Item>
              <List.Item>✓ Use NCCL backend for GPU communication</List.Item>
              <List.Item>✓ Enable CUDA graphs for small models</List.Item>
            </List>
          </Alert>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/scaling_efficiency.png"
              alt="Scaling Efficiency Chart"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        
      </Stack>
    </Container>
  );
};

export default MultiGPUScaling;