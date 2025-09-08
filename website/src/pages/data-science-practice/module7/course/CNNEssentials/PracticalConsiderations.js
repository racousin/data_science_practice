import React from 'react';
import { Text, Stack, List, Grid, Table, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';
import { AlertCircle } from 'lucide-react';

const PracticalConsiderations = () => {
  const memoryManagementCode = `
class MemoryEfficientCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # inplace operation saves memory
            nn.MaxPool2d(2)
        ])
    
    def forward(self, x):
        # Clear unnecessary buffers
        torch.cuda.empty_cache()
        
        # Use gradient checkpointing for memory efficiency
        with torch.cuda.amp.autocast():  # Mixed precision training
            for layer in self.features:
                x = layer(x)
                # Free memory after each layer if needed
                torch.cuda.empty_cache()
        
        return x

def train_with_efficiency(model, train_loader, epochs=10):
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for batch in train_loader:
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than .zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                output = model(batch)
                loss = criterion(output)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()`;

  const performanceOptimizationCode = `
import torch.nn.functional as F
from torch.profiler import profile, record_function

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Use efficient channel numbers (multiples of 8/16)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Use fused operations where possible
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling instead of multiple FC layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Profile different sections
        with record_function("initial_conv"):
            x = F.relu(self.bn1(self.conv1(x)))
        
        with record_function("main_features"):
            x = self.conv_bn_relu(x)
        
        with record_function("classifier"):
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        
        return x

# Performance monitoring
def profile_model(model, input_shape=(1, 3, 224, 224)):
    input_tensor = torch.randn(input_shape).cuda()
    model = model.cuda()
    
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        model(input_tensor)
    
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10))`;

const hardwareUtilizationCode = `
def configure_training(model, batch_size, num_gpus):
    if num_gpus > 1:
        # Multi-GPU setup with DistributedDataParallel
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    # Configure DataLoader for efficient data loading
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4 * num_gpus,  # Scale workers with GPUs
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True  # Keep workers alive
    )
    
    return model, train_loader

def optimize_memory_usage(model, input_shape):
    # Calculate theoretical memory usage
    batch_size, channels, height, width = input_shape
    
    # Feature map memory
    feature_map_size = batch_size * channels * height * width * 4  # 4 bytes per float
    
    # Model parameters memory
    param_memory = sum(p.numel() * 4 for p in model.parameters())
    
    # Gradient memory
    gradient_memory = param_memory
    
    total_memory = feature_map_size + param_memory + gradient_memory
    
    return {
        'feature_maps': feature_map_size / (1024**2),  # MB
        'parameters': param_memory / (1024**2),
        'gradients': gradient_memory / (1024**2),
        'total': total_memory / (1024**2)
    }`;

  const trainingOptimizationCode = `
class TrainingOptimizer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize learning rate finder
        self.lr_finder = LRFinder(model, optimizer, criterion)
    
    def find_optimal_lr(self):
        self.lr_finder.range_test(
            self.train_loader,
            end_lr=10,
            num_iter=100,
            step_mode='exp'
        )
        return self.lr_finder.suggestion()
    
    def configure_training(self):
        # Set up automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Set up learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimal_lr,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_loader)
        )
    
    def train_epoch(self):
        self.model.train()
        for batch in self.train_loader:
            with torch.cuda.amp.autocast():
                loss = self.training_step(batch)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            return self.validation_step()`;

  return (
    <Stack spacing="md">
      <Text>
        Practical considerations in CNN implementation significantly impact model
        performance, training efficiency, and deployment success. This section covers
        key aspects to consider when developing CNN-based solutions.
      </Text>

      <Text weight={700}>1. Memory Management</Text>

      <Alert icon={<AlertCircle />} color="blue">
        Efficient memory management is crucial for training deep CNNs, especially
        with limited GPU resources.
      </Alert>

      <CodeBlock
        language="python"
        code={memoryManagementCode}
      />

      <Text weight={700}>2. Performance Optimization</Text>

      <Text>
        Optimize model architecture and operations for better performance:
      </Text>

      <CodeBlock
        language="python"
        code={performanceOptimizationCode}
      />

      <Text weight={700}>3. Hardware Utilization</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text>
            Memory requirements for a CNN layer:
          </Text>
          <BlockMath>
            {`M_{layer} = (C_{in} × K^2 × C_{out}) + (H × W × C_{out})`}
          </BlockMath>
        </Grid.Col>

        <Grid.Col span={12} md={6}>
          <Text>
            Total FLOPS for convolution:
          </Text>
          <BlockMath>
            {`FLOPS = 2 × H × W × C_{in} × K^2 × C_{out}`}
          </BlockMath>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={hardwareUtilizationCode}
      />

      <Text weight={700}>4. Training Optimization</Text>

      <CodeBlock
        language="python"
        code={trainingOptimizationCode}
      />

      <Text weight={700}>5. Common Bottlenecks and Solutions</Text>

      <Table>
        <thead>
          <tr>
            <th>Bottleneck</th>
            <th>Impact</th>
            <th>Solution</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Memory Usage</td>
            <td>OOM errors, slow training</td>
            <td>Gradient checkpointing, mixed precision</td>
          </tr>
          <tr>
            <td>Data Loading</td>
            <td>GPU underutilization</td>
            <td>Proper num_workers, pin_memory</td>
          </tr>
          <tr>
            <td>CPU Bottleneck</td>
            <td>Slow preprocessing</td>
            <td>GPU preprocessing, efficient transforms</td>
          </tr>
          <tr>
            <td>GPU Utilization</td>
            <td>Inefficient compute usage</td>
            <td>Optimal batch size, model parallelism</td>
          </tr>
        </tbody>
      </Table>

      <Text weight={700}>6. Best Practices</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text weight={600}>Architecture Design:</Text>
          <List>
            <List.Item>Use power-of-2 channel numbers</List.Item>
            <List.Item>Balance depth vs. width</List.Item>
            <List.Item>Consider inference requirements</List.Item>
            <List.Item>Profile memory usage patterns</List.Item>
          </List>
        </Grid.Col>

        <Grid.Col span={12} md={6}>
          <Text weight={600}>Training Setup:</Text>
          <List>
            <List.Item>Use learning rate finder</List.Item>
            <List.Item>Implement gradient clipping</List.Item>
            <List.Item>Monitor training metrics</List.Item>
            <List.Item>Regular validation checks</List.Item>
          </List>
        </Grid.Col>
      </Grid>

      <Text weight={700}>7. Deployment Considerations</Text>

      <List>
        <List.Item>
          <strong>Model Optimization:</strong>
          <List withPadding>
            <List.Item>Quantization for reduced memory</List.Item>
            <List.Item>Pruning for faster inference</List.Item>
            <List.Item>Architecture distillation</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Hardware Adaptation:</strong>
          <List withPadding>
            <List.Item>Target platform requirements</List.Item>
            <List.Item>Memory constraints</List.Item>
            <List.Item>Inference time requirements</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Monitoring and Maintenance:</strong>
          <List withPadding>
            <List.Item>Performance metrics tracking</List.Item>
            <List.Item>Resource utilization</List.Item>
            <List.Item>Model versioning</List.Item>
          </List>
        </List.Item>
      </List>

      <Text weight={700}>8. Testing and Validation</Text>

      <List>
        <List.Item>Unit tests for model components</List.Item>
        <List.Item>Integration tests for full pipeline</List.Item>
        <List.Item>Performance benchmarking</List.Item>
        <List.Item>Resource utilization monitoring</List.Item>
      </List>

    </Stack>
  );
};

export default PracticalConsiderations;