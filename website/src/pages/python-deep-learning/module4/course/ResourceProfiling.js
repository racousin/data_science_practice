import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Group, Badge, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';

const ResourceProfiling = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <Title order={1}>Model Resource Profiling</Title>
        
        <Text>
          Understanding how deep learning models utilize system resources is crucial for optimization.
          We'll explore how different model components consume memory and compute resources.
        </Text>
We will use this simple MLP model to illustrate following sections
          <CodeBlock language="python" code={`class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x`}/>
        
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
                </List>
              </div>
              <div>
                <Badge color="green" size="lg">GPU</Badge>
                <List size="sm" mt="sm">
                  <List.Item>High throughput for parallel ops</List.Item>
                  <List.Item>Excellent for matrix operations</List.Item>
                  <List.Item>Limited memory (typically 8-80GB)</List.Item>
                </List>
              </div>
            </Group>

          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/cpugpubatch.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
                      <Title order={4}>Batch Size Impact on training</Title>
                      
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

          <Title order={3} mt="lg">Memory Transfer Overhead</Title>
          
          <Text>
            As we discussed in Module 1, your computer's operating system kernel runs on the CPU with its RAM. 
            All data must pass through CPU RAM before reaching the GPU for processing. This CPU-to-GPU memory 
            transfer can become a significant bottleneck since data needs to travel across the PCIe bus between 
            these two separate memory spaces. Understanding and optimizing these transfers is crucial for performance.
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
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/overhead.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
            <List size="sm">
              <List.Item>Use GPU for large batch training and inference</List.Item>
              <List.Item>Minimize CPU-GPU transfers by keeping data on GPU</List.Item>
              <List.Item>Profile your specific workload to find optimal device</List.Item>
            </List>

        
        
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



        <Title order={2}>Model Complexity Estimation</Title>
        
        <Text>
          Understanding the computational complexity of neural networks 
          is essential for designing efficient models and choosing appropriate hardware.
        </Text>

        <Title order={3}>Number of Operations Estimation (FLOPs)</Title>
        
        <Text>
          FLOPs (Floating Point Operations) provide a hardware-independent measure of computational complexity.
          We count multiply-add operations as 2 FLOPs.
        </Text>

        <Title order={4}>Number of Operations for Forward Pass</Title>
        
        <Text mt="md">
          <strong>Single Neuron:</strong> For input size n and single output:
        </Text>
        <CodeBlock language="python" code={`# Matrix multiplication: y = Wx + b
# Operations: n multiplications + (n-1) additions + 1 bias addition
FLOPs = 2n`} />
        
        <Text mt="md">
          <strong>Activation Functions:</strong>
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item>ReLU: 1 FLOP per element (comparison)</List.Item>
          <List.Item>Sigmoid/Tanh: ~10-20 FLOPs per element (exponential operations)</List.Item>
          <List.Item>Softmax: ~5m FLOPs for m outputs (exp + normalization)</List.Item>
        </List>
        
        <Text mt="md">
          <strong>MLP Layer:</strong> For input size n, output size m:
        </Text>
        <CodeBlock language="python" code={`# Linear layer: Y = XW + b
# X: batch_size × n, W: n × m, b: m
FLOPs_per_sample = 2nm  # Matrix multiplication
FLOPs_batch = batch_size × 2nm`} />

        <Title order={4} mt="lg">Number of Operations for Backward Pass</Title>
        
        <Text mt="md">
          The backward pass typically requires 2-3× the FLOPs of the forward pass:
        </Text>
        
        <List mt="md" spacing="sm">
          <List.Item>
            <strong>Gradient w.r.t weights:</strong> dL/dW = X^T × dL/dY
            <CodeBlock language="python" code={`FLOPs = 2nm (same as forward)`} />
          </List.Item>
          <List.Item>
            <strong>Gradient w.r.t inputs:</strong> dL/dX = dL/dY × W^T
            <CodeBlock language="python" code={`FLOPs = 2nm (same as forward)`} />
          </List.Item>
          <List.Item>
            <strong>Gradient w.r.t bias:</strong> dL/db = sum(dL/dY)
            <CodeBlock language="python" code={`FLOPs = m`} />
          </List.Item>
        </List>
        
        <Text mt="md">
          <strong>Total backward FLOPs ≈ 2 × forward FLOPs</strong>
        </Text>

        <Title order={4} mt="lg">Number of Operations for Optimizer</Title>
        
        <Text mt="md">
          Optimizer FLOPs depend on the algorithm complexity:
        </Text>
        
        <List mt="md" spacing="sm">
          <List.Item>
            <strong>SGD:</strong> W = W - lr × dW
            <CodeBlock language="python" code={`FLOPs = 2 × num_parameters`} />
          </List.Item>
          <List.Item>
            <strong>SGD with Momentum:</strong> v = β×v + dW; W = W - lr×v
            <CodeBlock language="python" code={`FLOPs = 4 × num_parameters`} />
          </List.Item>
          <List.Item>
            <strong>Adam:</strong> Updates both first and second moments
            <CodeBlock language="python" code={`FLOPs = 8 × num_parameters`} />
          </List.Item>
        </List>
        
        <Text mt="md">
          <strong>Example: ResNet-50 Complexity</strong>
        </Text>
        <CodeBlock language="python" code={`# ResNet-50 statistics
Parameters: 25.6M
Forward pass: 3.8 GFLOPs
Backward pass: ~7.6 GFLOPs
Adam update: ~0.2 GFLOPs
Total per iteration: ~11.6 GFLOPs`} />

<Title order={2} mt="xl">Memory and Computation Profiling with PyTorch Profiler</Title>

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
<CodeBlock language="python" code={`-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm         4.52%     211.996us        77.31%       3.627ms       1.814ms      26.432us        90.57%      40.544us      20.272us           0 B           0 B      33.50 KB      33.50 KB             2  
                                        aten::clamp_min         0.71%      33.130us         1.00%      47.066us      47.066us       2.752us         9.43%       2.752us       2.752us           0 B           0 B      32.00 KB      32.00 KB             1  
                                           aten::linear         7.82%     366.919us        98.21%       4.608ms       2.304ms       0.000us         0.00%      40.544us      20.272us           0 B           0 B      33.50 KB           0 B             2  
                                                aten::t         0.80%      37.622us        13.08%     613.940us     306.970us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             2  
                                        aten::transpose        12.03%     564.421us        12.28%     576.318us     288.159us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             2  
                                       aten::as_strided         0.25%      11.897us         0.25%      11.897us       5.948us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             2  
                                           Unrecognized        49.10%       2.304ms        49.10%       2.304ms       2.304ms      14.112us        48.36%      14.112us      14.112us           0 B           0 B           0 B           0 B             1  
          cudaOccupancyMaxActiveBlocksPerMultiprocessor        22.25%       1.044ms        22.25%       1.044ms       1.044ms       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             1  
                                       cudaLaunchKernel         1.74%      81.598us         1.74%      81.598us      20.399us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             4  
                        ampere_sgemm_32x32_sliced1x4_tn         0.00%       0.000us         0.00%       0.000us       0.000us      21.824us        74.78%      21.824us      10.912us           0 B           0 B           0 B           0 B             2  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.692ms
Self CUDA time total: 29.184us`}/>

          <Text mt="md" size="sm">
            <strong>Column explanations:</strong>
          </Text>
          <List size="sm" mt="xs">
            <List.Item><strong>Name:</strong> Operation name (e.g., aten::linear for linear layers, aten::addmm for matrix multiply-add)</List.Item>
            <List.Item><strong>Self CPU %/Self CPU:</strong> Time spent in this operation alone on CPU (excluding child operations)</List.Item>
            <List.Item><strong>CPU total %/CPU total:</strong> Total CPU time including all child operations</List.Item>
            <List.Item><strong>CPU time avg:</strong> Average CPU time per call</List.Item>
            <List.Item><strong>Self CUDA/Self CUDA %:</strong> Time spent in this operation alone on GPU</List.Item>
            <List.Item><strong>CUDA total/CUDA time avg:</strong> Total GPU time including child operations</List.Item>
            <List.Item><strong>CPU Mem/Self CPU Mem:</strong> Total and self CPU memory allocated</List.Item>
            <List.Item><strong>CUDA Mem/Self CUDA Mem:</strong> Total and self GPU memory allocated</List.Item>
            <List.Item><strong># of Calls:</strong> Number of times this operation was executed</List.Item>
          </List>


          <Title order={3} mt="lg">Detailed Operation Profiling</Title>
          
          <Text>
            Use record_function to annotate and track specific code sections:
          </Text>

          <CodeBlock language="python" code={`def profile_training_components(model, dataloader, optimizer):
    """Profile different components of training loop"""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 6:  # Profile 4 batches
                break
            if batch_idx < 2: # 2 warmup to avoid initial noise
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
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
    <CodeBlock code={`Time breakdown:
  data_transfer: 7.7%
  forward: 12.5%
  loss: 6.3%
  backward: 30.2%
  optimizer_step: 43.2%`}/>

          <Text mt="md" size="sm">
            <strong>Profiling insights:</strong>
          </Text>
          <List size="sm" mt="xs">
            <List.Item><strong>Optimizer step (43.2%):</strong> Largest bottleneck - gradient updates and parameter optimization dominate training time</List.Item>
            <List.Item><strong>Backward pass (30.2%):</strong> Second most expensive - gradient computation through the network</List.Item>
            <List.Item><strong>Forward pass (12.5%):</strong> Relatively efficient compared to backward pass</List.Item>
            <List.Item><strong>Data transfer (7.7%):</strong> CPU to GPU transfer is minimal - good data pipeline efficiency</List.Item>
            <List.Item><strong>Loss computation (6.3%):</strong> Negligible overhead from loss calculation</List.Item>
          </List>
          <Text size="sm" mt="xs">
            This profile suggests optimization efforts should focus on the optimizer (e.g., using fused optimizers) and backward pass efficiency.
          </Text>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/time.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
          <CodeBlock language="python" code={`def profile_model_flops(model, input_shape):
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
    
    print("\nTop operations by FLOPs:")
    for evt in events:
        print(f"  {evt.name}: {evt.flops:,} FLOPs")
    
    return flops`} />
                    <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/flop.png"
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