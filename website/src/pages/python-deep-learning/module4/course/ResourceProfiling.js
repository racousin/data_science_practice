import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Group, Badge, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from '../../../../components/CodeBlock';

const ResourceProfiling = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <div data-slide>
        <Title order={1}>Model Resource Profiling</Title>
        
        <Text>
          We'll explore how different model components consume memory and compute resources.
        </Text>
        </div>
<div data-slide>
          <Title id="gpu-vs-cpu" order={2} mt="xl">GPU vs CPU Performance</Title>
          

            <Title order={4}>Performance Characteristics</Title>
            <Group justify="center" mt="md">
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
</div>
<div data-slide>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/cpugpubatch.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
          </div>
          <div data-slide>
          <Title order={3} mt="lg">Memory Transfer Overhead</Title>
          
          <Text>
            As we discussed in Module 1, your computer's operating system kernel runs on the CPU with its RAM. 
            All data must pass through CPU RAM before reaching the GPU for processing. This CPU-to-GPU memory 
            transfer can become a significant bottleneck.
          </Text>
      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module1/gpuworkflow.jpg"
          alt="Matrix Multiplication Parallelization"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
        <Text component="p" ta="center" mt="xs">
          Source: https://www.cse.iitm.ac.in/
        </Text>
      </Flex>
    </div>
    <div data-slide>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/overhead.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
</div>
        <div data-slide>
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
</div>
<div data-slide>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/distribution.png"
              alt="Model Memory Components Visualization"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>

        
          <div data-slide>
            <Title id="memory-breakdown" order={2}>Memory Formula for Training</Title>
            
            <Text>
              During training, the total memory consumption is the sum of all components:
            </Text>
            
            <BlockMath>
              {`\\text{Total Memory} = \\text{Parameters} + \\text{Gradients} + \\text{Optimizer States} + \\text{Activations} +(\\text{Computation Graph})`}
            </BlockMath>
</div>
<div data-slide>
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
            </div>
<div data-slide>
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
            </div>
            <div data-slide>
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
            </div>
            <div data-slide>
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
</div>

        <div data-slide>
          <Title order={3} mt="lg">Graph Memory</Title>
          
          <Text>
            During training, PyTorch builds a computational graph to track operations for automatic differentiation.
            This graph stores the relationships between tensors and operations needed for backpropagation.
          </Text>

          <Text mt="md">Key components of graph memory:</Text>
          <List size="sm" mt="sm">
            <List.Item><strong>Node metadata:</strong> Information about each operation in the graph</List.Item>
            <List.Item><strong>Edge connections:</strong> Links between tensors and operations</List.Item>
            <List.Item><strong>Gradient functions:</strong> Pointers to functions that compute gradients during backward pass</List.Item>
            <List.Item><strong>Tensor references:</strong> Pointers to saved activations and gradients (not the data itself)</List.Item>
          </List>

          <Text mt="md">
            Graph nodes contain mostly pointers, not actual tensor data. Each layer typically creates 3-10 nodes 
            depending on complexity, with each node requiring approximately 300 bytes for metadata and pointer references.
          </Text>

          <BlockMath>
            {`\\text{Graph Memory} = \\text{nodes} \\times (\\text{metadata\\_size} + \\text{pointer\\_references})`}
          </BlockMath>

          <Text mt="sm">
            <strong>Example:</strong> GPT-3 with 96 layers:
          </Text>
          <List size="xm" mt="sm">
            <List.Item>Nodes per layer: ~8 (average)</List.Item>
            <List.Item>Memory per node: ~300 bytes</List.Item>
            <List.Item>Total graph memory: 96 × 8 × 300 ≈ 230KB</List.Item>
            <List.Item>Compare to model size: 175B parameters = 700GB in float32</List.Item>
          </List>

          <Text mt="md" size="sm">
            <strong>Key insight:</strong> Graph memory is typically only 0.1-1% of total memory usage. 
            The graph stores pointers to tensors, not the tensors themselves!
          </Text>
        </div>

<div data-slide>
        <Title order={2}>Model Complexity Estimation</Title>
        

        </div>
<div data-slide>
        <Title order={4}>Number of Operations for Forward Pass</Title>
        
        <Text mt="md">
          <strong>Single Neuron:</strong> For input size n and single output:
        </Text>
        <CodeBlock language="python" code={`# Vector multiplication: y = Wx + b
# Operations: n multiplications + (n-1) additions + 1 bias addition
FLOPs = 2n`} />
        
        <Text mt="md">
          <strong>Activation Functions:</strong>
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item>ReLU: 1 FLOP per element (comparison)</List.Item>
          <List.Item>Sigmoid/Tanh: ~10-20 FLOPs per element (exponential operations)</List.Item>
        </List>
        
        <Text mt="md">
          <strong>MLP Layer:</strong> For input size n, output size m:
        </Text>
        <CodeBlock language="python" code={`# Linear layer: Y = XW + b
# X: batch_size × n, W: n × m, b: m
FLOPs_per_sample = 2nm  # Matrix multiplication per sample
FLOPs_batch = batch_size × 2nm  # Total FLOPs scale with batch size`} />
</div>
<div data-slide>
        <Title order={4} mt="lg">Number of Operations for Backward Pass</Title>
        
        <Text mt="md">
          During backpropagation, we need to compute two things:
        </Text>
        
        <Text mt="md">
          For a layer with input size n and output size m:
        </Text>
        
        <List mt="md" spacing="sm">
          <List.Item>
            <strong>Weight gradients (for updating weights):</strong> We calculate how much each weight contributed to the loss
            <CodeBlock language="python" code={`# dL/dW = X^T × dL/dY
FLOPs_per_sample = 2nm
FLOPs_batch = batch_size × 2nm  # Scales with batch size`} />
          </List.Item>
          <List.Item>
            <strong>Gradient propagation (for previous layers):</strong> We calculate the gradient to send back to earlier layers in the network
            <CodeBlock language="python" code={`# dL/dX = dL/dY × W^T  
FLOPs_per_sample = 2nm
FLOPs_batch = batch_size × 2nm  # Scales with batch size`} />
          </List.Item>
        </List>
        
        <Text mt="md">
          <strong>Key insight:</strong> The backward pass takes about 2× the computations of the forward pass. We compute weight gradients to update our parameters, and we also compute the gradient signal to send backward through the network (in chain rule).
        </Text>
</div>
<div data-slide>
        <Title order={4} mt="lg">Number of Operations for Optimizer</Title>
        
        <Text mt="md">
          Optimizer FLOPs depend on the algorithm complexity:
        </Text>
        
        <List mt="md" spacing="sm">
          <List.Item>
            <strong>SGD:</strong> W = W - lr × dW
            <CodeBlock language="python" code={`FLOPs = 2 × num_parameters  # Independent of batch size!`} />
          </List.Item>
          <List.Item>
            <strong>SGD with Momentum:</strong> v = β×v + dW; W = W - lr×v
            <CodeBlock language="python" code={`FLOPs = 4 × num_parameters  # Independent of batch size!`} />
          </List.Item>
          <List.Item>
            <strong>Adam:</strong> Updates both first and second moments
            <CodeBlock language="python" code={`FLOPs = 8 × num_parameters  # Independent of batch size!`} />
          </List.Item>
        </List>

        </div>
        <div data-slide>
        <Text mt="md">
          <strong>Example: ResNet-50 Complexity</strong>
        </Text>
        <CodeBlock language="python" code={`# ResNet-50 statistics
Parameters: 25.6M
Forward pass: 3.8 GFLOPs
Backward pass: ~7.6 GFLOPs
Adam update: ~0.2 GFLOPs
Total per iteration: ~11.6 GFLOPs`} />
</div>
<div data-slide>
<Title order={2} mt="xl">Memory and Computation Profiling with PyTorch Profiler</Title>

          <Text>
            PyTorch provides a powerful built-in profiler that can track CPU, GPU, and memory usage with minimal overhead.
          </Text>
</div>
<div data-slide>
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
<CodeBlock language="python" code={`---------------------------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %   CPU total %  Self CUDA %      CUDA Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  
                     aten::linear         7.82%        98.21%        0.00%      33.50 KB             2  
                      aten::addmm         4.52%        77.31%       90.57%      33.50 KB             2  
                  aten::transpose        12.03%        12.28%        0.00%           0 B             2  
                  aten::clamp_min         0.71%         1.00%        9.43%      32.00 KB             1  
                     Unrecognized        49.10%        49.10%       48.36%           0 B             1  
cudaOccupancyMaxActiveBlocksPer..        22.25%        22.25%        0.00%           0 B             1  
---------------------------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.692ms
Self CUDA time total: 29.184us`}/>
</div>
<div data-slide>
          <Text mt="md" size="sm">
            <strong>Key columns:</strong>
          </Text>
          <List size="sm" mt="xs">
            <List.Item><strong>Name:</strong> Operation name (e.g., aten::linear for linear layers, aten::addmm for matrix multiply-add)</List.Item>
            <List.Item><strong>Self CPU %:</strong> Time spent in this operation alone on CPU (excluding child operations)</List.Item>
            <List.Item><strong>CPU total %:</strong> Total CPU time including all child operations</List.Item>
            <List.Item><strong>Self CUDA %:</strong> Time spent in this operation alone on GPU</List.Item>
            <List.Item><strong>CUDA Mem:</strong> Total GPU memory allocated by this operation</List.Item>
            <List.Item><strong># of Calls:</strong> Number of times this operation was executed</List.Item>
          </List>
</div>
<div data-slide>
          <Title order={3} mt="lg">Detailed Operation Profiling</Title>
          
          <Text>
            Use record_function to annotate and track specific code sections:
          </Text>

          <CodeBlock language="python" code={`def profile_training_components(model, dataloader, optimizer):
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
</div>
          <div data-slide>
          <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/time.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
          </div>
           <div data-slide>
          <CodeBlock language="python" code={`flops = sum([int(evt.flops) for evt in prof.events() if evt.flops])`} />
                    <Flex direction="column" align="center" mt="md">
            <Image
              src="/assets/python-deep-learning/module4/flop.png"
              alt="GPU vs CPU Performance Comparison"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
          </div>
      </Stack>
    </Container>
  );
};

export default ResourceProfiling;