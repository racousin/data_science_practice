import React from 'react';
import { Title, Text, Stack, Accordion, Container, List, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import WeightInitialization from './EssentialComponents/WeightInitialization';
import 'katex/dist/katex.min.css';

// Import components
import Activation from './EssentialComponents/Activation';
import Dropout from './EssentialComponents/Dropout';
import CategoricalEmbeddings from './EssentialComponents/CategoricalEmbeddings';
import BatchNormalization from './EssentialComponents/BatchNormalization';
import SkipConnections from './EssentialComponents/ResidualConnections';

const EssentialLayers = () => {
  return (
    <Container fluid>
      <Stack spacing="lg">
        <div data-slide>
        <Title order={1}>Essential Layers</Title>
        </div>
        <div data-slide>
          <Title order={2}>Foundation Layers</Title>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Linear/Fully Connected</Title>
          <Text>
            Core transformation layer performing matrix multiplication + bias.
          </Text>
          <Text mt="xs"><strong>Math Formulation:</strong></Text>
          <BlockMath math="y = xW^T + b" />
          <Text>where <InlineMath math="x \in \mathbb{R}^{n \times d_{in}}" />, <InlineMath math="W \in \mathbb{R}^{d_{out} \times d_{in}}" />, <InlineMath math="b \in \mathbb{R}^{d_{out}}" /></Text>
          <Text mt="xs"><strong>Input/Output Shape:</strong></Text>
          <List>
            <List.Item>Input: <InlineMath math="(\text{batch\_size}, \text{in\_features})" /></List.Item>
            <List.Item>Output: <InlineMath math="(\text{batch\_size}, \text{out\_features})" /></List.Item>
          </List>
          <Text mt="xs"><strong>Parameters:</strong> <InlineMath math="\text{in\_features} \times \text{out\_features} + \text{out\_features}" /></Text>
          <CodeBlock language="python" code={`nn.Linear(in_features=128, out_features=64)
# Performs: output = input @ weight.T + bias`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Convolution</Title>
          <Text>
            Spatial pattern detection in images. Local connectivity with shared weights.
          </Text>
          <Text mt="xs"><strong>Math Formulation (2D Conv):</strong></Text>
          <BlockMath math="y_{i,j,c} = \sum_{m,n,k} x_{i \cdot s + m, j \cdot s + n, k} \cdot W_{m,n,k,c} + b_c" />
          <Text>where <InlineMath math="s" /> is stride, <InlineMath math="W" /> is the kernel weights</Text>
          <Text mt="xs"><strong>Input/Output Shape (Conv2d):</strong></Text>
          <List>
            <List.Item>Input: <InlineMath math="(\text{batch}, \text{in\_channels}, \text{height}, \text{width})" /></List.Item>
            <List.Item>Output: <InlineMath math="(\text{batch}, \text{out\_channels}, H_{out}, W_{out})" /></List.Item>
          </List>
          <Text>where <InlineMath math="H_{out} = \lfloor \frac{H + 2p - k}{s} \rfloor + 1" />, <InlineMath math="W_{out} = \lfloor \frac{W + 2p - k}{s} \rfloor + 1" /></Text>
          <Text mt="xs"><strong>Parameters:</strong> <InlineMath math="(\text{kernel\_size}^2 \times \text{in\_channels} + 1) \times \text{out\_channels}" /></Text>
          <CodeBlock language="python" code={`nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# Slides kernel across input, computing local feature maps`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Recurrent</Title>
          <Text>
            Process sequential data by maintaining hidden state across time steps.
          </Text>
          <Text mt="xs"><strong>Math Formulation (LSTM):</strong></Text>
          <BlockMath math="f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)" />
          <BlockMath math="i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)" />
          <BlockMath math="o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)" />
          <BlockMath math="c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)" />
          <BlockMath math="h_t = o_t \odot \tanh(c_t)" />
          <Text mt="xs"><strong>Input/Output Shape (LSTM):</strong></Text>
          <List>
            <List.Item>Input: <InlineMath math="(\text{seq\_len}, \text{batch}, \text{input\_size})" /></List.Item>
            <List.Item>Output: <InlineMath math="(\text{seq\_len}, \text{batch}, \text{hidden\_size})" /></List.Item>
            <List.Item>Hidden: <InlineMath math="(\text{num\_layers}, \text{batch}, \text{hidden\_size})" /></List.Item>
          </List>
          <Text mt="xs"><strong>Parameters per layer:</strong> <InlineMath math="4 \times (\text{input\_size} \times \text{hidden\_size} + \text{hidden\_size}^2 + 2 \times \text{hidden\_size})" /></Text>
          <CodeBlock language="python" code= {`nn.LSTM(input_size=128, hidden_size=256, num_layers=2)
# Process sequences with memory of previous inputs`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Attention</Title>
          <Text>
            Dynamic focus mechanism that relates different positions in sequences.
          </Text>
          <Text mt="xs"><strong>Math Formulation (Scaled Dot-Product Attention):</strong></Text>
          <BlockMath math="Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V" />
          <Text>For Multi-Head Attention:</Text>
          <BlockMath math="MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O" />
          <BlockMath math="head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)" />
          <Text mt="xs"><strong>Input/Output Shape:</strong></Text>
          <List>
            <List.Item>Query: <InlineMath math="(\text{seq\_len}, \text{batch}, \text{embed\_dim})" /></List.Item>
            <List.Item>Key: <InlineMath math="(\text{seq\_len}, \text{batch}, \text{embed\_dim})" /></List.Item>
            <List.Item>Value: <InlineMath math="(\text{seq\_len}, \text{batch}, \text{embed\_dim})" /></List.Item>
            <List.Item>Output: <InlineMath math="(\text{seq\_len}, \text{batch}, \text{embed\_dim})" /></List.Item>
          </List>
          <Text mt="xs"><strong>Parameters:</strong> <InlineMath math="4 \times \text{embed\_dim}^2" /> (for Q, K, V, and output projections)</Text>
          <CodeBlock language="python" code={`nn.MultiheadAttention(embed_dim=512, num_heads=8)
# Computes weighted importance between sequence elements`}/>
        </div>

        <div data-slide>
          <Title order={3} mt="md">Why Specialized Layers Outperform MLPs</Title>
          <Text>
            These architectures achieve <strong>better performance with fewer parameters</strong> by exploiting data structure instead of learning it from scratch.
          </Text>
          
          <Text mt="md"><strong>Key Example - Images:</strong></Text>
          <Text>
            When you flatten a 28×28 image for an MLP, you lose spatial proximity information. Pixels next to each other become arbitrary positions in a vector. The MLP must learn these relationships from scratch using many parameters.
          </Text>
          <Text mt="sm">
            CNNs preserve the 2D structure. A 3×3 filter naturally captures local patterns (edges, corners) using only 9 shared weights instead of hundreds of connections per neuron.
          </Text>
        </div>

        
          <div data-slide>
  <Title order={2} mb="md">nn.Module Overview</Title>
  <Text mb="md">
    <Code>nn.Module</Code> is the base class for all neural network components in PyTorch. 
    It provides automatic differentiation machinery and parameter management.
  </Text>
  
  <CodeBlock language="python" code={`class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer definitions stored as module attributes
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Computation graph definition
        x = torch.relu(self.layer1(x))
        return self.layer2(x)`} />
</div>

<div data-slide>
  <Title order={2} mb="md">The __init__ Method</Title>
  <Text mb="md">
    The constructor registers all components that contain learnable parameters.
  </Text>
  
  <CodeBlock language="python" code={`def __init__(self):
    super().__init__()`} />
  
  <Text mt="sm" mb="sm">
    Pre-built layers contain parameters that will be automatically tracked:
  </Text>
  
  <CodeBlock language="python" code={`    # Pre-built layers (contain parameters)
    self.conv = nn.Conv2d(3, 64, kernel_size=3)
    self.bn = nn.BatchNorm2d(64)
    self.fc = nn.Linear(64 * 28 * 28, 10)`} />
  
  <Text mt="sm" mb="sm">
    Manual layer parameters must be wrapped in <Code>nn.Parameter</Code> to be tracked:
  </Text>
  
  <CodeBlock language="python" code={`    # Manual layers parameters
    self.params = nn.Parameter(torch.zeros(10))
    # self.params = torch.zeros(10, requires_grad=True)  # will not work correctly`} />
  
  <Text mt="sm" mb="sm">
    Non-parameter attributes (buffers) are tensors that move with the model but aren't trainable:
  </Text>
  
  <CodeBlock language="python" code={`    # Non-parameter attributes (buffers)
    self.register_buffer('running_mean', torch.zeros(64)))`} />
  
  <Text mt="sm" mb="sm">
    Regular Python attributes are not tracked, so it's okay to define them directly in forward:
  </Text>
  
  <CodeBlock language="python" code={`    # Regular Python attributes (not tracked)
    self.activation = nn.ReLU()`} />
  
</div>

<div data-slide>
  <Title order={2} mb="md">The forward Method</Title>
  <Text mb="md">
    Defines the computation performed at every call. Constructs the computational 
    graph dynamically during execution.
  </Text>
  
  <CodeBlock language="python" code={`def forward(self, x):
    # Sequential operations
    x = self.conv(x)
    x = self.bn(x)
    x = torch.relu(x)
    x = x.view(x.size(0), -1)  # Reshape
    x = self.fc(x)
    return x
    
# Usage
model = MyModel()
output = model(input_tensor)  # Calls forward()
# Equivalent to: output = model.forward(input_tensor)`} />
  
</div>

<div data-slide>
  <Title order={2} mb="md">Training and Evaluation Modes</Title>
  <Text mb="md">
    Modules have two modes affecting specific layers that behave differently during training and inference. 
    Mode setting propagates to all submodules.
  </Text>
  
  <CodeBlock language="python" code={`model.train()     # Set training mode (default)
model.eval()      # Set evaluation mode

# Check current mode
if model.training:
    print("Training mode")`} />
  
  <Text mt="md" mb="sm">
    <strong>Layers affected by mode:</strong>
  </Text>

  
  <CodeBlock language="python" code={`# Example behavior differences
dropout = nn.Dropout(p=0.5)
batchnorm = nn.BatchNorm1d(100)

# Training mode: dropout active, batchnorm updates stats
model.train()
x = dropout(x)      # ~50% of values set to zero
x = batchnorm(x)    # Uses batch mean/var, updates running stats

# Evaluation mode: dropout inactive, batchnorm uses fixed stats  
model.eval()
x = dropout(x)      # No dropout applied, x passes through unchanged
x = batchnorm(x)    # Uses fixed running mean/var, no updates`} />
</div>

<div data-slide>
  <Title order={2} mb="md">Accessing Parameters</Title>
  <Text mb="md">
    Multiple methods to access and inspect model parameters for optimization, 
    saving, or analysis.
  </Text>
  
  <CodeBlock language="python" code={`# Iterator of parameter tensors
for param in model.parameters():
    print(param.shape, param.requires_grad)

# Iterator of (name, tensor) pairs
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    # Example: "layer1.weight: torch.Size([128, 784])"

# Dictionary of all parameters
state = model.state_dict()
print(state.keys())
# dict_keys(['layer1.weight', 'layer1.bias', ...])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)`} />
</div>

<div data-slide>
  <Title order={2} mb="md">Custom Layers with Direct Parameters</Title>
  <Text mb="md">
    Define learnable parameters directly using <Code>nn.Parameter</Code> for 
    custom operations not covered by standard layers.
  </Text>
  
  <CodeBlock language="python" code={`class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Direct parameter definition
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Non-learnable buffer
        self.register_buffer('scale', torch.ones(1))
    
    def forward(self, x):
        # Custom computation
        return torch.matmul(x, self.weight.t()) + self.bias * self.scale

# You can then use it as pre-build layers
CustomLayer(10, 5)`} />
</div>

<div data-slide>
  <Title order={2} mb="md">Module Composition</Title>
  <Text mb="md">
    Modules can contain other modules, creating hierarchical structures. 
    Parameter management propagates through the hierarchy.
  </Text>
  
  <CodeBlock language="python" code={`class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, x):
        return x + self.mlp(self.norm(x))

model = nn.Sequential(
    Block(512),
    Block(512),
    nn.Linear(512, 10)
)`} />
</div>
          <WeightInitialization/>
          <Activation/>
        
          <Title order={2}>Regularization Layers</Title>
          
          
          <Dropout/>
          
<BatchNormalization/>
          <Title order={2}>Specialized Components</Title>
          
          
<CategoricalEmbeddings/>
          
          
<SkipConnections/>
        
      </Stack>
    </Container>
  );
};
export default EssentialLayers;