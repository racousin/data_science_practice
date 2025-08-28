import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const MLPFundamentals = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
        {/* Part 3: Multi-Layer Perceptron Fundamentals */}
        <div data-slide>
          <Title order={1} className="mb-6">
            Part 3: Multi-Layer Perceptron Fundamentals
          </Title>
          
          {/* The Neuron as Computational Unit */}
          <section className="mb-12">
            <Title order={2} className="mb-6" id="neuron">
              The Neuron as a Computational Unit
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 mb-6">
              <Title order={3} className="mb-4">Biological Inspiration</Title>
              <Text size="lg" className="mb-4">
                Artificial neurons are loosely inspired by biological neurons in the brain. While the biological 
                reality is far more complex, the mathematical abstraction captures the essential computational principle: 
                weighted integration of inputs followed by non-linear activation.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Biological Neuron</Title>
                    <List size="sm">
                      <List.Item><strong>Dendrites:</strong> Receive signals from other neurons</List.Item>
                      <List.Item><strong>Cell Body (Soma):</strong> Integrates incoming signals</List.Item>
                      <List.Item><strong>Axon:</strong> Transmits output signal</List.Item>
                      <List.Item><strong>Synapses:</strong> Connections with variable strength</List.Item>
                      <List.Item><strong>Action Potential:</strong> All-or-nothing firing response</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Artificial Neuron</Title>
                    <List size="sm">
                      <List.Item><strong>Inputs:</strong> Feature values <InlineMath>{`x_1, x_2, ..., x_n`}</InlineMath></List.Item>
                      <List.Item><strong>Weights:</strong> Connection strengths <InlineMath>{`w_1, w_2, ..., w_n`}</InlineMath></List.Item>
                      <List.Item><strong>Summation:</strong> Weighted sum <InlineMath>{`z = \\sum w_i x_i + b`}</InlineMath></List.Item>
                      <List.Item><strong>Activation:</strong> Non-linear function <InlineMath>{`a = \\sigma(z)`}</InlineMath></List.Item>
                      <List.Item><strong>Output:</strong> Activation value passed forward</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Mathematical Model of a Neuron */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Mathematical Model of a Neuron</Title>
              
              <Paper className="p-4 bg-white mb-4">
                <Title order={4} className="mb-3">The Perceptron Model</Title>
                <Text className="mb-3">
                  A single neuron computes a weighted sum of inputs, adds a bias term, and applies an activation function:
                </Text>
                
                <div className="space-y-4">
                  <div>
                    <Text className="font-semibold mb-2">Step 1: Linear Combination</Text>
                    <BlockMath>{`z = \\sum_{i=1}^n w_i x_i + b = w^T x + b`}</BlockMath>
                    <Text size="sm" color="dimmed">
                      where <InlineMath>{`w \\in \\mathbb{R}^n`}</InlineMath> are weights, 
                      <InlineMath>{`x \\in \\mathbb{R}^n`}</InlineMath> are inputs, 
                      <InlineMath>{`b \\in \\mathbb{R}`}</InlineMath> is bias
                    </Text>
                  </div>
                  
                  <div>
                    <Text className="font-semibold mb-2">Step 2: Activation</Text>
                    <BlockMath>{`a = \\sigma(z)`}</BlockMath>
                    <Text size="sm" color="dimmed">
                      where <InlineMath>{`\\sigma`}</InlineMath> is a non-linear activation function
                    </Text>
                  </div>
                </div>
                
                <CodeBlock language="python" code={`import torch
import torch.nn as nn

class Neuron(nn.Module):
    def __init__(self, input_dim, activation='relu'):
        super().__init__()
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
    
    def forward(self, x):
        # Linear combination
        z = torch.dot(x, self.weight) + self.bias
        # Apply activation
        a = self.activation(z)
        return a

# Example usage
neuron = Neuron(input_dim=3)
x = torch.tensor([1.0, 2.0, 3.0])
output = neuron(x)
print(f"Neuron output: {output.item():.4f}")`} />
              </Paper>

              {/* Why Non-linearity is Essential */}
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">Why Non-linearity is Essential</Title>
                <Text className="mb-3">
                  Without activation functions, stacking linear layers would be pointless:
                </Text>
                
                <BlockMath>{`f(x) = W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2 = W_{eq} x + b_{eq}`}</BlockMath>
                
                <Text size="sm" className="mt-3">
                  Multiple linear transformations collapse to a single linear transformation. 
                  Non-linear activations enable deep networks to learn complex, non-linear mappings.
                </Text>
              </Paper>
            </Paper>

            {/* Activation Functions */}
            <Paper className="p-6 bg-blue-50 mb-6">
              <Title order={3} className="mb-4">Common Activation Functions</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">ReLU (Rectified Linear Unit)</Title>
                    <BlockMath>{`\\text{ReLU}(z) = \\max(0, z)`}</BlockMath>
                    
                    <List size="sm" className="mt-3">
                      <List.Item color="green">✓ Simple and efficient computation</List.Item>
                      <List.Item color="green">✓ No vanishing gradient for positive values</List.Item>
                      <List.Item color="green">✓ Sparse activation (biological plausibility)</List.Item>
                      <List.Item color="red">✗ Dead neurons (zero gradient for negative inputs)</List.Item>
                    </List>
                    
                    <Text className="font-semibold text-sm mt-3">Gradient:</Text>
                    <BlockMath>{`\\frac{\\partial \\text{ReLU}}{\\partial z} = \\begin{cases} 1 & z > 0 \\\\ 0 & z \\leq 0 \\end{cases}`}</BlockMath>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Sigmoid</Title>
                    <BlockMath>{`\\sigma(z) = \\frac{1}{1 + e^{-z}}`}</BlockMath>
                    
                    <List size="sm" className="mt-3">
                      <List.Item color="green">✓ Smooth and differentiable</List.Item>
                      <List.Item color="green">✓ Output bounded in [0, 1]</List.Item>
                      <List.Item color="green">✓ Probability interpretation</List.Item>
                      <List.Item color="red">✗ Vanishing gradients for large |z|</List.Item>
                    </List>
                    
                    <Text className="font-semibold text-sm mt-3">Gradient:</Text>
                    <BlockMath>{`\\frac{\\partial \\sigma}{\\partial z} = \\sigma(z)(1 - \\sigma(z))`}</BlockMath>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Grid gutter="lg" className="mt-4">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Tanh (Hyperbolic Tangent)</Title>
                    <BlockMath>{`\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}`}</BlockMath>
                    
                    <List size="sm" className="mt-3">
                      <List.Item color="green">✓ Zero-centered output [-1, 1]</List.Item>
                      <List.Item color="green">✓ Stronger gradients than sigmoid</List.Item>
                      <List.Item color="red">✗ Still suffers from vanishing gradients</List.Item>
                    </List>
                    
                    <Text className="font-semibold text-sm mt-3">Gradient:</Text>
                    <BlockMath>{`\\frac{\\partial \\tanh}{\\partial z} = 1 - \\tanh^2(z)`}</BlockMath>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Modern Variants</Title>
                    
                    <div className="space-y-3">
                      <div>
                        <Text className="font-semibold text-sm">Leaky ReLU:</Text>
                        <BlockMath>{`\\text{LeakyReLU}(z) = \\begin{cases} z & z > 0 \\\\ \\alpha z & z \\leq 0 \\end{cases}`}</BlockMath>
                        <Text size="xs" color="dimmed">Small slope α (e.g., 0.01) prevents dead neurons</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">GELU (Gaussian Error Linear Unit):</Text>
                        <BlockMath>{`\\text{GELU}(z) = z \\cdot \\Phi(z)`}</BlockMath>
                        <Text size="xs" color="dimmed">Smooth approximation of ReLU, used in transformers</Text>
                      </div>
                    </div>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <CodeBlock language="python" code={`import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Generate input values
z = torch.linspace(-5, 5, 100)

# Compute activations
relu = F.relu(z)
sigmoid = torch.sigmoid(z)
tanh = torch.tanh(z)
leaky_relu = F.leaky_relu(z, negative_slope=0.01)
gelu = F.gelu(z)

# Plotting (conceptual)
activations = {
    'ReLU': relu,
    'Sigmoid': sigmoid,
    'Tanh': tanh,
    'Leaky ReLU': leaky_relu,
    'GELU': gelu
}

for name, activation in activations.items():
    plt.plot(z, activation, label=name)
plt.legend()
plt.grid(True)
plt.xlabel('z')
plt.ylabel('activation(z)')
plt.title('Activation Functions Comparison')`} />
            </Paper>
          </section>
        </div>

        {/* Network Architecture and Forward Propagation */}
        <div data-slide>
          <section className="mb-12">
            <Title order={2} className="mb-6" id="network-architecture">
              Network Architecture and Forward Propagation
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-green-50 to-teal-50 mb-6">
              <Title order={3} className="mb-4">Multi-Layer Perceptron Architecture</Title>
              <Text size="lg" className="mb-4">
                A Multi-Layer Perceptron (MLP) consists of multiple layers of neurons, where each neuron in a layer 
                is connected to all neurons in the previous layer (fully connected or dense layers).
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Input Layer</Title>
                    <Text size="sm" className="mb-2">
                      Dimension: <InlineMath>{`\\mathbb{R}^{d_{in}}`}</InlineMath>
                    </Text>
                    <List size="sm">
                      <List.Item>Receives raw features</List.Item>
                      <List.Item>No computation</List.Item>
                      <List.Item>Size = number of features</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Hidden Layers</Title>
                    <Text size="sm" className="mb-2">
                      Dimension: <InlineMath>{`\\mathbb{R}^{h_1}, \\mathbb{R}^{h_2}, ...`}</InlineMath>
                    </Text>
                    <List size="sm">
                      <List.Item>Learn representations</List.Item>
                      <List.Item>Non-linear transformations</List.Item>
                      <List.Item>Width and depth are hyperparameters</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Output Layer</Title>
                    <Text size="sm" className="mb-2">
                      Dimension: <InlineMath>{`\\mathbb{R}^{d_{out}}`}</InlineMath>
                    </Text>
                    <List size="sm">
                      <List.Item>Produces predictions</List.Item>
                      <List.Item>Size = number of outputs/classes</List.Item>
                      <List.Item>Task-specific activation</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Forward Propagation Mathematics */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Forward Propagation Mathematics</Title>
              
              <Text className="mb-4">
                Forward propagation computes the network output by sequentially applying transformations through each layer:
              </Text>
              
              <Paper className="p-4 bg-white mb-4">
                <Title order={4} className="mb-3">Layer-wise Computation</Title>
                
                <div className="space-y-4">
                  <div>
                    <Text className="font-semibold mb-2">For layer <InlineMath>{`l`}</InlineMath> with input <InlineMath>{`h^{(l-1)}`}</InlineMath>:</Text>
                    <BlockMath>{`z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}`}</BlockMath>
                    <BlockMath>{`h^{(l)} = \\sigma^{(l)}(z^{(l)})`}</BlockMath>
                  </div>
                  
                  <div>
                    <Text className="font-semibold mb-2">Complete forward pass for L layers:</Text>
                    <BlockMath>{`\\begin{align}
                      h^{(0)} &= x \\quad \\text{(input)} \\\\
                      h^{(1)} &= \\sigma^{(1)}(W^{(1)} h^{(0)} + b^{(1)}) \\\\
                      h^{(2)} &= \\sigma^{(2)}(W^{(2)} h^{(1)} + b^{(2)}) \\\\
                      &\\vdots \\\\
                      \\hat{y} &= h^{(L)} = \\sigma^{(L)}(W^{(L)} h^{(L-1)} + b^{(L)})
                    \\end{align}`}</BlockMath>
                  </div>
                </div>
              </Paper>

              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Vectorized Implementation</Title>
                <Text size="sm" className="mb-3">
                  For batch processing with B samples:
                </Text>
                
                <List size="sm">
                  <List.Item>Input: <InlineMath>{`X \\in \\mathbb{R}^{B \\times d_{in}}`}</InlineMath></List.Item>
                  <List.Item>Weights: <InlineMath>{`W^{(l)} \\in \\mathbb{R}^{h_l \\times h_{l-1}}`}</InlineMath></List.Item>
                  <List.Item>Computation: <InlineMath>{`H^{(l)} = \\sigma(H^{(l-1)} W^{(l)T} + b^{(l)})`}</InlineMath></List.Item>
                </List>
                
                <CodeBlock language="python" code={`import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example: 3-layer MLP for classification
model = MLP(
    input_dim=784,      # e.g., flattened 28x28 image
    hidden_dims=[256, 128],  # two hidden layers
    output_dim=10       # 10 classes
)

# Forward pass
batch_size = 32
x = torch.randn(batch_size, 784)
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]`} />
              </Paper>
            </Paper>

            {/* Representation Learning */}
            <Paper className="p-6 bg-purple-50 mb-6">
              <Title order={3} className="mb-4">Hierarchical Representation Learning</Title>
              
              <Text className="mb-4">
                Deep networks learn increasingly abstract representations through layers:
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={12}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Feature Hierarchy in Vision Tasks</Title>
                    
                    <div className="space-y-3">
                      <div className="flex items-center space-x-4">
                        <div className="w-20 text-right font-semibold">Layer 1:</div>
                        <div className="flex-1">
                          <Text size="sm">Edge detectors, simple patterns</Text>
                          <Text size="xs" color="dimmed">Learns Gabor-like filters, color blobs</Text>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="w-20 text-right font-semibold">Layer 2:</div>
                        <div className="flex-1">
                          <Text size="sm">Corners, texture patterns</Text>
                          <Text size="xs" color="dimmed">Combines edges into more complex shapes</Text>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="w-20 text-right font-semibold">Layer 3:</div>
                        <div className="flex-1">
                          <Text size="sm">Object parts</Text>
                          <Text size="xs" color="dimmed">Eyes, wheels, windows</Text>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="w-20 text-right font-semibold">Layer 4+:</div>
                        <div className="flex-1">
                          <Text size="sm">Complete objects and concepts</Text>
                          <Text size="xs" color="dimmed">Faces, cars, scenes</Text>
                        </div>
                      </div>
                    </div>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper className="p-4 bg-yellow-50 mt-4">
                <Title order={4} className="mb-3">Universal Approximation Theorem</Title>
                <Text size="sm" className="mb-3">
                  A foundational result showing MLPs can approximate any continuous function:
                </Text>
                
                <div className="p-3 bg-white rounded">
                  <Text className="italic text-sm">
                    "A feedforward network with a single hidden layer containing a finite number of neurons can 
                    approximate any continuous function on compact subsets of ℝⁿ, under mild assumptions on the 
                    activation function."
                  </Text>
                </div>
                
                <Text size="sm" className="mt-3">
                  <strong>Important caveats:</strong>
                </Text>
                <List size="sm">
                  <List.Item>May require exponentially many neurons</List.Item>
                  <List.Item>Doesn't guarantee learnability via gradient descent</List.Item>
                  <List.Item>Deep networks often more efficient than wide shallow networks</List.Item>
                </List>
              </Paper>
            </Paper>
          </section>
        </div>

        {/* Parameters to Optimize */}
        <div data-slide>
          <section className="mb-12">
            <Title order={2} className="mb-6" id="parameters">
              Parameters to Optimize: Weights and Biases
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-orange-50 to-red-50 mb-6">
              <Title order={3} className="mb-4">Understanding Network Parameters</Title>
              
              <Text size="lg" className="mb-4">
                The learnable parameters of a neural network are the weights and biases that determine how 
                information flows and transforms through the network.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Weights <InlineMath>{`W`}</InlineMath></Title>
                    <Text size="sm" className="mb-3">
                      Control the strength of connections between neurons:
                    </Text>
                    <List size="sm">
                      <List.Item><strong>Positive weights:</strong> Excitatory connections</List.Item>
                      <List.Item><strong>Negative weights:</strong> Inhibitory connections</List.Item>
                      <List.Item><strong>Large magnitude:</strong> Strong influence</List.Item>
                      <List.Item><strong>Near zero:</strong> Weak/no connection</List.Item>
                    </List>
                    <Text size="sm" className="mt-3">
                      Matrix <InlineMath>{`W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}`}</InlineMath> connects 
                      layer <InlineMath>{`l-1`}</InlineMath> to layer <InlineMath>{`l`}</InlineMath>
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Biases <InlineMath>{`b`}</InlineMath></Title>
                    <Text size="sm" className="mb-3">
                      Shift the activation threshold of neurons:
                    </Text>
                    <List size="sm">
                      <List.Item><strong>Positive bias:</strong> Easier to activate</List.Item>
                      <List.Item><strong>Negative bias:</strong> Harder to activate</List.Item>
                      <List.Item><strong>Flexibility:</strong> Allows learning of any offset</List.Item>
                      <List.Item><strong>Essential:</strong> Without bias, functions pass through origin</List.Item>
                    </List>
                    <Text size="sm" className="mt-3">
                      Vector <InlineMath>{`b^{(l)} \\in \\mathbb{R}^{n_l}`}</InlineMath> for layer <InlineMath>{`l`}</InlineMath>
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Parameter Initialization */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Parameter Initialization Strategies</Title>
              
              <Text className="mb-4">
                Proper initialization is crucial for training deep networks. Poor initialization can lead to 
                vanishing/exploding gradients or slow convergence.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-blue-50">
                    <Title order={4} className="mb-3">Xavier/Glorot Initialization</Title>
                    <Text size="sm" className="mb-3">For sigmoid/tanh activations:</Text>
                    <BlockMath>{`W_{ij} \\sim \\mathcal{N}\\left(0, \\frac{2}{n_{in} + n_{out}}\\right)`}</BlockMath>
                    <Text size="sm" className="mt-2">or uniform:</Text>
                    <BlockMath>{`W_{ij} \\sim U\\left[-\\sqrt{\\frac{6}{n_{in} + n_{out}}}, \\sqrt{\\frac{6}{n_{in} + n_{out}}}\\right]`}</BlockMath>
                    <Text size="xs" color="dimmed" className="mt-2">
                      Maintains variance of activations across layers
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-green-50">
                    <Title order={4} className="mb-3">He/Kaiming Initialization</Title>
                    <Text size="sm" className="mb-3">For ReLU activations:</Text>
                    <BlockMath>{`W_{ij} \\sim \\mathcal{N}\\left(0, \\frac{2}{n_{in}}\\right)`}</BlockMath>
                    <Text size="sm" className="mt-2">or uniform:</Text>
                    <BlockMath>{`W_{ij} \\sim U\\left[-\\sqrt{\\frac{6}{n_{in}}}, \\sqrt{\\frac{6}{n_{in}}}\\right]`}</BlockMath>
                    <Text size="xs" color="dimmed" className="mt-2">
                      Accounts for ReLU killing half the neurons
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <CodeBlock language="python" code={`import torch
import torch.nn as nn
import math

# Manual initialization
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', 
                                nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply to model
model = MLP(784, [256, 128], 10)
model.apply(he_init)  # Initialize all layers

# PyTorch default initialization (He for ReLU)
layer = nn.Linear(100, 50)
# Automatically uses appropriate initialization

# Custom initialization
with torch.no_grad():
    layer.weight.normal_(0, 0.01)  # Small random weights
    layer.bias.zero_()              # Zero bias`} />
            </Paper>

            {/* Parameter Counting and Memory */}
            <Paper className="p-6 bg-yellow-50 mb-6">
              <Title order={3} className="mb-4">Parameter Counting and Memory Requirements</Title>
              
              <Paper className="p-4 bg-white mb-4">
                <Title order={4} className="mb-3">Calculating Total Parameters</Title>
                
                <Text className="mb-3">
                  For an MLP with layer sizes <InlineMath>{`[d_0, d_1, d_2, ..., d_L]`}</InlineMath>:
                </Text>
                
                <BlockMath>{`\\text{Total Parameters} = \\sum_{l=1}^{L} (d_{l-1} \\times d_l + d_l)`}</BlockMath>
                
                <Text size="sm" color="dimmed" className="mt-2">
                  where <InlineMath>{`d_{l-1} \\times d_l`}</InlineMath> are weights and <InlineMath>{`d_l`}</InlineMath> are biases for layer l
                </Text>
                
                <div className="mt-4 p-3 bg-gray-50 rounded">
                  <Text className="font-semibold text-sm mb-2">Example: Image Classification Network</Text>
                  <Text size="sm">
                    Input: 784 (28×28 image) → Hidden: 512 → Hidden: 256 → Output: 10
                  </Text>
                  <List size="sm" className="mt-2">
                    <List.Item>Layer 1: 784 × 512 + 512 = 401,920 parameters</List.Item>
                    <List.Item>Layer 2: 512 × 256 + 256 = 131,328 parameters</List.Item>
                    <List.Item>Layer 3: 256 × 10 + 10 = 2,570 parameters</List.Item>
                    <List.Item><strong>Total: 535,818 parameters</strong></List.Item>
                    <List.Item>Memory (FP32): 535,818 × 4 bytes ≈ 2.04 MB</List.Item>
                  </List>
                </div>
              </Paper>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-orange-50">
                    <Title order={4} className="mb-3">Memory Considerations</Title>
                    <List size="sm">
                      <List.Item><strong>FP32:</strong> 4 bytes per parameter</List.Item>
                      <List.Item><strong>FP16:</strong> 2 bytes per parameter</List.Item>
                      <List.Item><strong>INT8:</strong> 1 byte per parameter</List.Item>
                      <List.Item><strong>Gradients:</strong> Same size as parameters</List.Item>
                      <List.Item><strong>Optimizer states:</strong> 2× for Adam (momentum + variance)</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-pink-50">
                    <Title order={4} className="mb-3">Efficiency Insights</Title>
                    <List size="sm">
                      <List.Item>Deep narrow networks often better than shallow wide</List.Item>
                      <List.Item>Parameter sharing (CNNs, RNNs) reduces count</List.Item>
                      <List.Item>Pruning can remove 90%+ parameters</List.Item>
                      <List.Item>Quantization reduces memory 2-4×</List.Item>
                      <List.Item>Knowledge distillation creates smaller models</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Backpropagation Preview */}
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mb-6">
              <Title order={3} className="mb-4">How Parameters are Learned: Backpropagation</Title>
              
              <Text className="mb-4">
                Backpropagation efficiently computes gradients of the loss with respect to all parameters using the chain rule:
              </Text>
              
              <Paper className="p-4 bg-white">
                <Title order={4} className="mb-3">The Backpropagation Algorithm</Title>
                
                <div className="space-y-4">
                  <div>
                    <Text className="font-semibold mb-2">1. Forward Pass:</Text>
                    <Text size="sm">Compute activations layer by layer, storing intermediate values</Text>
                    <BlockMath>{`h^{(l)} = \\sigma(W^{(l)} h^{(l-1)} + b^{(l)})`}</BlockMath>
                  </div>
                  
                  <div>
                    <Text className="font-semibold mb-2">2. Compute Loss:</Text>
                    <Text size="sm">Calculate error between prediction and target</Text>
                    <BlockMath>{`\\mathcal{L} = \\ell(\\hat{y}, y)`}</BlockMath>
                  </div>
                  
                  <div>
                    <Text className="font-semibold mb-2">3. Backward Pass:</Text>
                    <Text size="sm">Propagate gradients from output to input</Text>
                    <BlockMath>{`\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\frac{\\partial \\mathcal{L}}{\\partial h^{(l)}} \\cdot \\frac{\\partial h^{(l)}}{\\partial z^{(l)}} \\cdot \\frac{\\partial z^{(l)}}{\\partial W^{(l)}}`}</BlockMath>
                  </div>
                  
                  <div>
                    <Text className="font-semibold mb-2">4. Update Parameters:</Text>
                    <Text size="sm">Apply gradient descent update rule</Text>
                    <BlockMath>{`W^{(l)} \\leftarrow W^{(l)} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}}`}</BlockMath>
                  </div>
                </div>
                
                <CodeBlock language="python" code={`# PyTorch handles backpropagation automatically
model = MLP(784, [256, 128], 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training step
def train_step(model, x, y):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients via backprop
    
    # Parameter update
    optimizer.step()       # Update weights and biases
    
    return loss.item()

# Gradient flow visualization
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape}")
        # After backward(), param.grad contains gradients`} />
              </Paper>
            </Paper>
          </section>
        </div>

        {/* Practical Implementation */}
        <div data-slide>
          <section className="mb-12">
            <Title order={2} className="mb-6" id="implementation">
              Complete MLP Implementation Example
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-blue-50 to-purple-50 mb-6">
              <Title order={3} className="mb-4">Building and Training an MLP from Scratch</Title>
              
              <CodeBlock language="python" code={`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class CustomMLP(nn.Module):
    """
    Flexible MLP with dropout and batch normalization options
    """
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 dropout_rate=0.2, use_batchnorm=True):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Build network architecture
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Linear layer
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Optional batch normalization
            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(dims[i+1]))
            else:
                self.batch_norms.append(nn.Identity())
            
            # Dropout for regularization
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
    def forward(self, x):
        # Process through hidden layers
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
        
        # Output layer (no activation for flexibility)
        x = self.output_layer(x)
        return x

# Training function
def train_mlp(model, train_loader, val_loader, epochs=10, lr=0.001):
    """
    Complete training loop with validation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.2f}%')
    
    return model, history

# Example usage: MNIST-like classification
if __name__ == "__main__":
    # Generate synthetic data
    n_samples = 10000
    input_dim = 784  # 28x28 images flattened
    n_classes = 10
    
    # Create synthetic dataset
    X_train = torch.randn(n_samples, input_dim)
    y_train = torch.randint(0, n_classes, (n_samples,))
    
    X_val = torch.randn(2000, input_dim)
    y_val = torch.randint(0, n_classes, (2000,))
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = CustomMLP(
        input_dim=784,
        hidden_dims=[512, 256, 128],  # 3 hidden layers
        output_dim=10,
        dropout_rate=0.3,
        use_batchnorm=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    trained_model, history = train_mlp(
        model, train_loader, val_loader, 
        epochs=20, lr=0.001
    )
    
    print("\\nTraining complete!")`} />
            </Paper>

            {/* Common Pitfalls and Solutions */}
            <Paper className="p-6 bg-red-50">
              <Title order={3} className="mb-4">Common Pitfalls and Solutions</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Training Issues</Title>
                    <List size="sm">
                      <List.Item>
                        <strong>Vanishing Gradients:</strong> Use ReLU, proper initialization, batch norm
                      </List.Item>
                      <List.Item>
                        <strong>Exploding Gradients:</strong> Gradient clipping, smaller learning rate
                      </List.Item>
                      <List.Item>
                        <strong>Overfitting:</strong> Dropout, L2 regularization, more data
                      </List.Item>
                      <List.Item>
                        <strong>Slow Convergence:</strong> Better initialization, learning rate tuning
                      </List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Architecture Choices</Title>
                    <List size="sm">
                      <List.Item>
                        <strong>Width vs Depth:</strong> Start with 2-3 hidden layers, tune width
                      </List.Item>
                      <List.Item>
                        <strong>Activation Functions:</strong> ReLU default, GELU for transformers
                      </List.Item>
                      <List.Item>
                        <strong>Output Layer:</strong> No activation for regression, softmax for classification
                      </List.Item>
                      <List.Item>
                        <strong>Batch Size:</strong> 32-256 typical, larger for stable gradients
                      </List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>
          </section>
        </div>

        {/* Summary */}
        <div data-slide>
          <section>
            <Title order={2} className="mb-6">Part 3 Summary: MLP Fundamentals</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                  <Title order={3} className="mb-4">Core MLP Concepts</Title>
                  <List spacing="md">
                    <List.Item>Neurons perform weighted sum + non-linear activation</List.Item>
                    <List.Item>Deep networks learn hierarchical representations</List.Item>
                    <List.Item>Forward propagation computes predictions layer by layer</List.Item>
                    <List.Item>Parameters (weights and biases) are learned via backpropagation</List.Item>
                    <List.Item>Proper initialization crucial for training success</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                  <Title order={3} className="mb-4">Practical Insights</Title>
                  <List spacing="md">
                    <List.Item>ReLU activation default choice for hidden layers</List.Item>
                    <List.Item>Batch normalization stabilizes training</List.Item>
                    <List.Item>Dropout provides regularization</List.Item>
                    <List.Item>Adam optimizer works well in practice</List.Item>
                    <List.Item>Monitor validation metrics to prevent overfitting</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
            
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6">
              <Title order={3} className="mb-4 text-center">The Foundation is Set</Title>
              <Text size="lg" className="text-center">
                Multi-Layer Perceptrons form the backbone of deep learning. While modern architectures like 
                CNNs and Transformers add specialized components, they all build upon these fundamental principles 
                of neurons, layers, and gradient-based learning.
              </Text>
            </Paper>
          </section>
        </div>

      </Stack>
    </Container>
  );
};

export default MLPFundamentals;