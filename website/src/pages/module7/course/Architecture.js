import React from 'react';
import { Title, Text, Stack, List, Grid, Box, Paper } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';


import ArchitectureSvg from './ArchitectureSvg';
import SingleNeuronSvg from './SingleNeuronSvg'

const NeuralNetworkArchitecture = () => {
  return (
    <Stack spacing="xl" className="w-full">
      <Title order={1} id="network-structure">Neural Network Architecture</Title>
      
      {/* Single Neuron Section */}
      <Stack spacing="md">
        <Title order={2} id="artificial-neuron">The Artificial Neuron</Title>
        <SingleNeuronSvg />
        <Text>
          An artificial neuron is the fundamental building block of neural networks. It performs the following computation:
        </Text>
        <BlockMath>
          {`y = f(\\sum_{i=1}^n w_ix_i + b)`}
        </BlockMath>
        <Text>
          where:
        </Text>
        <List>
          <List.Item><InlineMath>{`x_i`}</InlineMath>: input values</List.Item>
          <List.Item><InlineMath>{`w_i`}</InlineMath>: weights (learnable parameters)</List.Item>
          <List.Item><InlineMath>{`b`}</InlineMath>: bias term (learnable parameter)</List.Item>
          <List.Item><InlineMath>{`f`}</InlineMath>: activation function</List.Item>
        </List>

        <CodeBlock
          language="python"
          code={`
# Define a single neuron with Linear activation function (this is a linear regression function)
class Neuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Initialize weights and bias as learnable parameters
        self.weights = nn.Parameter(torch.randn(input_size, 1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # Perform the linear transformation manually: y = xW + b
        return x @ self.weights + self.bias  # Using matrix multiplication

# Same as
class Neuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)  # Single neuron with weights and bias
        
    def forward(self, x):
        return self.linear(x)

# Example usage
input_size = 3
neuron = Neuron(input_size)
x = torch.randn(1, input_size)  # Sample input
output = neuron(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
`}
        />

<CodeBlock
          language="python"
          code={`
# Define a single neuron with Sigmoid activation function (this is a logistic regression function)
class Neuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Initialize weights and bias as learnable parameters
        self.weights = nn.Parameter(torch.randn(input_size, 1))
        self.bias = nn.Parameter(torch.randn(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Perform the linear transformation manually: y= 1/(1+e**−(xW+b))

        return self.sigmoid(x @ self.weights + self.bias)

# Same as
class Neuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)  # Single neuron with weights and bias
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        return self.sigmoid(self.linear(x)) 

# Example usage
input_size = 3
neuron = Neuron(input_size)
x = torch.randn(1, input_size)  # Sample input
output = neuron(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
`}
        />
      </Stack>

      {/* Network Structure Section */}
      <Stack spacing="md">
        <Title order={2} id="network-structure">Network Structure</Title>
        <Text>
          A fully connected neural network consists of multiple layers where each neuron is connected to all neurons in the adjacent layers.
        </Text>
        <ArchitectureSvg/>
        <Paper p="md" className="bg-gray-50">
          <Title order={3}>Layer Types</Title>
          <List spacing="sm">
            <List.Item>
              <strong>Input Layer:</strong> Receives raw features 
              (<InlineMath>{`x \\in \\mathbb{R}^{n_{input}}`}</InlineMath>)
            </List.Item>
            <List.Item>
      <strong>Hidden Layers:</strong> Transform features where each neuron i in layer l computes: 
      (<InlineMath>{`h_i^{(l)} = f(\\sum_{j} W_{ij}^{(l)}h_j^{(l-1)} + b_i^{(l)})`}</InlineMath>)
      
      <List withPadding listStyleType="none" spacing={0}>
        <List.Item>• l = 1,...,L is the layer index</List.Item>
        <List.Item>• i indexes neurons in layer l</List.Item>
        <List.Item>• j indexes neurons in layer l-1</List.Item>
      </List>
    </List.Item>
            <List.Item>
              <strong>Output Layer:</strong> Produces final predictions 
              (<InlineMath>{`y \\in \\mathbb{R}^{n_{output}}`}</InlineMath>)
            </List.Item>
          </List>
        </Paper>

        <CodeBlock
          language="python"
          code={`
import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        # Create list to hold all layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example architecture
model = FullyConnectedNetwork(
    input_size=784,      # e.g., MNIST image (28x28)
    hidden_sizes=[256, 128],  # Two hidden layers
    output_size=10       # e.g., 10 digit classes
)

# Print model architecture
print(model)
`}
        />
      </Stack>

      {/* Model Capacity Section */}
      <Stack spacing="md">
        <Title order={2} id="model-capacity">Model Capacity and Depth</Title>
        <Text>
          The capacity of a neural network is determined by its architecture:
        </Text>
        <List>
          <List.Item>
            <strong>Width (neurons per layer):</strong> More neurons allow the network to learn more complex patterns within each layer
          </List.Item>
          <List.Item>
            <strong>Depth (number of layers):</strong> Deeper networks can learn hierarchical features and more abstract representations
          </List.Item>
        </List>

        <Paper p="md" className="bg-gray-50">
          <Title order={3}>Architectural Considerations</Title>
          <Grid>
            <Grid.Col span={6}>
              <Text weight={700}>Wider Networks</Text>
              <List size="sm">
                <List.Item>Better at memorization</List.Item>
                <List.Item>Easier to train</List.Item>
                <List.Item>More parameters per layer</List.Item>
              </List>
            </Grid.Col>
            <Grid.Col span={6}>
              <Text weight={700}>Deeper Networks</Text>
              <List size="sm">
                <List.Item>Better at generalization</List.Item>
                <List.Item>Can be harder to train</List.Item>
                <List.Item>More efficient parameter usage</List.Item>
              </List>
            </Grid.Col>
          </Grid>
        </Paper>

        <CodeBlock
          language="python"
          code={`
# Example: Comparing different network capacities
import torch.nn as nn

# Shallow and wide network
wide_model = nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
)

# Deep and narrow network
deep_model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Wide model parameters: {count_parameters(wide_model):,}")
print(f"Deep model parameters: {count_parameters(deep_model):,}")
`}
        />
      </Stack>
    </Stack>
  );
};

export default NeuralNetworkArchitecture;