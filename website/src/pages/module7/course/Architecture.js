import React from 'react';
import { Title, Text, Stack, List, Grid } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const Architecture = () => {
  return (
    <Stack spacing="xl" className="w-full">
      <Title order={1} id="network-structure">Network Structure and Components</Title>
      <Text>
        A fully connected neural network, also known as a feedforward neural network or multilayer perceptron (MLP), 
        consists of layers of interconnected neurons. Each neuron receives inputs from all neurons in the previous layer, 
        applies weights and biases, and passes the result through an activation function.
      </Text>

      <Grid gutter="md">
        <Grid.Col span={12}>
          <Title order={3} className="mb-4">Key Components</Title>
          <List spacing="sm">
            <List.Item><strong>Neurons:</strong> Basic computational units that perform weighted sum operations</List.Item>
            <List.Item><strong>Weights:</strong> Learnable parameters that determine the strength of connections</List.Item>
            <List.Item><strong>Biases:</strong> Learnable offsets added to each neuron's output</List.Item>
            <List.Item><strong>Activation Functions:</strong> Non-linear transformations applied to neuron outputs</List.Item>
          </List>
        </Grid.Col>
      </Grid>

      <Title order={2} id="layers" className="mt-6">Understanding Layers</Title>
      <Text>
        Neural networks are organized into distinct layers, each serving a specific purpose in the network's architecture.
      </Text>

      <Grid gutter="md">
        <Grid.Col span={12}>
          <List spacing="sm">
            <List.Item>
              <strong>Input Layer:</strong> Receives the raw input features (dimension determined by data)
            </List.Item>
            <List.Item>
              <strong>Hidden Layers:</strong> Intermediate layers where most computation occurs
            </List.Item>
            <List.Item>
              <strong>Output Layer:</strong> Produces the final prediction (dimension determined by task)
            </List.Item>
          </List>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class FullyConnectedNet(nn.Module):
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

# Example usage
model = FullyConnectedNet(
    input_size=10,
    hidden_sizes=[64, 32],
    output_size=2
)

# Print model architecture
print(model)`}
      />

      <Title order={2} id="model-capacity" className="mt-6">Model Capacity and Depth</Title>
      <Text>
        The capacity of a neural network is determined by its architecture - specifically the number of layers (depth) 
        and neurons per layer (width). These choices affect the model's ability to learn complex patterns.
      </Text>

      <Grid gutter="md">
        <Grid.Col span={12}>
          <Title order={4}>Impact of Network Size</Title>
          <List spacing="sm">
            <List.Item>
              <strong>Width:</strong> More neurons per layer increase the model's ability to represent complex patterns 
              within each layer
            </List.Item>
            <List.Item>
              <strong>Depth:</strong> More layers allow the model to learn hierarchical representations and more 
              complex functions
            </List.Item>
            <List.Item>
              <strong>Trade-offs:</strong> Larger networks have more capacity but require more data, compute resources, 
              and are prone to overfitting
            </List.Item>
          </List>
        </Grid.Col>
      </Grid>

      <Title order={2} id="computations" className="mt-6">Forward and Backward Computations</Title>
      
      <Stack spacing="md">
        <Text>
          The forward pass computes the output of each layer sequentially. For a single neuron:
        </Text>
        <BlockMath math={`z = \\sum_{i=1}^n w_i x_i + b`} />
        <Text>where:</Text>
        <List>
          <List.Item>z is the pre-activation output</List.Item>
          <List.Item>w_i are the weights</List.Item>
          <List.Item>x_i are the inputs</List.Item>
          <List.Item>b is the bias term</List.Item>
        </List>
        
        <Text className="mt-4">
          The backward pass computes gradients using the chain rule:
        </Text>
        <BlockMath math={`\\frac{\\partial L}{\\partial w_i} = \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w_i}`} />
      </Stack>

      <CodeBlock
        language="python"
        code={`
# Example of manual forward and backward computation
import torch

# Create a simple layer
input_size = 3
output_size = 2
x = torch.randn(input_size)
weights = torch.randn(output_size, input_size, requires_grad=True)
bias = torch.randn(output_size, requires_grad=True)

# Forward pass
z = torch.matmul(weights, x) + bias
output = torch.relu(z)

# Backward pass (assuming MSE loss)
target = torch.randn(output_size)
loss = torch.mean((output - target) ** 2)
loss.backward()

# Print gradients
print("Weight gradients:", weights.grad)
print("Bias gradients:", bias.grad)`}
      />
    </Stack>
  );
};

export default Architecture;