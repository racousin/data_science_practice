import React from 'react';
import { Title, Text, Stack, Container, Table, Alert } from '@mantine/core';
import { Info } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const Initialization = () => {
  const initializationCode = `
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_distribution(weights, title):
    plt.figure(figsize=(8, 4))
    plt.hist(weights.flatten().numpy(), bins=50)
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.show()

# Create layers with different initializations
input_size = 784
hidden_size = 256

# Zero initialization
zero_layer = nn.Linear(input_size, hidden_size)
nn.init.zeros_(zero_layer.weight)

# Random normal initialization
random_layer = nn.Linear(input_size, hidden_size)
nn.init.normal_(random_layer.weight, mean=0.0, std=0.01)

# Xavier/Glorot initialization
xavier_layer = nn.Linear(input_size, hidden_size)
nn.init.xavier_normal_(xavier_layer.weight)

# He initialization
he_layer = nn.Linear(input_size, hidden_size)
nn.init.kaiming_normal_(he_layer.weight)

# Plot distributions
plot_distribution(zero_layer.weight, 'Zero Initialization')
plot_distribution(random_layer.weight, 'Random Normal Initialization')
plot_distribution(xavier_layer.weight, 'Xavier/Glorot Initialization')
plot_distribution(he_layer.weight, 'He Initialization')`;

  const practicalCode = `
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Choose initialization based on activation
        if activation == 'relu':
            nn.init.kaiming_normal_(self.fc1.weight)
            nn.init.kaiming_normal_(self.fc2.weight)
        else:  # tanh, sigmoid
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

# Create models with different initializations
relu_model = NeuralNetwork(784, 256, 10, activation='relu')
tanh_model = NeuralNetwork(784, 256, 10, activation='tanh')`;

  return (
    <Container fluid>
      <Stack gap="xl">
        {/* Importance Section */}
        <div>
          <Title order={2} mb="md" id="importance">Importance of Weight Initialization</Title>
          <Text>
            Weight initialization is crucial for training deep neural networks effectively. 
            Poor initialization can lead to:
          </Text>
          <ul className="list-disc pl-6 mt-2">
            <li>Vanishing gradients: When gradients become too small to enable learning</li>
            <li>Exploding gradients: When gradients become too large, causing unstable training</li>
            <li>Dead neurons: Particularly with ReLU activation when neurons output zero for all inputs</li>
            <li>Slow convergence: When the network takes too long to reach optimal weights</li>
          </ul>
        </div>

        {/* Methods Section */}
        <div>
          <Title order={2} mb="md" id="methods">Initialization Methods</Title>
          
          <Title order={3} mb="sm">Zero Initialization</Title>
          <Text mb="md">
            Setting all weights to zero (or the same value) is generally a poor choice as it makes all neurons in the same layer learn the same features:
          </Text>
          <BlockMath>{String.raw`W = 0`}</BlockMath>

          <Title order={3} mt="lg" mb="sm">Random Initialization</Title>
          <Text mb="md">
            Randomly initializing weights from a normal or uniform distribution with small values:
          </Text>
          <BlockMath>{String.raw`W \sim \mathcal{N}(0, \sigma^2)`}</BlockMath>

          <Title order={3} mt="lg" mb="sm">Xavier/Glorot Initialization</Title>
          <Text mb="md">
            Designed for linear and tanh activations, scales weights based on layer sizes:
          </Text>
          <BlockMath>
  {String.raw`W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})`}
</BlockMath>

          <Title order={3} mt="lg" mb="sm">He Initialization</Title>
          <Text mb="md">
            Designed for ReLU activations, accounts for the ReLU's properties:
          </Text>
          <BlockMath>
            {"W_{ij} \\sim \\mathcal{N}(0, \\sqrt{\\frac{2}{n_{in}}})"}
          </BlockMath>

          <CodeBlock
            language="python"
            code={initializationCode}
          />
        </div>

        {/* Guidelines Section */}
        <div>
          <Title order={2} mb="md" id="guidelines">Selection Guidelines</Title>
          
          <Alert 
            icon={<Info size={16} />}
            title="Key Considerations"
            color="blue"
            radius="md"
            mb="md"
            >
            Choose initialization based on your activation function and architecture depth.
          </Alert>

          <Table striped withTableBorder>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Activation Function</Table.Th>
                <Table.Th>Recommended Initialization</Table.Th>
                <Table.Th>PyTorch Implementation</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              <Table.Tr>
                <Table.Td>ReLU</Table.Td>
                <Table.Td>He initialization</Table.Td>
                <Table.Td>nn.init.kaiming_normal_</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Tanh</Table.Td>
                <Table.Td>Xavier initialization</Table.Td>
                <Table.Td>nn.init.xavier_normal_</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Sigmoid</Table.Td>
                <Table.Td>Xavier initialization</Table.Td>
                <Table.Td>nn.init.xavier_normal_</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Linear</Table.Td>
                <Table.Td>Xavier initialization</Table.Td>
                <Table.Td>nn.init.xavier_normal_</Table.Td>
              </Table.Tr>
            </Table.Tbody>
          </Table>

          <Title order={3} mt="xl" mb="md">Practical Implementation</Title>
          <Text mb="md">
            Here's how to implement proper initialization in a PyTorch model:
          </Text>

          <CodeBlock
            language="python"
            code={practicalCode}
          />

          <Text mt="lg" mb="md">
            Best practices for initialization:
          </Text>
          <ul className="list-disc pl-6">
            <li>Initialize biases to zero or small constant values</li>
            <li>Use He initialization for ReLU-based networks (most common)</li>
            <li>Use Xavier initialization for tanh/sigmoid networks</li>
            <li>Consider the network depth when choosing initialization scale</li>
            <li>Monitor initial gradients to ensure they're neither too large nor too small</li>
          </ul>
        </div>
      </Stack>
    </Container>
  );
};

export default Initialization;