import React from 'react';
import { Title, Text, Stack, Grid, Timeline, Box, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import { BookOpen, Cpu, Network, Brain } from 'lucide-react';

const Introduction = () => {
  return (
    <Stack spacing="xl" className="w-full">
      <Title order={1} id="introduction">Introduction to Deep Learning</Title>
      
      <Title order={2} id="historical-context" className="mt-6">
        Historical Context and Evolution
      </Title>
      <Text>
        Deep Learning emerged from the broader field of Machine Learning, revolutionizing artificial intelligence through its ability to learn hierarchical representations of data. The field has evolved from the Perceptron (1957) to modern architectures capable of solving complex tasks across various domains.
      </Text>
      
      <Timeline active={3} bulletSize={24} lineWidth={2}>
        <Timeline.Item bullet={<BookOpen size={12} />} title="1943-1957: Early Foundations">
          <Text size="sm">McCulloch-Pitts neuron (1943) and Perceptron (1957) established the foundational concepts of artificial neurons</Text>
        </Timeline.Item>
        <Timeline.Item bullet={<Cpu size={12} />} title="1960-1980: First AI Winter">
          <Text size="sm">Limited computing power and the XOR problem highlighted limitations of single-layer networks</Text>
        </Timeline.Item>
        <Timeline.Item bullet={<Network size={12} />} title="1986-2006: Backpropagation Era">
          <Text size="sm">Introduction of backpropagation algorithm and multi-layer networks</Text>
        </Timeline.Item>
        <Timeline.Item bullet={<Brain size={12} />} title="2012-Present: Deep Learning Revolution">
          <Text size="sm">AlexNet (2012) demonstrated the power of deep networks, leading to breakthroughs across various domains</Text>
        </Timeline.Item>
      </Timeline>

      <Title order={2} id="frameworks" className="mt-6">
        Deep Learning Frameworks
      </Title>
      <Text>
        Modern deep learning development relies heavily on specialized frameworks that provide efficient implementations of automatic differentiation and GPU acceleration.
      </Text>
      
      <Grid grow className="mt-4">
        <Grid.Col span={4}>
          <Box className="p-4 border rounded">
            <Title order={4}>PyTorch 2.5</Title>
            <List>
              <List.Item>Dynamic computational graphs</List.Item>
              <List.Item>Python-first approach</List.Item>
              <List.Item>TorchScript for production</List.Item>
            </List>
          </Box>
        </Grid.Col>
        <Grid.Col span={4}>
          <Box className="p-4 border rounded">
            <Title order={4}>TensorFlow</Title>
            <List>
              <List.Item>Static computational graphs</List.Item>
              <List.Item>Production-ready deployment</List.Item>
              <List.Item>TensorBoard visualization</List.Item>
            </List>
          </Box>
        </Grid.Col>
        <Grid.Col span={4}>
          <Box className="p-4 border rounded">
            <Title order={4}>JAX</Title>
            <List>
              <List.Item>Functional approach</List.Item>
              <List.Item>XLA compilation</List.Item>
              <List.Item>Advanced automatic differentiation</List.Item>
            </List>
          </Box>
        </Grid.Col>
      </Grid>

      <Title order={2} id="backpropagation" className="mt-6">
        Backpropagation and Automatic Differentiation
      </Title>
      <Text>
        Backpropagation is the cornerstone algorithm for training neural networks, utilizing the chain rule to compute gradients efficiently through the network.
      </Text>

      <Box className="p-4 bg-gray-50 rounded mt-4">
        <Title order={4}>Mathematical Formulation</Title>
        <BlockMath>
          {`\\frac{\\partial L}{\\partial w_{i,j}^{(l)}} = \\frac{\\partial L}{\\partial a_j^{(l)}} \\frac{\\partial a_j^{(l)}}{\\partial z_j^{(l)}} \\frac{\\partial z_j^{(l)}}{\\partial w_{i,j}^{(l)}}`}
        </BlockMath>
        <Text className="mt-2">
          Where:
        </Text>
        <List>
          <List.Item><InlineMath>{`L`}</InlineMath> is the loss function</List.Item>
          <List.Item><InlineMath>{`w_{i,j}^{(l)}`}</InlineMath> is the weight connecting neuron i in layer l-1 to neuron j in layer l</List.Item>
          <List.Item><InlineMath>{`a_j^{(l)}`}</InlineMath> is the activation of neuron j in layer l</List.Item>
          <List.Item><InlineMath>{`z_j^{(l)}`}</InlineMath> is the weighted input to neuron j in layer l</List.Item>
        </List>
      </Box>

      <Box className="mt-4">
        <Title order={4}>Automatic Differentiation in PyTorch</Title>
        <Text>
          PyTorch implements reverse-mode automatic differentiation, building a dynamic computational graph that tracks operations for gradient computation.
        </Text>
        <CodeBlock
          language="python"
          code={`
import torch

# Create tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Forward pass: define the computation
z = x**2 + y**3

# Backward pass: compute gradients
z.backward()

# Access gradients
print(f"dz/dx: {x.grad}")  # Output: 4.0
print(f"dz/dy: {y.grad}")  # Output: 27.0
          `}
        />
      </Box>

      <Title order={2} id="basic-example" className="mt-6">
        Simple Neural Network Example
      </Title>
      <Text>
        Let's implement a basic neural network to demonstrate these concepts in practice:
      </Text>
      
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

# Create and test the model
model = SimpleNN(input_size=2, hidden_size=4, output_size=1)
x = torch.tensor([[1.0, 2.0]])  # Example input
y = model(x)

# Compute gradients
loss = y.sum()
loss.backward()

# Print parameter gradients
for name, param in model.named_parameters():
    print(f"{name} gradient shape: {param.grad.shape}")
        `}
      />
    </Stack>
  );
};

export default Introduction;