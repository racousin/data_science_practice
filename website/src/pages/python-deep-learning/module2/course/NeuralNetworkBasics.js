import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const NeuralNetworkBasics = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
        {/* Slide 1: Title */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Neural Network Basics
          </Title>
          <Text size="xl" className="text-center mb-6">
            From Perceptrons to Deep Networks
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" className="mb-4">
                Neural networks are computational models inspired by biological neurons.
                They consist of interconnected nodes that process and transmit information.
              </Text>
              <List>
                <List.Item>Perceptron: Single neuron model</List.Item>
                <List.Item>Multilayer Perceptron: Networks of neurons</List.Item>
                <List.Item>Deep Networks: Many hidden layers</List.Item>
                <List.Item>Universal Approximation: Can learn any function</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: The Perceptron */}
        <div data-slide className="min-h-[500px]" id="perceptron">
          <Title order={2} className="mb-6">The Perceptron</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Mathematical Model</Title>
                <Text className="mb-3">A perceptron computes:</Text>
                <BlockMath>y = f(\sum_{i=1}^n w_i x_i + b)</BlockMath>
                <Text className="mb-3">Where:</Text>
                <List size="sm">
                  <List.Item><InlineMath>x_i</InlineMath> = input features</List.Item>
                  <List.Item><InlineMath>w_i</InlineMath> = weights</List.Item>
                  <List.Item><InlineMath>b</InlineMath> = bias</List.Item>
                  <List.Item><InlineMath>f</InlineMath> = activation function</List.Item>
                </List>
              </Paper>
              
              <Paper className="p-4 bg-blue-50 mt-4">
                <Title order={4} className="mb-3">PyTorch Implementation</Title>
                <Code block language="python">{`import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        return self.activation(self.linear(x))

# Usage
model = Perceptron(input_size=2)
x = torch.randn(32, 2)  # Batch of 32 samples
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 1]`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">Geometric Interpretation</Title>
                <div className="text-center font-mono bg-white p-4 rounded mb-4">
                  <div>Decision Boundary: w₁x₁ + w₂x₂ + b = 0</div>
                  <br/>
                  <div>x₂ ^</div>
                  <div>   |</div>
                  <div>   |  • (Class 1)</div>
                  <div>   | /</div>
                  <div>   |/______ Decision Line</div>
                  <div>   /</div>
                  <div>  /   •</div>
                  <div> /  (Class 0)    x₁ →</div>
                </div>
                <Text size="sm">
                  The perceptron creates a linear decision boundary that separates two classes.
                  It can only solve linearly separable problems.
                </Text>
              </Paper>
              
              <Paper className="p-4 bg-purple-50 mt-4">
                <Title order={4} className="mb-3">Learning Algorithm</Title>
                <Code block language="python">{`# Perceptron learning rule
for epoch in range(num_epochs):
    for x, target in dataset:
        # Forward pass
        prediction = model(x)
        
        # Compute error
        error = target - prediction
        
        # Update weights
        model.weight.data += learning_rate * error * x
        model.bias.data += learning_rate * error
        
        # Modern PyTorch uses automatic differentiation:
        # loss = criterion(prediction, target)
        # loss.backward()
        # optimizer.step()`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Multilayer Perceptrons */}
        <div data-slide className="min-h-[500px]" id="multilayer">
          <Title order={2} className="mb-6">Multilayer Perceptrons (MLPs)</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-gray-50 mb-4">
                <Title order={4} className="mb-3">Architecture Overview</Title>
                <div className="text-center font-mono bg-white p-4 rounded">
                  <div>Input Layer → Hidden Layer(s) → Output Layer</div>
                  <br/>
                  <div>x₁ ──────→ h₁ ──────→ y₁</div>
                  <div>x₂ ──┐  ┌→ h₂ ──┐  ┌→ y₂</div>
                  <div>x₃ ──┼──┼─ h₃ ──┼──┤</div>
                  <div>x₄ ──┘  └→ h₄ ──┘  └→ y₃</div>
                </div>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">PyTorch MLP Implementation</Title>
                <Code block language="python">{`class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create model
model = MLP(input_size=784,    # 28x28 image
           hidden_size=128,
           output_size=10)     # 10 classes

# Forward pass
x = torch.randn(32, 784)
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Universal Approximation Theorem</Title>
                <Text className="mb-3">
                  <strong>Theorem:</strong> A feedforward network with a single hidden layer 
                  containing a finite number of neurons can approximate any continuous function 
                  on compact subsets of ℝⁿ to arbitrary accuracy.
                </Text>
                <List size="sm">
                  <List.Item>MLPs are universal function approximators</List.Item>
                  <List.Item>Depth vs Width trade-offs exist</List.Item>
                  <List.Item>Deep networks often more efficient than wide ones</List.Item>
                  <List.Item>Expressivity doesn't guarantee learnability</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* More slides would continue... */}
        
      </Stack>
    </Container>
  );
};

export default NeuralNetworkBasics;