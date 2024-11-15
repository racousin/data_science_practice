import React from 'react';
import { Title, Text, Stack, Grid, Box, List, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const Backpropagation = () => {
  return (
    <Stack spacing="xl" className="w-full">
      {/* Loss Function Introduction */}
      <Title order={1} id="backpropagation">Backpropagation: The Learning Algorithm</Title>
      
      <Stack spacing="md">
        <Title order={2} id="loss-function">Loss Function and Parameter Updates</Title>
        <Text>
          Neural networks learn by minimizing a loss function L that measures the discrepancy between predictions and desired outputs. The objective is to find parameters θ (weights and biases) that minimize this loss.
        </Text>

        <Box className="p-4 border rounded">
          <Title order={4}>Common Loss Functions</Title>
          <BlockMath>{`
            \\begin{aligned}
            \\text{MSE (Regression):} & \\quad L = \\frac{1}{N}\\sum_{i=1}^N (y_i - \\hat{y}_i)^2 \\\\
            \\text{Cross-Entropy (Classification):} & \\quad L = -\\frac{1}{N}\\sum_{i=1}^N \\sum_{c=1}^C y_{i,c}\\log(\\hat{y}_{i,c})
            \\end{aligned}
          `}</BlockMath>
          
          <Text mt="md">Parameter updates follow the gradient descent rule:</Text>
          <BlockMath>{`\\theta_{t+1} = \\theta_t - \\alpha \\nabla_\\theta L`}</BlockMath>
          <Text size="sm">where α is the learning rate</Text>
        </Box>
      </Stack>

      {/* Gradient Computation Methods */}
      <Stack spacing="md">
  <Title order={2} id="gradient-methods">Gradient Computation Methods</Title>
  
  <Text>
    To train neural networks, we need to compute gradients of the loss with respect to parameters. 
    Let's explore different approaches and understand why automatic differentiation is preferred.
  </Text>

  <Table withTableBorder withColumnBorders>
    <Table.Thead>
      <Table.Tr>
        <Table.Th>Method</Table.Th>
        <Table.Th>Principle</Table.Th>
        <Table.Th>Limitations for Neural Networks</Table.Th>
      </Table.Tr>
    </Table.Thead>
    <Table.Tbody>
      <Table.Tr>
        <Table.Td>
          <strong>Symbolic Differentiation</strong>
        </Table.Td>
        <Table.Td>
          Applies differentiation rules to derive exact mathematical expressions
          <BlockMath>{`\\frac{d}{dx}(x^2) = 2x`}</BlockMath>
        </Table.Td>
        <Table.Td>
          • Expression complexity explodes with network size
          • Impractical for dynamic computations in deep networks
          • Memory intensive for large expressions
        </Table.Td>
      </Table.Tr>

      <Table.Tr>
        <Table.Td>
          <strong>Finite Differences</strong>
        </Table.Td>
        <Table.Td>
          Approximates derivatives using small perturbations
          <BlockMath>{`\\frac{\\partial L}{\\partial \\theta} \\approx \\frac{L(\\theta + h) - L(\\theta)}{h}`}</BlockMath>
        </Table.Td>
        <Table.Td>
          • Requires O(n) evaluations for n parameters
          • Numerically unstable (sensitive to h)
          • Computationally expensive for millions of parameters
        </Table.Td>
      </Table.Tr>

      <Table.Tr>
        <Table.Td>
          <strong>Automatic Differentiation</strong>
        </Table.Td>
        <Table.Td>
          Builds computation graph and applies chain rule systematically
          <BlockMath>{`\\frac{\\partial z}{\\partial x} = \\sum_i \\frac{\\partial z}{\\partial y_i}\\frac{\\partial y_i}{\\partial x}`}</BlockMath>
        </Table.Td>
        <Table.Td>
          ✓ Efficient computation
          ✓ Exact gradients
          ✓ Handles dynamic computations
        </Table.Td>
      </Table.Tr>
    </Table.Tbody>
  </Table>

  <Title order={3} mt="lg">Automatic Differentiation Modes</Title>
  
  <Grid gutter="xl">
    <Grid.Col span={6}>
      <Box className="p-4 bg-blue-50 rounded">
        <Title order={4}>Forward Mode</Title>
        <Text>
          Propagates derivatives forward through computation graph from inputs to outputs.
        </Text>
        <List>
          <List.Item>Efficient for functions with few inputs and many outputs</List.Item>
          <List.Item>Computes one input derivative at a time</List.Item>
          <List.Item>O(n) complexity for n inputs</List.Item>
        </List>
        <BlockMath>{`\\dot{y} = \\frac{\\partial f}{\\partial x}\\dot{x}`}</BlockMath>
      </Box>
    </Grid.Col>

    <Grid.Col span={6}>
      <Box className="p-4 bg-green-50 rounded">
        <Title order={4}>Reverse Mode (Backpropagation)</Title>
        <Text>
          Propagates derivatives backward from outputs to inputs.
        </Text>
        <List>
          <List.Item>Efficient for functions with many inputs and few outputs</List.Item>
          <List.Item>Computes all input derivatives in one pass</List.Item>
          <List.Item>O(1) complexity regardless of input size</List.Item>
        </List>
        <BlockMath>{`\\bar{x} = \\bar{y}\\frac{\\partial f}{\\partial x}`}</BlockMath>
      </Box>
    </Grid.Col>
  </Grid>

  <Text mt="md">
    Backpropagation is reverse-mode automatic differentiation specialized for scalar outputs (loss function),
    making it ideal for neural networks with millions of parameters but single scalar loss.
  </Text>
</Stack>
      {/* Backpropagation Details */}
      <Stack spacing="md">
        <Title order={2} id="backprop-details">Backpropagation: A Reverse-Mode Automatic Differentiation</Title>
        
        <Text>
          Backpropagation efficiently computes gradients by decomposing the computation graph and applying the chain rule backward from the output to inputs.
        </Text>
  <Title order={2} id="notation">Mathematical Notation</Title>
  
  <Table withTableBorder withColumnBorders>
    <Table.Thead>
      <Table.Tr>
        <Table.Th>Symbol</Table.Th>
        <Table.Th>Description</Table.Th>
        <Table.Th>Dimension</Table.Th>
      </Table.Tr>
    </Table.Thead>
    <Table.Tbody>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`l`}</InlineMath>
        </Table.Td>
        <Table.Td>Layer index</Table.Td>
        <Table.Td>Scalar</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`n_l`}</InlineMath>
        </Table.Td>
        <Table.Td>Number of neurons in layer l</Table.Td>
        <Table.Td>Scalar</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}`}</InlineMath>
        </Table.Td>
        <Table.Td>Weight matrix for layer l</Table.Td>
        <Table.Td>Matrix</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`b^{(l)} \\in \\mathbb{R}^{n_l}`}</InlineMath>
        </Table.Td>
        <Table.Td>Bias vector for layer l</Table.Td>
        <Table.Td>Vector</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`z^{(l)} \\in \\mathbb{R}^{n_l}`}</InlineMath>
        </Table.Td>
        <Table.Td>Pre-activation vector in layer l</Table.Td>
        <Table.Td>Vector</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`a^{(l)} \\in \\mathbb{R}^{n_l}`}</InlineMath>
        </Table.Td>
        <Table.Td>Activation vector in layer l</Table.Td>
        <Table.Td>Vector</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`f`}</InlineMath>
        </Table.Td>
        <Table.Td>Activation function</Table.Td>
        <Table.Td>Scalar function</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`\\delta^{(l)} \\in \\mathbb{R}^{n_l}`}</InlineMath>
        </Table.Td>
        <Table.Td>Error term vector for layer l</Table.Td>
        <Table.Td>Vector</Table.Td>
      </Table.Tr>
      <Table.Tr>
        <Table.Td>
          <InlineMath>{`L`}</InlineMath>
        </Table.Td>
        <Table.Td>Loss function</Table.Td>
        <Table.Td>Scalar</Table.Td>
      </Table.Tr>
    </Table.Tbody>
  </Table>

  <Title order={3} mt="lg">Key Equations</Title>
  <Grid gutter="xl">
    <Grid.Col span={6}>
      <Box className="p-4 bg-gray-50 rounded">
        <Title order={4}>Layer Operations</Title>
        <BlockMath>{`
          \\begin{aligned}
          z^{(l)} &= W^{(l)}a^{(l-1)} + b^{(l)} \\\\
          a^{(l)} &= f(z^{(l)})
          \\end{aligned}
        `}</BlockMath>
      </Box>
    </Grid.Col>
    <Grid.Col span={6}>
      <Box className="p-4 bg-gray-50 rounded">
        <Title order={4}>Gradient Flow</Title>
        <BlockMath>{`
          \\begin{aligned}
          \\frac{\\partial L}{\\partial W^{(l)}} &= \\frac{\\partial L}{\\partial z^{(l)}}\\frac{\\partial z^{(l)}}{\\partial W^{(l)}} \\\\
          \\frac{\\partial L}{\\partial b^{(l)}} &= \\frac{\\partial L}{\\partial z^{(l)}}
          \\end{aligned}
        `}</BlockMath>
      </Box>
    </Grid.Col>
  </Grid>
        <Box className="p-4 border rounded">
          <Title order={4}>Forward Pass</Title>
          <BlockMath>{`
            \\begin{aligned}
            z^{(l)}_j &= \\sum_i w^{(l)}_{ij} a^{(l-1)}_i + b^{(l)}_j \\\\
            a^{(l)}_j &= f(z^{(l)}_j)
            \\end{aligned}
          `}</BlockMath>
        </Box>

        <Box className="p-4 border rounded">
          <Title order={4}>Backward Pass</Title>
          <Text>Define the error term δ for each layer:</Text>
          <BlockMath>{`
            \\begin{aligned}
            \\delta^{(L)}_j &= \\frac{\\partial L}{\\partial z^{(L)}_j} = \\frac{\\partial L}{\\partial a^{(L)}_j}f'(z^{(L)}_j) \\\\
            \\delta^{(l)}_j &= \\sum_k \\delta^{(l+1)}_k w^{(l+1)}_{jk} f'(z^{(l)}_j)
            \\end{aligned}
          `}</BlockMath>

          <Text mt="md">Gradient computations:</Text>
          <BlockMath>{`
            \\begin{aligned}
            \\frac{\\partial L}{\\partial w^{(l)}_{ij}} &= a^{(l-1)}_i \\delta^{(l)}_j \\\\
            \\frac{\\partial L}{\\partial b^{(l)}_j} &= \\delta^{(l)}_j
            \\end{aligned}
          `}</BlockMath>
        </Box>

        <Box className="p-4 border rounded">
          <Title order={4}>Computational Graph Example</Title>
          <Text>Consider a two-layer network with MSE loss:</Text>
          <BlockMath>{`
            \\begin{aligned}
            &\\text{Forward:} \\\\
            &z^{(1)} = W^{(1)}x + b^{(1)} \\\\
            &a^{(1)} = f(z^{(1)}) \\\\
            &z^{(2)} = W^{(2)}a^{(1)} + b^{(2)} \\\\
            &\\hat{y} = a^{(2)} = f(z^{(2)}) \\\\
            &L = \\frac{1}{2}(\\hat{y} - y)^2
            \\end{aligned}
          `}</BlockMath>
          
          <Text mt="md">Backward flow of gradients:</Text>
          <BlockMath>{`
            \\begin{aligned}
            &\\frac{\\partial L}{\\partial \\hat{y}} = \\hat{y} - y \\\\
            &\\frac{\\partial L}{\\partial z^{(2)}} = \\frac{\\partial L}{\\partial \\hat{y}}f'(z^{(2)}) \\\\
            &\\frac{\\partial L}{\\partial W^{(2)}} = \\frac{\\partial L}{\\partial z^{(2)}}(a^{(1)})^T \\\\
            &\\frac{\\partial L}{\\partial a^{(1)}} = (W^{(2)})^T\\frac{\\partial L}{\\partial z^{(2)}} \\\\
            &\\frac{\\partial L}{\\partial z^{(1)}} = \\frac{\\partial L}{\\partial a^{(1)}}f'(z^{(1)})
            \\end{aligned}
          `}</BlockMath>
        </Box>

        {/* PyTorch Implementation */}
        <Title order={2} id="implementation">Implementation in PyTorch</Title>
        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn

# Define a simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)  # 2 inputs, 3 hidden units
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(3, 1)  # 3 hidden units, 1 output
        
    def forward(self, x):
        # Forward pass with intermediate values
        z1 = self.linear1(x)
        a1 = self.activation(z1)
        z2 = self.linear2(a1)
        return z2

# Example usage
x = torch.tensor([[1., 2.]], requires_grad=True)
y = torch.tensor([[3.]], requires_grad=True)

model = SimpleNet()
criterion = nn.MSELoss()

# Forward pass
output = model(x)
loss = criterion(output, y)

# Backward pass
loss.backward()

# Print gradients
for name, param in model.named_parameters():
    print(f"{name} grad:", param.grad)
`}
        />
      </Stack>

  <Title order={2} id="training-loop">Training Loop Implementation</Title>
  <Text>
    Let's implement a simple training loop to see backpropagation in action. We'll train a network on a basic regression task.
  </Text>

  <CodeBlock
    language="python"
    code={`
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate synthetic data
X = torch.linspace(-5, 5, 100).reshape(-1, 1)
y = 0.2 * X**2 + 0.5 * X + 2 + torch.randn_like(X) * 0.2

# Define a simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize model and training parameters
model = SimpleNet()
criterion = nn.MSELoss()
learning_rate = 0.01
n_epochs = 100

# Training loop
losses = []
for epoch in range(n_epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Manual parameter update using gradients
    with torch.no_grad():  # Disable gradient tracking for updates
        for param in model.parameters():
            param.data -= learning_rate * param.grad
            
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Visualize results
plt.figure(figsize=(12, 4))

# Plot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')

# Plot 2: Predictions vs True Values
plt.subplot(1, 2, 2)
with torch.no_grad():
    y_pred = model(X)
plt.scatter(X.numpy(), y.numpy(), label='True Values', alpha=0.5)
plt.scatter(X.numpy(), y_pred.numpy(), label='Predictions', alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Predictions vs True Values')
plt.legend()

plt.tight_layout()
plt.show()
`}
  />

  <Box className="p-4 bg-gray-50 rounded">
    <Title order={4}>Key Points in the Training Loop</Title>
    <List>
      <List.Item>
        <strong>Zero Gradients:</strong> Call <InlineMath>{`\\text{model.zero\\_grad()}`}</InlineMath> before backward pass to clear previous gradients
      </List.Item>
      <List.Item>
        <strong>Backward Pass:</strong> <InlineMath>{`\\text{loss.backward()}`}</InlineMath> computes gradients for all parameters
      </List.Item>
      <List.Item>
        <strong>Manual Updates:</strong> Update parameters using gradient descent rule: <InlineMath>{`\\theta = \\theta - \\alpha \\nabla_\\theta L`}</InlineMath>
      </List.Item>
      <List.Item>
        <strong>Gradient Tracking:</strong> Use <InlineMath>{`\\text{torch.no\\_grad()}`}</InlineMath> context for parameter updates
      </List.Item>
    </List>
  </Box>

  <Text>
    This example shows how backpropagation computes gradients that are then used to update the model parameters. In practice, we would use an optimizer like SGD or Adam instead of manual updates (see next section).
  </Text>
</Stack>
  );
};

export default Backpropagation;