import React from 'react';
import { Container, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const PytorchMlpFundamentals = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={1} mb="lg">PyTorch MLP first steps</Title>
        
        <Text>
          A Multi-Layer Perceptron (MLP) in PyTorch consists of three core components: layers, activation functions, and the model container.
        </Text>
      </div>

      <div data-slide>
        <Title order={2} mt="xl">1. Linear Layer</Title>
        
        <Text>
          The building block of neural networks. Performs affine transformation: <InlineMath>{`y = xW^T + b`}</InlineMath>
        </Text>
        
        <CodeBlock language="python" code={`import torch
import torch.nn as nn

# Create a linear layer: input_dim=10, output_dim=5
layer = nn.Linear(in_features=10, out_features=5)

# Parameters
print(f"Weight shape: {layer.weight.shape}")  # [5, 10]
print(f"Bias shape: {layer.bias.shape}")      # [5]
print(f"Total parameters: {10*5 + 5}")        # 55`} />
      </div>

      <div data-slide>
        <Title order={2} mt="xl">2. Activation Functions</Title>
        
        <Text>
          Non-linear functions applied element-wise to introduce non-linearity:
        </Text>
        
        <CodeBlock language="python" code={`# Common activation functions
relu = nn.ReLU()        # f(x) = max(0, x)
sigmoid = nn.Sigmoid()   # f(x) = 1/(1+e^(-x))
tanh = nn.Tanh()        # f(x) = (e^x - e^(-x))/(e^x + e^(-x))

# Apply activation
x = torch.randn(2, 5)
output = relu(x)        # Negative values become 0`} />
      </div>

      <div data-slide>
        <Title order={2} mt="xl">3. Sequential Model</Title>
        
        <Text>
          Combine layers and activations into a sequential pipeline:
        </Text>
        
        <CodeBlock language="python" code={`# Define a simple 2-layer MLP
model = nn.Sequential(
    nn.Linear(784, 128),    # Input layer: 784 / 128
    nn.ReLU(),              # Activation
    nn.Linear(128, 64),     # Hidden layer: 128 / 64  
    nn.ReLU(),              # Activation
    nn.Linear(64, 10)       # Output layer: 64 / 10
)

# Forward pass
input_data = torch.randn(784)
output = model(input_data)         # Shape: [10]`} />
      </div>

      <div data-slide>
        <Title order={2} mt="xl">4. Counting Parameters</Title>
        
        <Text>
          Each linear layer has weights and biases. Total parameters determine model capacity:
        </Text>
        
        <CodeBlock language="python" code={`# Count parameters in our model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Example: 3-layer MLP
model = nn.Sequential(
    nn.Linear(784, 256),   # 784*256 + 256 = 200,960 params
    nn.ReLU(),
    nn.Linear(256, 128),   # 256*128 + 128 = 32,896 params  
    nn.ReLU(),
    nn.Linear(128, 10)     # 128*10 + 10 = 1,290 params
)

print(f"Total parameters: {count_parameters(model)}")  # 235,146`} />
      </div>

      <div data-slide>
        <Title order={2} mt="xl">5. Input and Output Shapes</Title>
        
        <Text>
          Understanding tensor dimensions through the network:
        </Text>
        
        <Text mt="md">
          <strong>Example 1: Single observation</strong>
        </Text>
        <CodeBlock language="python" code={`input_dim = 784  # e.g., 28x28 flattened image
x = torch.randn(input_dim)
print(f"Shape: {x.shape}")  # Shape: torch.Size([784])`} />
        
        <Text mt="md">
          <strong>Example 2: 5 observations</strong>
        </Text>
        <CodeBlock language="python" code={`x = torch.randn(5, input_dim)
print(f"Shape: {x.shape}")  # Shape: torch.Size([5, 784])
# 5 observations, each with 784 features`} />
        
        <Text mt="md">
          <strong>Example 3: 100 observations</strong>
        </Text>
        <CodeBlock language="python" code={`x = torch.randn(100, input_dim)
print(f"Shape: {x.shape}")  # Shape: torch.Size([100, 784])
# 100 observations, each with 784 features`} />
        
        <Text mt="md">
          <strong>Example 4: 3D tensor with shape (20, 100, input_dim)</strong>
        </Text>
        <CodeBlock language="python" code={`# Can represent 20 batches of 100 observations each
x = torch.randn(20, 100, input_dim)
print(f"Input shape: {x.shape}")  # Shape: torch.Size([20, 100, 784])`} />
        
        <Text mt="md">
          <strong>How dimensions transform through the network:</strong>
        </Text>
        <CodeBlock language="python" code={`# Define layers
layer1 = nn.Linear(784, 256)
layer2 = nn.Linear(256, 10)

# Process 2D tensor (50 observations)
x = torch.randn(50, 784)
x = layer1(x)  # [50, 784] → [50, 256]
x = layer2(x)  # [50, 256] → [50, 10]
print(f"2D output: {x.shape}")  # torch.Size([50, 10])

# Process 3D tensor (20 batches of 100 observations)
x = torch.randn(20, 100, 784)
x = layer1(x)  # [20, 100, 784] → [20, 100, 256]
x = layer2(x)  # [20, 100, 256] → [20, 100, 10]
print(f"3D output: {x.shape}")  # torch.Size([20, 100, 10])

# Note: Only the last dimension (features) changes
# All other dimensions remain constant`} />
      </div>

    </Container>
  );
};

export default PytorchMlpFundamentals;