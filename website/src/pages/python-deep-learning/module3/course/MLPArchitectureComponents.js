import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const MLPArchitectureComponents = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        <div id="multilayer-perceptron">
          <Title order={1} mb="xl">
            MLP Architecture & Components
          </Title>
          <Text size="xl" className="mb-6">
            Multilayer Perceptron Mathematics
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Mathematical Foundation of MLPs</Title>
            <Text className="mb-4">
              A multilayer perceptron (MLP) is a feedforward neural network with multiple layers of neurons.
              Each layer applies an affine transformation followed by a nonlinear activation function.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">MLP Forward Pass</Title>
                  <CodeBlock language="python" code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        self.activation = getattr(F, activation)
    
    def forward(self, x):
        # Forward pass through hidden layers
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

# Create and test MLP
mlp = MLP(input_dim=784, hidden_dims=[512, 256, 128], output_dim=10)
x = torch.randn(32, 784)  # Batch of 32 samples
output = mlp(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of parameters: {sum(p.numel() for p in mlp.parameters())}")

# Manual forward pass for understanding
def manual_forward(x, weights, biases, activations):
    """Manual implementation of MLP forward pass"""
    current = x
    
    for i, (W, b, activation) in enumerate(zip(weights, biases, activations)):
        # Affine transformation: z = xW^T + b
        z = torch.mm(current, W.t()) + b
        
        # Apply activation (except for output layer)
        if activation is not None:
            current = activation(z)
        else:
            current = z
    
    return current

# Extract weights and biases
weights = [layer.weight for layer in mlp.layers]
biases = [layer.bias for layer in mlp.layers]
activations = [F.relu] * (len(mlp.layers) - 1) + [None]  # No activation for output

manual_output = manual_forward(x, weights, biases, activations)
print(f"Manual forward pass matches: {torch.allclose(output, manual_output)}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Parameter Counting</Title>
                  <CodeBlock language="python" code={`def count_parameters(model):
    """Count parameters in a neural network"""
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        print(f"{name}: {list(param.shape)} -> {num_params:,} parameters")
    
    return total_params, trainable_params

# Analyze MLP parameters
total, trainable = count_parameters(mlp)
print(f"\\nTotal parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")

# Memory usage estimation
def estimate_memory_usage(model, input_shape, batch_size=1):
    """Estimate memory usage for model and activations"""
    # Model parameters (weights and biases)
    param_memory = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    
    # Forward pass activations
    x = torch.randn(batch_size, *input_shape)
    activation_memory = 0
    
    with torch.no_grad():
        current = x
        activation_memory += current.numel() * 4
        
        for layer in model.layers[:-1]:
            current = F.relu(layer(current))
            activation_memory += current.numel() * 4
        
        current = model.layers[-1](current)
        activation_memory += current.numel() * 4
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    total_memory = param_memory + activation_memory + gradient_memory
    
    print(f"Parameter memory: {param_memory / 1024**2:.2f} MB")
    print(f"Activation memory: {activation_memory / 1024**2:.2f} MB")
    print(f"Gradient memory: {gradient_memory / 1024**2:.2f} MB")
    print(f"Total memory: {total_memory / 1024**2:.2f} MB")
    
    return total_memory

memory_usage = estimate_memory_usage(mlp, (784,), batch_size=32)`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        <div id="universal-approximation">
          <Title order={2} className="mb-6">Universal Approximation Theorem</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Theoretical Foundation</Title>
            <Text className="mb-4">
              The Universal Approximation Theorem states that a feedforward network with a single hidden layer 
              can approximate any continuous function on a compact subset of ℝⁿ to arbitrary accuracy.
            </Text>
            
            <CodeBlock language="python" code={`# Demonstrate universal approximation with simple examples
import numpy as np
import matplotlib.pyplot as plt

def approximate_function(target_func, x_range, hidden_units=50, epochs=1000):
    """Approximate a target function using a single hidden layer MLP"""
    
    # Generate training data
    x_train = torch.linspace(x_range[0], x_range[1], 1000).unsqueeze(1)
    y_train = target_func(x_train)
    
    # Single hidden layer network
    class UniversalApproximator(nn.Module):
        def __init__(self, hidden_units):
            super().__init__()
            self.hidden = nn.Linear(1, hidden_units)
            self.output = nn.Linear(hidden_units, 1)
        
        def forward(self, x):
            h = torch.sigmoid(self.hidden(x))  # Sigmoid activation
            return self.output(h)
    
    model = UniversalApproximator(hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    return model, losses

# Test functions to approximate
def target_functions():
    return {
        'sine': lambda x: torch.sin(2 * np.pi * x),
        'polynomial': lambda x: x**3 - 2*x**2 + x,
        'step': lambda x: torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x)),
        'gaussian': lambda x: torch.exp(-x**2)
    }

# Demonstrate approximation capabilities
functions = target_functions()

for name, func in functions.items():
    print(f"\\nApproximating {name} function:")
    model, losses = approximate_function(func, (-2, 2), hidden_units=50, epochs=500)
    
    # Test approximation quality
    x_test = torch.linspace(-2, 2, 100).unsqueeze(1)
    y_true = func(x_test)
    y_pred = model(x_test)
    
    mse = F.mse_loss(y_pred, y_true)
    print(f"Final MSE: {mse:.6f}")
    
    # Compute approximation error
    max_error = torch.max(torch.abs(y_pred - y_true))
    print(f"Maximum approximation error: {max_error:.6f}")

print("\\nUniversal Approximation Theorem demonstration completed!")
print("Note: With enough hidden units and training, any continuous function can be approximated.")`} />
          </Paper>
        </div>

        <div>
          <Title order={2} className="mb-8">Summary: MLP Architecture & Components</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} className="mb-4">Key Concepts</Title>
                <List spacing="md">
                  <List.Item>MLPs use affine transformations with nonlinear activations</List.Item>
                  <List.Item>Parameter count grows quadratically with layer width</List.Item>
                  <List.Item>Universal approximation guarantees expressiveness</List.Item>
                  <List.Item>Memory usage includes parameters, activations, and gradients</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} className="mb-4">Practical Implications</Title>
                <List spacing="md">
                  <List.Item>Depth vs. width trade-offs affect representational capacity</List.Item>
                  <List.Item>Activation functions critically impact gradient flow</List.Item>
                  <List.Item>Weight initialization affects training dynamics</List.Item>
                  <List.Item>Regularization prevents overfitting in MLPs</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default MLPArchitectureComponents;