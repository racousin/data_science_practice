import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const AdvancedGradientMechanics = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        <div id="gradient-flow">
          <Title order={1} mb="xl">
            Advanced Gradient Mechanics
          </Title>
          <Text size="xl" className="mb-6">
            Gradient Flow & Vanishing/Exploding Gradients
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} mb="md">Understanding Gradient Flow</Title>
            <Text className="mb-4">
              Gradient flow refers to how gradients propagate through network layers during backpropagation.
              Poor gradient flow can lead to vanishing or exploding gradients.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Vanishing Gradients</Title>
                  <CodeBlock language="python" code={`import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DeepNetwork(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.Sigmoid())  # Problematic activation
        layers.append(nn.Linear(100, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Analyze gradient magnitudes
model = DeepNetwork(num_layers=10)
x = torch.randn(32, 100)
y = torch.randn(32, 1)

output = model(x)
loss = ((output - y)**2).mean()
loss.backward()

# Check gradient norms by layer
grad_norms = []
for i, param in enumerate(model.parameters()):
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        print(f"Layer {i}: grad norm = {grad_norm:.6f}")

print(f"Gradient ratio (first/last): {grad_norms[0]/grad_norms[-1]:.2e}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Exploding Gradients</Title>
                  <CodeBlock language="python" code={`# Demonstrate exploding gradients
class UnstableNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(20)
        ])
        # Initialize with large weights
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0, std=2.0)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x.sum()

model = UnstableNetwork()
x = torch.randn(1, 10)

# Monitor gradient explosion
try:
    for step in range(10):
        model.zero_grad()
        output = model(x)
        output.backward()
        
        # Check for gradient explosion
        total_norm = 0
        for param in model.parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        print(f"Step {step}: Gradient norm = {total_norm:.2e}")
        
        if total_norm > 1e6:
            print("Gradient explosion detected!")
            break
            
except RuntimeError as e:
    print(f"Runtime error: {e}")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>


       

      </Stack>
    </Container>
  );
};

export default AdvancedGradientMechanics;