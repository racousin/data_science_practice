import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const PyTorchNNModule = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            PyTorch nn.Module
          </Title>
          <Text size="xl" className="text-center mb-6">
            Building Neural Networks with PyTorch
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                nn.Module is the base class for all neural network modules in PyTorch.
                It provides automatic parameter management, device handling, and training/evaluation modes.
              </Text>
              <List>
                <List.Item>Automatic parameter registration</List.Item>
                <List.Item>Gradient computation support</List.Item>
                <List.Item>Device management (CPU/GPU)</List.Item>
                <List.Item>Training/evaluation mode switching</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        <div data-slide className="min-h-[500px]" id="module-basics">
          <Title order={2} mb="xl">Module Basics</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Basic nn.Module Structure</Title>
                <Code block language="python">{`import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Initialize parent class
        
        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Create and use model
model = MyModel(784, 128, 10)
x = torch.randn(32, 784)
output = model(x)  # Calls forward() automatically`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Parameter Management</Title>
                <Code block language="python">{`# Automatic parameter registration
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# All parameters accessible
params = list(model.parameters())
print(f"Total parameters: {sum(p.numel() for p in params)}")

# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training/evaluation modes
model.train()    # Set to training mode
model.eval()     # Set to evaluation mode

# Check current mode
print(f"Training mode: {model.training}")

# Save and load models
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default PyTorchNNModule;