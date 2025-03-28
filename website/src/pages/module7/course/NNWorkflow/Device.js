import React from 'react';
import { Stack, Title, Text, Alert, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const Device = () => {
  return (
    <Stack spacing="md">

      <Title order={4}>1. Define Model Architecture</Title>
      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)`}
      />

      <Title order={4}>2. Setup Device and Model</Title>
      <CodeBlock
        language="python"
        code={`
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model and move to device
model = RegressionNet().to(device)

# Initialize loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)`}
      />

      <Title order={4}>Important Considerations</Title>
      <List>
        <List.Item>
          Always move both the model and input data to the same device
        </List.Item>
        <List.Item>
          Initialize optimizer after moving model to device
        </List.Item>
        <List.Item>
          Check device compatibility before training:
        </List.Item>
      </List>

      <CodeBlock
        language="python"
        code={`
def check_device_setup(model, device):
    # Verify model device
    print(f"Model device: {next(model.parameters()).device}")
    
    # Check available GPU memory if using CUDA
    if device.type == 'cuda':
        print(f"GPU Memory Allocated: "
              f"{torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        
check_device_setup(model, device)`}
      />
    </Stack>
  );
};

export default Device;