import React from 'react';
import { Title, Text, Stack, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const CustomLoss = () => {
  return (
    <Stack spacing="md">
      <Title order={3}>PyTorch Built-in Loss Functions</Title>
      <Text>
        PyTorch provides common loss functions in <Code>torch.nn</Code>:
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

# Classification losses
criterion = nn.CrossEntropyLoss()      # Multi-class classification
criterion = nn.BCELoss()               # Binary classification (with sigmoid)
criterion = nn.BCEWithLogitsLoss()     # Binary classification (combines sigmoid + BCE)
criterion = nn.NLLLoss()               # Negative log likelihood

# Regression losses  
criterion = nn.MSELoss()               # Mean squared error
criterion = nn.L1Loss()                # Mean absolute error
criterion = nn.SmoothL1Loss()          # Huber loss

# Other losses
criterion = nn.KLDivLoss()             # Kullback-Leibler divergence
criterion = nn.CosineEmbeddingLoss()   # Cosine similarity
criterion = nn.TripletMarginLoss()     # Triplet loss for embeddings`}
      />

      <Title order={3} mt="md">Custom Loss Example</Title>
      <Text>
        Create custom losses by subclassing <Code>nn.Module</Code>:
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """MSE loss with sample weights"""
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets, weights=None):
        squared_diff = (predictions - targets) ** 2
        if weights is not None:
            squared_diff = squared_diff * weights
        return torch.mean(squared_diff)

# Usage
model = YourModel()
criterion = WeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters())

for inputs, targets, sample_weights in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, targets, sample_weights)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
      />
    </Stack>
  );
};

export default CustomLoss;