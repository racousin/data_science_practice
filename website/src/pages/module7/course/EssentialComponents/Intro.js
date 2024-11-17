import React from 'react';
import { Title, Text, Stack, Grid, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const Intro = () => {

  return (
    <Stack spacing="xl">
      <Title order={3} id="introduction">Simple Example</Title>

      <CodeBlock
        language="python"
        code={`
import torch
import numpy as np

# Generate synthetic regression data
X = np.linspace(-5, 5, 1000).reshape(-1, 1)
y = 0.2 * X**2 + 0.5 * X + 2 + np.random.normal(0, 0.2, X.shape)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)
        `}
      />


      <Text>
        To solve this problem, we can create a simple neural network. Here's a basic architecture:
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1, 32),    # Input layer → 32 neurons
    nn.ReLU(),           # Activation function
    nn.Linear(32, 64),   # Hidden layer with 64 neurons
    nn.ReLU(),           # Another activation
    nn.Linear(64, 32),   # Hidden layer with 32 neurons
    nn.ReLU(),           # Another activation
    nn.Linear(32, 1)     # Output layer
)
        `}
      />

      <Text>
        Training this network involves showing it examples and adjusting its parameters to minimize errors:
      </Text>

      <CodeBlock
        language="python"
        code={`
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        `}
      />

      <Title order={3} mt="xl">Common Learning Issues</Title>
      
      <Grid grow>
        <Grid.Col span={6}>
          <Alert title="Vanishing Gradients" color="red">
            When gradients become too small during backpropagation, deeper layers learn very slowly or not at all.
            This often occurs with certain activation functions or deep architectures.
          </Alert>
        </Grid.Col>
        <Grid.Col span={6}>
          <Alert title="Exploding Gradients" color="orange">
            Gradients can grow exponentially large, causing unstable training and making the model unable to learn.
            Proper weight initialization and gradient clipping help prevent this.
          </Alert>
        </Grid.Col>
        <Grid.Col span={6}>
          <Alert title="Overfitting" color="yellow">
            The model learns the training data too well, including noise, and fails to generalize to new data.
            Regularization techniques help combat this issue.
          </Alert>
        </Grid.Col>
        <Grid.Col span={6}>
          <Alert title="Poor Convergence" color="blue">
            The model might learn slowly or get stuck in poor solutions. This can be addressed through proper
            learning rate scheduling and optimization techniques.
          </Alert>
        </Grid.Col>
      </Grid>

      <Text mt="xl">
        These components - activation functions, weight initialization, optimization techniques, and regularization methods - 
        work together to address these challenges.
      </Text>
    </Stack>
  );
};

export default Intro;