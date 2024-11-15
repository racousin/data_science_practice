import React from 'react';
import { Title, Text, Stack, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

const NNLearningIssues = () => {
  return (
    <Stack spacing="lg" w="100%">
      <Title order={1} id="nn-learning-issues">Neural Network Learning Issues</Title>
      
      {/* Simple NN Example */}
      <Stack spacing="md">
        <Title order={2} id="simple-example">Simple Neural Network Example</Title>
        <Text size="sm">
          Let's start with a basic neural network to classify MNIST digits:
        </Text>
        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),    # Input layer (28x28 = 784)
            nn.ReLU(),              # Activation
            nn.Linear(256, 128),    # Hidden layer 1
            nn.ReLU(),              # Activation
            nn.Linear(128, 10)      # Output layer (10 digits)
        )
    
    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

# Training setup
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()`}
        />
      </Stack>

      {/* Common Issues */}
      <Title order={2} id="common-issues" mt="xl">Common Learning Issues</Title>
      
      <Stack spacing="md">
        {/* Vanishing Gradients */}
        <div>
          <Title order={3}>Vanishing Gradients</Title>
          <Text size="sm">
            Occurs when gradients become extremely small as they propagate backwards through the network.
          </Text>
          <BlockMath>{`\\frac{\\partial L}{\\partial w} \\approx 0`}</BlockMath>
          <Text size="sm" c="dimmed">
            • Commonly seen in deep networks with sigmoid/tanh activations
            • Earlier layers learn very slowly or not at all
            • Network struggles to capture long-range dependencies
          </Text>
        </div>

        {/* Exploding Gradients */}
        <div>
          <Title order={3}>Exploding Gradients</Title>
          <Text size="sm">
            Occurs when gradients become extremely large, causing unstable updates.
          </Text>
          <BlockMath>{`\\frac{\\partial L}{\\partial w} \\rightarrow \\infty`}</BlockMath>
          <Text size="sm" c="dimmed">
            • Results in numerical overflow
            • Causes unstable training and NaN values
            • Often occurs in recurrent networks or very deep architectures
          </Text>
        </div>

        {/* Dead Neurons */}
        <div>
          <Title order={3}>Dead Neurons</Title>
          <Text size="sm">
            Neurons that always output zero regardless of input.
          </Text>
          <Code>output = 0 for all inputs</Code>
          <Text size="sm" c="dimmed">
            • Common with ReLU activation when learning rate is too high
            • Reduces effective capacity of the network
            • Can affect large portions of the network
          </Text>
        </div>

        {/* Slow Convergence */}
        <div>
          <Title order={3}>Slow Convergence</Title>
          <Text size="sm">
            Network takes an excessive number of epochs to reach optimal performance.
          </Text>
          <Text size="sm" c="dimmed">
            • Poor learning rate choice
            • Inefficient optimization algorithm
            • Bad weight initialization
            • Poorly scaled input data
          </Text>
        </div>

        {/* Overfitting */}
        <div>
          <Title order={3}>Overfitting</Title>
          <Text size="sm">
            Model performs well on training data but fails to generalize to new data.
          </Text>
          <Text size="sm" c="dimmed">
            • High training accuracy, low validation accuracy
            • Model memorizes training data
            • Complex patterns learned don't generalize
          </Text>
        </div>

        {/* Other Issues */}
        <div>
          <Title order={3}>Other Common Issues</Title>
          <Stack spacing="xs">
            <Text size="sm">• Internal Covariate Shift: Distribution of layer inputs changes during training</Text>
            <Text size="sm">• Gradient Saturation: Activation functions operating in saturated regions</Text>
            <Text size="sm">• Poor Weight Initialization: Initial weights causing activation collapse</Text>
            <Text size="sm">• Imbalanced Data: Uneven class distribution affecting learning</Text>
          </Stack>
        </div>
      </Stack>

      <Text mt="xl" c="dimmed" size="sm">
        Note: Solutions to these issues will be covered in subsequent chapters on regularization,
        optimization techniques, and best practices.
      </Text>
    </Stack>
  );
};

export default NNLearningIssues;