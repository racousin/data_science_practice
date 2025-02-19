import React from 'react';
import { Title, Text, Stack, List, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const BatchNormalization = () => {
  return (
    <Stack spacing="md">
      <Text>
        Batch Normalization (BatchNorm) is a technique that normalizes the intermediate activations of neural networks, 
        significantly improving training stability and speed. It addresses the internal covariate shift problem by 
        normalizing layer inputs to have zero mean and unit variance.
      </Text>

      <Title order={3}>How Batch Normalization Works</Title>
      <Text>
        For each feature in a mini-batch, BatchNorm performs the following steps:
      </Text>

      <Text>1. Calculate mean and variance across the batch:</Text>
      <BlockMath>
        {`\\mu_B = \\frac{1}{m}\\sum_{i=1}^m x_i`}
      </BlockMath>
      <BlockMath>
        {`\\sigma_B^2 = \\frac{1}{m}\\sum_{i=1}^m (x_i - \\mu_B)^2`}
      </BlockMath>

      <Text>2. Normalize the values:</Text>
      <BlockMath>
        {`\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}`}
      </BlockMath>

      <Text>3. Scale and shift using learnable parameters γ and β:</Text>
      <BlockMath>
        {`y_i = \\gamma\\hat{x}_i + \\beta`}
      </BlockMath>

      <Title order={3} mt="md">Implementation in PyTorch</Title>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

# Define the sequential model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10)
)`}
      />

      <Title order={3} mt="md">Training vs. Inference Behavior</Title>
      <CodeBlock
        language="python"
        code={`
# Training mode: use batch statistics
model.train()
train_predictions = model(train_data)

# Evaluation mode: use running statistics
model.eval()
with torch.no_grad():
    test_predictions = model(test_data)`}
      />

      <Title order={3} mt="md">Key Considerations</Title>
      <List spacing="xs">
        <List.Item>
          <strong>Batch Size:</strong> BatchNorm requires sufficiently large batch sizes (typically 32+) to compute reliable statistics
        </List.Item>
        <List.Item>
          <strong>Layer Position:</strong> Typically applied after linear/conv layers but before activation functions
        </List.Item>
      </List>
    </Stack>
  );
};

export default BatchNormalization;