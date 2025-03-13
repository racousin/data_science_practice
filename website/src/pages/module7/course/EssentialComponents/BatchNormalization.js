import React from 'react';
import { Title, Text, Stack, List, Alert, Divider, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const BatchNormalization = () => {
  return (
    <Stack spacing="md">
      <Text>
        Batch Normalization (BatchNorm) is a technique introduced by Sergey Ioffe and Christian Szegedy in 2015
        that normalizes the intermediate activations of neural networks, significantly improving training 
        stability and speed. It addresses the internal covariate shift problem by normalizing layer inputs.
      </Text>

      <Title order={3}>The Problem: Internal Covariate Shift</Title>
      <Text>
        As parameters of earlier layers change during training, the distribution of inputs to subsequent layers 
        also changes. This phenomenon, called internal covariate shift, makes training deeper networks difficult because:
      </Text>
      <List>
        <List.Item>Each layer must continuously adapt to changing input distributions</List.Item>
        <List.Item>It requires lower learning rates and careful parameter initialization</List.Item>
        <List.Item>It slows down convergence</List.Item>
      </List>

      <Title order={3}>Implementation Formulation</Title>
      <Text>
        Consider a mini-batch of activations at some layer: <InlineMath>{`\\mathcal{B} = \\{x_1, x_2, ..., x_m\\}`}</InlineMath>
      </Text>

      <Title order={4}>Step 1: Calculate Mini-batch Statistics</Title>
      <Text>
        First, we calculate the mean and variance of the mini-batch:
      </Text>
      <BlockMath>
        {`\\mu_\\mathcal{B} = \\frac{1}{m}\\sum_{i=1}^m x_i`}
      </BlockMath>
      <BlockMath>
        {`\\sigma_\\mathcal{B}^2 = \\frac{1}{m}\\sum_{i=1}^m (x_i - \\mu_\\mathcal{B})^2`}
      </BlockMath>

      <Title order={4}>Step 2: Normalize the Activations</Title>
      <Text>
        Next, we normalize each activation:
      </Text>
      <BlockMath>
        {`\\hat{x}_i = \\frac{x_i - \\mu_\\mathcal{B}}{\\sqrt{\\sigma_\\mathcal{B}^2 + \\epsilon}}`}
      </BlockMath>
      <Text>
        Where <InlineMath>{`\\epsilon`}</InlineMath> is a small constant (e.g., 1e-5) added for numerical stability.
      </Text>
      <Alert color="blue">
        <strong>Important:</strong> At this point, the normalized values <InlineMath>{`\\hat{x}_i`}</InlineMath> will have:
        <List>
          <List.Item>Mean = 0 (centered)</List.Item>
          <List.Item>Variance = 1</List.Item>
        </List>
        This normalization happens independently for each new mini-batch during training.
      </Alert>

      <Title order={4}>Step 3: Scale and Shift (Learnable Parameters)</Title>
      <Text>
        Simply normalizing might reduce the representational power of the network. BatchNorm introduces 
        two learnable parameters per activation:
      </Text>
      <BlockMath>
        {`y_i = \\gamma \\hat{x}_i + \\beta`}
      </BlockMath>
      <Text>
        These parameters <InlineMath>{`\\gamma`}</InlineMath> (scale) and <InlineMath>{`\\beta`}</InlineMath> (shift) allow the 
        network to learn to undo the normalization if necessary, or to find the optimal transformation.
      </Text>

      <Title order={3}>Training vs. Inference</Title>
      <Text>
        There are key differences in how BatchNorm operates during training versus inference:
      </Text>

      <Title order={4}>Training Phase</Title>
      <Text>
        During training, for each mini-batch:
      </Text>
      <List>
        <List.Item>We calculate and use that batch's mean and variance for normalization</List.Item>
        <List.Item>We update running estimates of the global statistics:</List.Item>

      <BlockMath>
        {`E[x] \\leftarrow \\alpha \\cdot E[x] + (1 - \\alpha) \\cdot \\mu_\\mathcal{B}`}
      </BlockMath>
      <BlockMath>
        {`Var[x] \\leftarrow \\alpha \\cdot Var[x] + (1 - \\alpha) \\cdot \\sigma_\\mathcal{B}^2`}
      </BlockMath>
      <Text>
        Where <InlineMath>{`\\alpha`}</InlineMath> is the momentum (typically 0.9 or 0.99).
      </Text>
      <List.Item>We update the scale <InlineMath>{`\\gamma`}</InlineMath> and shift <InlineMath>{`\\beta`}</InlineMath> parameters using gradients:</List.Item>
      <BlockMath>
        {`\\frac{\\partial \\mathcal{L}}{\\partial \\gamma} = \\sum_{i=1}^m \\frac{\\partial \\mathcal{L}}{\\partial y_i} \\cdot \\hat{x}_i`}
      </BlockMath>
      <BlockMath>
        {`\\frac{\\partial \\mathcal{L}}{\\partial \\beta} = \\sum_{i=1}^m \\frac{\\partial \\mathcal{L}}{\\partial y_i}`}
      </BlockMath>
      </List>

      <Alert color="green">
        <strong>Key point:</strong> During training, the mini-batch statistics are used for normalization, while running 
        statistics are only updated for later use during inference.
      </Alert>

      <Title order={4}>Inference Phase</Title>
      <Text>
        During inference (testing), we don't have mini-batches, or we might process examples one by one. 
        Here we use the stored running statistics:
      </Text>
      <BlockMath>
        {`\\hat{x} = \\frac{x - E[x]}{\\sqrt{Var[x] + \\epsilon}}`}
      </BlockMath>
      <BlockMath>
        {`y = \\gamma \\hat{x} + \\beta`}
      </BlockMath>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

# Define a network with BatchNorm
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply BatchNorm before activation
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Switch between training and inference modes
model = Net()

# Training mode: use mini-batch statistics
model.train()
train_predictions = model(train_data)

# Inference mode: use running statistics
model.eval()
with torch.no_grad():
    test_predictions = model(test_data)`}
      />

    </Stack>
  );
};

export default BatchNormalization;