import React from 'react';
import { Title, Text, Stack, Group } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

const Optimization = () => {
  return (
    <Stack spacing="lg" w="100%">
    <Stack spacing="md">
      {/* SGD */}
      <div>
        <Title order={4}>Stochastic Gradient Descent (SGD)</Title>
        <BlockMath>{`w_{t+1} = w_t - \\eta \\nabla L(w_t)`}</BlockMath>
        <Text size="sm">Basic gradient descent with fixed learning rate η</Text>
      </div>

      {/* Momentum */}
      <div>
        <Title order={4}>SGD with Momentum</Title>
        <BlockMath>{`v_{t+1} = \\beta v_t + \\nabla L(w_t)`}</BlockMath>
        <BlockMath>{`w_{t+1} = w_t - \\eta v_{t+1}`}</BlockMath>
        <Text size="sm">Adds velocity term to dampen oscillations and accelerate convergence</Text>
      </div>

      {/* Adam */}
      <div>
        <Title order={4}>Adam (Adaptive Moment Estimation)</Title>
        <BlockMath>{`m_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\nabla L(w_t)`}</BlockMath>
        <BlockMath>{`v_t = \\beta_2 v_{t-1} + (1-\\beta_2)(\\nabla L(w_t))^2`}</BlockMath>
        <BlockMath>{`\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}`}</BlockMath>
        <BlockMath>{`\\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}`}</BlockMath>
        <BlockMath>{`w_{t+1} = w_t - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}`}</BlockMath>
        <Text size="sm">Combines momentum with adaptive learning rates per parameter</Text>
        <Text size="sm">* First moment (m_t): Tracks the mean of gradients (similar to momentum)</Text>
        <Text size="sm">* Second moment (v_t): Tracks the mean of squared gradients</Text>
        <Text size="sm">* t is the iteration counter (timestep), starting from 1</Text>
        <Text size="sm">* Bias correction terms (1-β₁ᵗ) and (1-β₂ᵗ) counteract initialization bias</Text>
      </div>
    </Stack>

      <CodeBlock
        language="python"
        code={`
import torch.optim as optim

# Initialize optimizers
model = YourModel()
sgd = optim.SGD(model.parameters(), lr=0.01)
momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Training loop example
optimizer.zero_grad()
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()`}
      />
      <Group grow align="flex-start">
        <Stack spacing="xs">
          <Title order={3}>Learning Rate (η)</Title>
          <Text size="sm">Controls step size in gradient descent</Text>
          <Text size="sm" c="dimmed">
            • High: Faster learning, risk of divergence
            • Low: Stable but slow convergence
            • Typical: 1e-4 to 1e-1
          </Text>
        </Stack>

        <Stack spacing="xs">
          <Title order={3}>Momentum (β)</Title>
          <Text size="sm">Controls influence of past gradients</Text>
          <Text size="sm" c="dimmed">
            • Higher: More momentum (0.9 typical)
            • Helps escape local minima
            • Smooths optimization trajectory
          </Text>
        </Stack>
      </Group>

    </Stack>
  );
};

export default Optimization;

// TODO ReduceLROnPlateau