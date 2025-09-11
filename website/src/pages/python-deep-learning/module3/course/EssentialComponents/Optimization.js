import React from 'react';
import { Title, Text, Stack, Group, Image, Flex } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

const Optimization = () => {
  return (
    <Stack spacing="lg" w="100%">
<div data-slide>
      <Title order={3} mt="md">Optimizers</Title>
      {/* SGD */}
                    <CodeBlock
        language="python"
        code={`import torch.optim as optim`}/>
                  <Flex direction="column" align="center" mt="md" mb="md">
                    <Image
                      src="/assets/python-deep-learning/module3/optimize_with_momentum.gif"
                      alt="Gradient Flow Visualization"
                      style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
                      fluid
                    />
                              </Flex>
        </div>
        <div data-slide>
      <div>
        <Title order={4}>Stochastic Gradient Descent (SGD)</Title>
        <BlockMath>{`w_{t+1} = w_t - \\eta \\nabla L(w_t)`}</BlockMath>
        <Text size="sm">Basic gradient descent with fixed learning rate η</Text>
              <CodeBlock
        language="python"
        code={`sgd = optim.SGD(model.parameters(), lr=0.01)`}
      />
      <Text>Stored values SGD: No additional state</Text>
      </div></div>
<div data-slide>
      {/* Momentum */}
      <div>
        <Title order={4}>SGD with Momentum</Title>
        <BlockMath>{`m_{t} = \\beta m_{t-1} + (1-\\beta)\\nabla L(w_t)`}</BlockMath>
        <BlockMath>{`w_{t+1} = w_t - \\eta m_{t}`}</BlockMath>
        <Text size="sm">Adds velocity term to dampen oscillations and accelerate convergence</Text>
      </div>
              <CodeBlock
        language="python"
        code={`momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`}
      />
            <Text>Stored values SGD: Velocity buffer for each parameter</Text>
      </div>
      <div data-slide>
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
                    <CodeBlock
        language="python"
        code={`adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))`}
      />
      <Text>First moment (m) and second moment (v) estimates for each parameter</Text>
      </div>

      


    </Stack>
  );
};

export default Optimization;

// TODO ReduceLROnPlateau