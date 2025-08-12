import React from 'react';
import { Title, Text, Stack, Container, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

const WeightInitialization = () => {
  return (
    <Container fluid>
      <Stack gap="xl">

        {/* Methods Section */}
        <div>
          
        <Title order={3} mb="sm">Zero/Constant Initialization</Title>
        <Text mb="md">
          Initializing all weights to zero (or the same constant) creates symmetry among neurons, preventing them from learning distinct features and hindering effective training:
        </Text>
        <BlockMath>{String.raw`W = 0`}</BlockMath>


          <Title order={3} mt="lg" mb="sm">Random Normal Initialization</Title>
          <Text mb="md">
            Basic approach using fixed standard deviation (problematic for deep networks):
          </Text>
          <BlockMath>{String.raw`W \sim \mathcal{N}(0, \sigma^2)`}</BlockMath>

          <Title order={3} mt="lg" mb="sm">Xavier/Glorot Initialization</Title>
          <Text mb="md">
            Scales based on layer sizes, ideal for linear/tanh/sigmoid activations:
          </Text>
          <BlockMath>
            {String.raw`W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})`}
          </BlockMath>

          <Title order={3} mt="lg" mb="sm">He Initialization</Title>
          <Text mb="md">
            Modified for ReLU activations, accounts for rectification:
          </Text>
          <BlockMath>
            {String.raw`W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})`}
          </BlockMath>

          <Title order={3} mt="lg" mb="sm">PyTorch Default Initializations</Title>
          <Text mb="md">
          Linear Layer (nn.Linear) weights are initialized using a uniform distribution bounded by
          </Text>
          <Text mb="md">
    where <InlineMath>{String.raw`\text{fan\_in}`}</InlineMath> is the number of input features to the layer. Biases are initialized to zero.
  </Text>
          <BlockMath>
    {String.raw`W \sim \mathcal{U} \left(-\frac{1}{\sqrt{\text{fan\_in}}}, \frac{1}{\sqrt{\text{fan\_in}}}\right)`}
  </BlockMath>
        </div>

        {/* PyTorch Defaults */}
        <div>
          <CodeBlock
            language="python"
            code={`
# Linear Layer (nn.Linear) default initialization:
# weights: uniform distribution bounded by +/- 1/sqrt(fan_in)
layer = nn.Linear(input_size, output_size)  # Automatically initialized

# Default initialization can be overridden:
nn.init.xavier_normal_(layer.weight)  # For tanh/sigmoid
nn.init.kaiming_normal_(layer.weight)  # For ReLU
nn.init.constant_(layer.bias, 0)  # Typically zero for biases



import torch
import torch.nn as nn
import torch.nn.init as init

# Define a simple Sequential model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

# Custom initialization function
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)  # Applying Xavier initialization
        if m.bias is not None:
            init.constant_(m.bias, 0)  # Initialize biases to zero

# Apply the custom initialization function to the model
model.apply(initialize_weights)`}
          />
        </div>
        <div>
          <Image
            src="/assets/module7/weight_distributions.png"
            alt="Weight initialization distributions comparison"
            radius="md"
          />
        </div>
      </Stack>
    </Container>
  );
};

export default WeightInitialization;