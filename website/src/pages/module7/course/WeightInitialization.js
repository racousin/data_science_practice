import React from 'react';
import { Title, Text, Stack, Container, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

const WeightInitialization = () => {
  return (
    <Container fluid>
      <Stack gap="xl">
        {/* Importance Section */}
        <div>
          <Title order={2} mb="md" id="importance">Weight Initialization</Title>
          <Text size="sm">
            Weight initialization is crucial for neural network training. Poor initialization can lead to vanishing/exploding 
            gradients and slow convergence. Let's explore different initialization methods and their impacts.
          </Text>
        </div>

        {/* Initialization Distribution Visualization */}
        <div>
          <Title order={3} mb="md">Weight Distributions Across Layers</Title>
          <Image
            src="/assets/module7/weight_distributions.png"
            alt="Weight initialization distributions comparison"
            radius="md"
          />
          <Text size="sm" mt="xs" c="dimmed">
            Comparison of weight distributions: Basic Normal (fixed σ=0.1) vs Xavier vs He initialization 
            across different layer sizes. Notice how Xavier and He adapt to layer size while Basic Normal doesn't.
          </Text>
        </div>

        {/* Methods Section */}
        <div>
          <Title order={2} mb="md" id="methods">Initialization Methods</Title>
          
          <Title order={3} mb="sm">Zero/Constant Initialization</Title>
          <Text mb="md">
            Setting all weights to zero (or any constant) breaks symmetry and prevents learning:
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
        </div>

        {/* PyTorch Defaults */}
        <div>
          <Title order={2} mb="md" id="pytorch-defaults">PyTorch Default Initializations</Title>
          <Text mb="md">
            PyTorch uses different default initializations depending on the layer type:
          </Text>
          <CodeBlock
            language="python"
            code={`
# Linear Layer (nn.Linear) default initialization:
# weights: uniform distribution bounded by +/- 1/sqrt(fan_in)
layer = nn.Linear(input_size, output_size)  # Automatically initialized

# Conv2d Layer (nn.Conv2d) default initialization:
# weights: kaiming_uniform_ with gain calculated based on negative_slope
layer = nn.Conv2d(in_channels, out_channels, kernel_size)

# Default initialization can be overridden:
nn.init.xavier_normal_(layer.weight)  # For tanh/sigmoid
nn.init.kaiming_normal_(layer.weight)  # For ReLU
nn.init.constant_(layer.bias, 0)  # Typically zero for biases`}
          />
        </div>

        {/* Implementation Examples */}
        <div>
          <Title order={2} mb="md" id="examples">Implementation Examples</Title>
          <CodeBlock
            language="python"
            code={`
# Example 1: Network with Xavier Initialization (for tanh/sigmoid)
class XavierNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(XavierNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.tanh = nn.Tanh()
        
        # Xavier initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        return self.fc2(x)

# Example 2: Network with He Initialization (for ReLU)
class HeNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(HeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
        # He initialization
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)`}
          />
        </div>
      </Stack>
    </Container>
  );
};

export default WeightInitialization;