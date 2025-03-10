import React from 'react';
import { Title, Text, Stack, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const Dropout = () => {
  return (
    <Stack spacing="xl">
      
      <Text>
        Dropout randomly deactivates neurons during training, forcing the network to learn redundant 
        representations and preventing co-adaptation of features.
      </Text>
      
      <div>
        <Title order={4} mb="md">Key Properties</Title>
        <Text>
          <strong>Training:</strong> Each neuron has a probability p of being dropped (set to 0), remaining neurons are scaled by <InlineMath math="1/(1-p)" />.
        </Text>
        <BlockMath math="y_{train} = \frac{x \odot m}{1-p}" />
        <Text size="sm" mb="md" color="dimmed">where m is a binary mask with m_i ~ Bernoulli(1-p)</Text>
        
        <Text>
          <strong>Inference:</strong> All neurons are active, no scaling is applied.
        </Text>
        <BlockMath math="y_{eval} = x" />
        
        <Text>
          <strong>Common dropout rates:</strong> 0.2 to 0.5 (20% to 50% of neurons dropped)
        </Text>
      </div>

      <CodeBlock 
        language="python"
        code={`
import torch
import torch.nn as nn

# Define the sequential model with dropout
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # Dropout after ReLU
    nn.Linear(256, 10)
)

# Training mode example - dropout active with scaling by 1/(1-p)
model.train()
x = torch.ones((1, 784))
output = model(x)  # Some neurons randomly dropped, others scaled by 2.0

# Evaluation mode example - all neurons active, no scaling
model.eval()
output = model(x)  # All neurons active, no scaling applied
`}
      />
    </Stack>
  );
};

export default Dropout;