import React from 'react';
import { Title, Text, Stack, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const Dropout = () => {
  return (
    <Stack spacing="xl">
      
      <Text>
        Dropout randomly deactivates neurons during training, forcing the network to learn redundant 
        representations and preventing co-adaptation of features.
      </Text>
      
      <Alert variant="light" title="Key Properties">
        • Training: Each neuron has a probability p of being dropped (set to 0)
        • Inference: All neurons are active, but outputs are scaled by (1-p)
        • Common dropout rates: 0.2 to 0.5 (20% to 50% of neurons dropped)
      </Alert>

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

# Set model to training or evaluation mode as needed
model.train()  # Enable dropout during training
# ... training loop ...
model.eval()  # Disable dropout for evaluation
`}
      />
    </Stack>
  );
};

export default Dropout;