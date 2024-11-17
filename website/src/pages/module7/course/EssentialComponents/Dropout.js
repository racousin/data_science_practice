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
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Add dropout after activation
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Usage
model = NetworkWithDropout()
model.train()  # Enable dropout during training
# ... training loop ...
model.eval()   # Disable dropout during evaluation`}
      />
    </Stack>
  );
};

export default Dropout;