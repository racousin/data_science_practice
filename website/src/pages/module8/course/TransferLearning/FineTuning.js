import React from 'react';
import { Container, Stack, Title, Text, Paper, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain } from 'lucide-react';

const FineTuning = () => {
  return (
    <Stack spacing="lg">
      <Text size="lg">
        Fine-tuning strategies determine which layers to update during training. The choice depends on 
        dataset size, similarity to the original task, and computational resources.
      </Text>

      <Paper p="md" withBorder>
        <Title order={4} mb="sm">Common Fine-tuning Strategies:</Title>
        <Text>1. Feature Extraction: Freeze all layers except the final classifier</Text>
        <Text>2. Partial Fine-tuning: Gradually unfreeze layers from top to bottom</Text>
        <Text>3. Full Fine-tuning: Update all layers with a small learning rate</Text>
      </Paper>

      <CodeBlock
        language="python"
        code={`
# Load and modify the model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_classes = 10
model.fc = nn.Linear(2048, num_classes)

# Freeze all layers except fc
for param in model.parameters():
    param.requires_grad = False
    
for param in model.fc.parameters():
    param.requires_grad = True

# Create optimizer (only training fc parameters)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)`}
      />

    </Stack>
  );
};

export default FineTuning