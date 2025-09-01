import React from 'react';
import { Title, Text, Stack, Grid } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const TrainingBasics = () => {
  return (
    <Stack spacing="xl">
      <Grid grow>
        <Grid.Col span={6}>
          <Stack>
            <Title order={4}>Epochs</Title>
            <Text>
              One complete pass through the entire training dataset. Multiple epochs allow the model 
              to iteratively improve its predictions.
            </Text>
          </Stack>
        </Grid.Col>

        <Grid.Col span={6}>
          <Stack>
            <Title order={4}>Batch Size</Title>
            <Text>
              Number of training examples processed before updating model parameters. Common sizes: 32, 64, 128.
              Smaller batches → better generalization, larger batches → faster training.
            </Text>
          </Stack>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
# Basic training setup
batch_size = 32
num_epochs = 10

# Create data loader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()`}
      />
    </Stack>
  );
};

export default TrainingBasics;