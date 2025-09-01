import React from 'react';
import { Container, Title, Text, Stack, Paper } from '@mantine/core';

const DataPipelineTrainingLoop = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        <div>
          <Title order={1} mb="xl">
            Data Pipeline & Training Loop
          </Title>
          <Text size="xl" className="mb-6">
            DataLoader Architecture & Multiprocessing
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} mb="md">Course Content</Title>
            <Text>
              This section covers PyTorch's data loading infrastructure, training loop design,
              and batch sampling strategies for efficient neural network training.
            </Text>
          </Paper>
        </div>
      </Stack>
    </Container>
  );
};

export default DataPipelineTrainingLoop;