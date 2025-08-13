import React from 'react';
import { Container, Title, Text, Stack, Paper } from '@mantine/core';

const AdvancedPyTorchArchitecture = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        <div>
          <Title order={1} className="mb-6">
            Advanced PyTorch & Architecture Overview
          </Title>
          <Text size="xl" className="mb-6">
            Hooks & Their Applications
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Course Content</Title>
            <Text>
              This section covers hooks and their applications, dynamic computation graphs,
              CNN convolution mathematics, attention mechanism mathematics, and custom C++ extensions.
            </Text>
          </Paper>
        </div>
      </Stack>
    </Container>
  );
};

export default AdvancedPyTorchArchitecture;