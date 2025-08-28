import React from 'react';
import { Container, Title, Text, Stack, Paper } from '@mantine/core';

const DeviceManagementResources = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        <div>
          <Title order={1} mb="xl">
            Device Management & Resources
          </Title>
          <Text size="xl" className="mb-6">
            GPU Architecture for Deep Learning
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Course Content</Title>
            <Text>
              This section covers GPU architecture for deep learning, memory management strategies,
              calculating FLOPs and memory requirements, and mixed precision training mathematics.
            </Text>
          </Paper>
        </div>
      </Stack>
    </Container>
  );
};

export default DeviceManagementResources;