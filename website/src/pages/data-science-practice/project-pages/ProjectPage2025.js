import React from 'react';
import { Container, Title, Text, Paper, Center, Stack } from '@mantine/core';
import { IconClock } from '@tabler/icons-react';

const ProjectPage2025 = () => {
  return (
    <Container size="md" py="xl">
      <Paper shadow="sm" p="xl" radius="md" withBorder>
        <Center>
          <Stack align="center" spacing="lg">
            <IconClock size={64} stroke={1.5} />
            <Title order={1}>2025 Project</Title>
            <Text size="xl" weight={500} c="dimmed">
              Coming Soon
            </Text>
          </Stack>
        </Center>
      </Paper>
    </Container>
  );
};

export default ProjectPage2025;