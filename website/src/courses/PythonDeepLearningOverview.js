import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Stack, Badge, Group } from '@mantine/core';
import { Link } from 'react-router-dom';
import { coursesData, getModuleIndex } from '../components/SideNavigation';

// Get modules from centralized course data
const modules = coursesData['python-deep-learning'].modules.map(m => ({
  id: getModuleIndex(m.id),
  title: m.name,
  icon: m.icon
}));

const PythonDeepLearningOverview = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl" mb={50}>
        <Title order={1} align="center">Python for Deep Learning (PyTorch)</Title>
        <Text size="lg" color="dimmed" align="center" maw={800} mx="auto">
          Master PyTorch from the ground up
        </Text>
      </Stack>

      <Title order={2} mb="xl">Course Modules</Title>
      
      <SimpleGrid cols={{ base: 1, sm: 2, md: 3, lg: 4 }} spacing="md">
        {modules.map((module) => {
          const Icon = module.icon;
          return (
            <Card 
              key={module.id} 
              shadow="sm" 
              padding="lg" 
              radius="md" 
              withBorder
              style={{ display: 'flex', flexDirection: 'column' }}
            >
              <Stack spacing="sm" style={{ flex: 1 }}>
                <Group justify="space-between">
                  <Icon size={20} style={{ color: 'var(--mantine-color-gray-6)' }} />
                  <Badge color="gray">Module {module.id}</Badge>
                </Group>
                
                <Text fw={500} size="sm">
                  {module.title}
                </Text>
                
                <Group gap="xs" mt="auto">
                  <Button 
                    size="xs"
                    variant="light"
                    color="gray"
                    component={Link}
                    to={`/courses/python-deep-learning/module${module.id}/course`}
                    fullWidth
                  >
                    Course
                  </Button>
                  <Button 
                    size="xs"
                    variant="filled"
                    color="dark"
                    component={Link}
                    to={`/courses/python-deep-learning/module${module.id}/exercise`}
                    fullWidth
                  >
                    Exercise
                  </Button>
                </Group>
              </Stack>
            </Card>
          );
        })}
      </SimpleGrid>
    </Container>
  );
};

export default PythonDeepLearningOverview;