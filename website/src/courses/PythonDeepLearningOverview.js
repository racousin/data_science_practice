import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Stack, Badge, Group, Progress, Alert } from '@mantine/core';
import { Link } from 'react-router-dom';
import { 
  IconCube,
  IconBrandPython,
  IconChartLine,
  IconInfoCircle
} from '@tabler/icons-react';

const modules = [
  { 
    id: 1, 
    title: 'Introduction to Tensors', 
    description: 'Understanding multi-dimensional arrays and tensor operations',
    icon: IconCube, 
    color: 'blue',
    topics: ['Tensor basics', 'Operations', 'Broadcasting', 'GPU acceleration']
  },
  { 
    id: 2, 
    title: 'PyTorch Fundamentals', 
    description: 'Building and training neural networks with PyTorch',
    icon: IconBrandPython, 
    color: 'orange',
    topics: ['Autograd', 'Neural networks', 'Optimizers', 'Data loaders']
  },
  { 
    id: 3, 
    title: 'TensorBoard Visualization', 
    description: 'Monitoring and visualizing deep learning experiments',
    icon: IconChartLine, 
    color: 'green',
    topics: ['Metrics tracking', 'Model graphs', 'Hyperparameter tuning', 'Embeddings']
  },
];

const PythonDeepLearningOverview = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl" mb={50}>
        <Title order={1} align="center">Python for Deep Learning: PyTorch</Title>
        <Text size="lg" color="dimmed" align="center" maw={800} mx="auto">
          Master the Python ecosystem for deep learning with hands-on PyTorch tutorials
        </Text>
        
        <Alert icon={<IconInfoCircle size={16} />} title="Course Status" color="blue">
          This course is currently under development. New modules will be added progressively.
        </Alert>
      </Stack>

      <Title order={2} mb="xl">Course Modules</Title>
      
      <SimpleGrid cols={{ base: 1, md: 3 }} spacing="lg">
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
              <Card.Section py="md" bg={`${module.color}.1`}>
                <Stack align="center" spacing="xs">
                  <Icon size={40} />
                  <Badge size="lg" color={module.color}>
                    Module {module.id}
                  </Badge>
                </Stack>
              </Card.Section>
              
              <Stack spacing="md" mt="md" style={{ flex: 1 }}>
                <Title order={4}>
                  {module.title}
                </Title>
                
                <Text size="sm" color="dimmed">
                  {module.description}
                </Text>
                
                <Stack spacing="xs">
                  <Text size="xs" fw={500}>Topics covered:</Text>
                  <Group gap="xs">
                    {module.topics.map((topic) => (
                      <Badge key={topic} variant="light" color={module.color} size="sm">
                        {topic}
                      </Badge>
                    ))}
                  </Group>
                </Stack>
                
                <Button 
                  component={Link}
                  to={`/courses/python-deep-learning/module${module.id}`}
                  variant="filled"
                  color={module.color}
                  fullWidth
                  mt="auto"
                >
                  Start Module
                </Button>
              </Stack>
            </Card>
          );
        })}
      </SimpleGrid>
      
      <Stack spacing="md" mt={50}>
        <Title order={3}>Course Progress</Title>
        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Stack spacing="md">
            <Group justify="space-between">
              <Text>Overall Progress</Text>
              <Text fw={500}>0 / 3 Modules</Text>
            </Group>
            <Progress value={0} size="lg" radius="xl" />
            <Text size="sm" color="dimmed">
              Complete all modules to earn your certificate
            </Text>
          </Stack>
        </Card>
      </Stack>
    </Container>
  );
};

export default PythonDeepLearningOverview;