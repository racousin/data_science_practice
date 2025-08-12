import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Stack, Badge, Group, Progress, Alert } from '@mantine/core';
import { Link } from 'react-router-dom';
import { 
  IconBrain,
  IconMathFunction,
  IconCpu,
  IconCode,
  IconCube,
  IconInfoCircle
} from '@tabler/icons-react';

const modules = [
  { 
    id: 1, 
    title: 'PyTorch Core Components & Tensor Mathematics', 
    description: 'Deep dive into tensor fundamentals, mathematical operations, memory management, and advanced indexing',
    icon: IconCube, 
    color: 'blue',
    topics: ['Tensor Operations', 'Linear Algebra', 'Memory Management', 'Broadcasting']
  },
  { 
    id: 2, 
    title: 'Automatic Differentiation & Gradient Mechanics', 
    description: 'Master autograd system, computational graphs, custom gradients, and advanced differentiation',
    icon: IconMathFunction, 
    color: 'green',
    topics: ['Autograd System', 'Custom Functions', 'Gradient Hooks', 'Higher-order Derivatives']
  },
  { 
    id: 3, 
    title: 'Infrastructure & Performance Optimization', 
    description: 'CPU/GPU optimization, memory profiling, parallelization, and production deployment',
    icon: IconCpu, 
    color: 'orange',
    topics: ['Performance Profiling', 'Memory Optimization', 'Parallelization', 'Compilation']
  },
  { 
    id: 4, 
    title: 'Advanced PyTorch Features & Custom Operations', 
    description: 'C++/CUDA extensions, custom operations, advanced debugging, and production systems',
    icon: IconCode, 
    color: 'red',
    topics: ['C++ Extensions', 'CUDA Programming', 'Custom Operations', 'Production Deployment']
  }
];

const PythonDeepLearningOverview = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl" mb={50}>
        <div style={{ textAlign: 'center' }}>
          <IconBrain size={48} color="var(--mantine-color-blue-6)" style={{ marginBottom: 16 }} />
          <Title order={1}>Python for Deep Learning (PyTorch)</Title>
        </div>
        <Text size="lg" color="dimmed" align="center" maw={800} mx="auto">
          Master PyTorch from the ground up with deep mathematical understanding, 
          performance optimization, and production-ready implementations
        </Text>
        
        <Group justify="center" gap="md">
          <Badge size="lg" variant="light" color="blue">
            4 Modules × 3 Hours
          </Badge>
          <Badge size="lg" variant="light" color="green">
            Theory + Hands-on Practice
          </Badge>
          <Badge size="lg" variant="light" color="orange">
            Advanced PyTorch
          </Badge>
        </Group>
        
        <Alert icon={<IconInfoCircle size={16} />} title="Course Focus" color="blue">
          This course emphasizes deep PyTorch understanding and mathematical foundations rather than high-level abstractions. 
          Students will implement core operations from scratch and master performance optimization.
        </Alert>
      </Stack>

      <Title order={2} mb="xl">Course Modules</Title>
      
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
        {modules.map((module) => {
          const Icon = module.icon;
          return (
            <Card 
              key={module.id} 
              shadow="md" 
              padding="lg" 
              radius="md" 
              withBorder
              style={{ 
                display: 'flex', 
                flexDirection: 'column',
                height: '100%',
                transition: 'transform 0.2s ease, box-shadow 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = 'var(--mantine-shadow-lg)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'var(--mantine-shadow-md)';
              }}
            >
              <Stack spacing="md" style={{ flex: 1 }}>
                <Group justify="space-between" align="flex-start">
                  <Icon size={24} color={`var(--mantine-color-${module.color}-6)`} />
                  <Badge color={module.color} size="sm">Module {module.id}</Badge>
                </Group>
                
                <div>
                  <Title order={4} mb="xs" style={{ lineHeight: 1.3 }}>
                    {module.title}
                  </Title>
                  <Text size="sm" color="dimmed" mb="md">
                    {module.description}
                  </Text>
                </div>

                <div>
                  <Text size="xs" fw={600} color="gray.6" mb="xs">KEY TOPICS</Text>
                  <Group gap="xs">
                    {module.topics.map((topic, index) => (
                      <Badge 
                        key={index} 
                        variant="dot" 
                        color="gray" 
                        size="xs"
                        style={{ textTransform: 'none' }}
                      >
                        {topic}
                      </Badge>
                    ))}
                  </Group>
                </div>
                
                <Group gap="xs" mt="auto" pt="md">
                  <Button 
                    size="sm"
                    variant="light"
                    color={module.color}
                    component={Link}
                    to={`/courses/python-deep-learning/module${module.id}/course`}
                    flex={1}
                  >
                    Theory
                  </Button>
                  <Button 
                    size="sm"
                    variant="filled"
                    color={module.color}
                    component={Link}
                    to={`/courses/python-deep-learning/module${module.id}/exercise`}
                    flex={1}
                  >
                    Practice
                  </Button>
                </Group>
              </Stack>
            </Card>
          );
        })}
      </SimpleGrid>

      <Stack spacing="md" mt="xl" p="md" style={{ 
        backgroundColor: 'var(--mantine-color-gray-0)', 
        borderRadius: 'var(--mantine-radius-md)',
        border: '1px solid var(--mantine-color-gray-3)'
      }}>
        <Title order={3} size="md">Course Philosophy</Title>
        <Text size="sm" color="dimmed">
          This course emphasizes <strong>mathematical understanding</strong> and <strong>implementation mastery</strong> 
          over high-level abstractions. Students will write PyTorch code from scratch, understand memory layouts, 
          optimize performance, and build production-ready systems. Each concept is tested with advanced theoretical 
          and practical challenges that require deep understanding rather than copy-paste solutions.
        </Text>
      </Stack>
      
      <Stack spacing="md" mt={50}>
        <Title order={3}>Course Progress</Title>
        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Stack spacing="md">
            <Group justify="space-between">
              <Text>Overall Progress</Text>
              <Text fw={500}>0 / 4 Modules</Text>
            </Group>
            <Progress value={0} size="lg" radius="xl" />
            <Text size="sm" color="dimmed">
              Complete all modules and hands-on exercises to master PyTorch fundamentals
            </Text>
          </Stack>
        </Card>
      </Stack>
    </Container>
  );
};

export default PythonDeepLearningOverview;