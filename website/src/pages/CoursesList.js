import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Stack, Badge, Group } from '@mantine/core';
import { Link } from 'react-router-dom';
import { IconBook, IconCode, IconChartBar, IconBrain } from '@tabler/icons-react';

const courses = [
    {
    id: 'python-deep-learning',
    title: 'Python for Deep Learning',
    description: 'Master PyTorch and deep learning fundamentals',
    modules: 4,
    icon: IconBrain,
    color: 'indigo',
    topics: ['PyTorch', 'Tensors', 'Neural Networks', 'TensorBoard'],
    path: '/courses/python-deep-learning'
  },
  {
    id: 'data-science-practice',
    title: 'Data Science Practice',
    description: 'Complete data science pipeline from data collection to deployment',
    modules: 15,
    icon: IconChartBar,
    color: 'blue',
    topics: ['Python', 'Machine Learning', 'Deep Learning', 'MLOps'],
    path: '/courses/data-science-practice'
  }
];

const CoursesList = () => {
  return (
    <Container size="lg" py="xl">
      <Stack align="center" spacing="xl" mb={50}>
        <Title order={1}>Available Courses</Title>
        <Text size="lg" color="dimmed" align="center" maw={700}>
          Choose from our comprehensive courses designed to take you from beginner to expert
        </Text>
        <Text size="sm" color="dimmed" align="center">
          Courses from the{' '}
          <Text 
            component="a" 
            href="https://ms2a.lpsm.paris/" 
            target="_blank"
            style={{ textDecoration: 'none', color: 'inherit', fontWeight: 500 }}
          >
            Master Mathématiques et Applications
          </Text>
        </Text>
      </Stack>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="xl">
        {courses.map((course) => {
          const Icon = course.icon;
          return (
            <Card key={course.id} shadow="sm" padding="lg" radius="md" withBorder>
              <Card.Section py="md" bg="gray.0">
                <Stack align="center" spacing="sm">
                  <Icon size={32} color="gray.6" />
                  <Badge size="sm" color="gray">
                    {course.modules} Modules
                  </Badge>
                </Stack>
              </Card.Section>

              <Stack spacing="md" mt="md">
                <Title order={3} align="center">
                  {course.title}
                </Title>

                <Text align="center" color="dimmed" size="sm">
                  {course.description}
                </Text>

                <Group justify="center" gap="xs">
                  {course.topics.map((topic) => (
                    <Badge key={topic} variant="light" color="gray">
                      {topic}
                    </Badge>
                  ))}
                </Group>

                <Button
                  fullWidth
                  component={Link}
                  to={course.path}
                  variant="filled"
                  color="dark"
                  leftSection={<IconBook size={18} />}
                >
                  View Course
                </Button>
              </Stack>
            </Card>
          );
        })}
      </SimpleGrid>
    </Container>
  );
};

export default CoursesList;