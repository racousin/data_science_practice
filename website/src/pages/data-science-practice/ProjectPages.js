import React from 'react';
import { Container, Title, SimpleGrid, Card, Text, Badge, Group, Button, Stack } from '@mantine/core';
import { IconCalendar } from '@tabler/icons-react';
import { useNavigate } from 'react-router-dom';

const ProjectPages = () => {
  const navigate = useNavigate();

  const projects = [
        {
      year: '2025',
      title: 'Data Science Practice Project 2025',
      status: 'on-going',
      path: '/courses/data-science-practice/project/2025'
    },
    {
      year: '2024',
      title: 'Data Science Practice Project 2024',
      status: 'completed',
      path: '/courses/data-science-practice/project/2024'
    }

  ];

  return (
    <Container size="lg" py="xl">
      <Stack spacing="xl">
        <div>
          <Title order={1} mb="md">Final Projects</Title>
        </div>

        <SimpleGrid cols={2} spacing="lg" breakpoints={[{ maxWidth: 'sm', cols: 1 }]}>
          {projects.map((project) => (
            <Card key={project.year} shadow="sm" p="lg" radius="md" withBorder>
              <Card.Section withBorder inheritPadding py="xs">
                <Group position="apart">
                  <Group>
                    <IconCalendar size={20} />
                    <Text weight={500}>{project.year}</Text>
                  </Group>
                  <Badge 
                    color={project.status === 'on-going' ? 'green' : project.status === 'active' ? 'blue' : 'gray'}
                    variant="light"
                  >
                    {project.status === 'on-going' ? 'On-going' : project.status === 'active' ? 'Active' : 'Completed'}
                  </Badge>
                </Group>
              </Card.Section>

              <Stack spacing="sm" mt="md">
                <Text size="lg" weight={500}>{project.title}</Text>
                <Text size="sm" c="dimmed"> 
                  {project.description}
                </Text>
                

                <Button 
                  fullWidth 
                  mt="md"
                  variant={project.status === 'on-going' || project.status === 'active' ? 'filled' : 'light'}
                  onClick={() => navigate(project.path)}
                >
                  {project.status === 'on-going' ? 'View Project' : 'View Project'}
                </Button>
              </Stack>
            </Card>
          ))}
        </SimpleGrid>

      </Stack>
    </Container>
  );
};

export default ProjectPages;