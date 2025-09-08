import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Stack, Badge, Group } from '@mantine/core';
import { Link } from 'react-router-dom';
import { IconTrophy } from '@tabler/icons-react';
import { coursesData, getModuleIndex } from '../components/SideNavigation';

// Get modules from centralized course data
const modules = coursesData['data-science-practice'].modules
  .filter(m => m.id !== 'project')
  .map(m => ({
    id: getModuleIndex(m.id),
    title: m.name,
    icon: m.icon
  }));

const DataSciencePracticeOverview = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl" mb={50}>
        <Title order={1} align="center">Data Science Practice</Title>
        <Text size="lg" color="dimmed" align="center" maw={800} mx="auto">
        </Text>
        
        <Group justify="center" gap="md">
          <Button 
            component={Link} 
            to="/courses/data-science-practice/results"
            variant="light"
            color="green"
          >
            View Results
          </Button>
          <Button 
            component={Link} 
            to="/courses/data-science-practice/project"
            variant="light"
            color="blue"
          >
            Final Project
          </Button>
        </Group>
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
                  {module.id > 0 && (
                    <>
                      <Button 
                        size="xs"
                        variant="light"
                        color="gray"
                        component={Link}
                        to={`/courses/data-science-practice/module${module.id}/course`}
                        fullWidth
                      >
                        Course
                      </Button>
                      <Button 
                        size="xs"
                        variant="filled"
                        color="dark"
                        component={Link}
                        to={`/courses/data-science-practice/module${module.id}/exercise`}
                        fullWidth
                      >
                        Exercise
                      </Button>
                    </>
                  )}
                  {module.id === 0 && (
                    <Button 
                      size="xs"
                      variant="light"
                      color="gray"
                      component={Link}
                      to={`/courses/data-science-practice/module${module.id}`}
                      fullWidth
                    >
                      View
                    </Button>
                  )}
                </Group>
              </Stack>
            </Card>
          );
        })}
        
        <Card 
          shadow="sm" 
          padding="lg" 
          radius="md" 
          withBorder
          style={{ display: 'flex', flexDirection: 'column', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
        >
          <Stack spacing="sm" style={{ flex: 1 }}>
            <Group justify="space-between">
              <IconTrophy size={24} color="white" />
              <Badge color="yellow">Final</Badge>
            </Group>
            
            <Text fw={500} size="sm" c="white">
              Final Project
            </Text>
            
            <Button 
              size="xs"
              variant="white"
              component={Link}
              to="/courses/data-science-practice/project"
              fullWidth
              mt="auto"
            >
              Start Project
            </Button>
          </Stack>
        </Card>
      </SimpleGrid>
    </Container>
  );
};

export default DataSciencePracticeOverview;