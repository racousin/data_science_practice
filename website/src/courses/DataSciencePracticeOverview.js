import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Stack, Badge, Group, Timeline } from '@mantine/core';
import { Link } from 'react-router-dom';
import { 
  IconGitBranch, 
  IconBrandPython, 
  IconChartBar,
  IconDatabase,
  IconFilter,
  IconTable,
  IconBrain,
  IconPhoto,
  IconClock,
  IconFileText,
  IconSparkles,
  IconUsers,
  IconRobot,
  IconBrandDocker,
  IconCloud,
  IconTrophy
} from '@tabler/icons-react';

const modules = [
  { id: 0, title: 'Prerequisites & Methodology', icon: IconChartBar, color: 'gray' },
  { id: 1, title: 'Git & Version Control', icon: IconGitBranch, color: 'orange' },
  { id: 2, title: 'Python Ecosystem', icon: IconBrandPython, color: 'blue' },
  { id: 3, title: 'Data Science Landscape', icon: IconChartBar, color: 'teal' },
  { id: 4, title: 'Data Collection', icon: IconDatabase, color: 'green' },
  { id: 5, title: 'Data Preprocessing', icon: IconFilter, color: 'lime' },
  { id: 6, title: 'Tabular Models', icon: IconTable, color: 'yellow' },
  { id: 7, title: 'Deep Learning Fundamentals', icon: IconBrain, color: 'orange' },
  { id: 8, title: 'Image Processing', icon: IconPhoto, color: 'red' },
  { id: 9, title: 'Time Series Processing', icon: IconClock, color: 'pink' },
  { id: 10, title: 'Text Processing & NLP', icon: IconFileText, color: 'grape' },
  { id: 11, title: 'Generative Models', icon: IconSparkles, color: 'violet' },
  { id: 12, title: 'Recommendation Systems', icon: IconUsers, color: 'indigo' },
  { id: 13, title: 'Reinforcement Learning', icon: IconRobot, color: 'blue' },
  { id: 14, title: 'Docker & Containers', icon: IconBrandDocker, color: 'cyan' },
  { id: 15, title: 'Cloud Integration', icon: IconCloud, color: 'teal' },
];

const DataSciencePracticeOverview = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl" mb={50}>
        <Title order={1} align="center">Data Science Practice</Title>
        <Text size="lg" color="dimmed" align="center" maw={800} mx="auto">
          A comprehensive journey through the entire data science pipeline, from fundamentals to deployment
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
                  <Icon size={24} />
                  <Badge color={module.color}>Module {module.id}</Badge>
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
                        component={Link}
                        to={`/courses/data-science-practice/module${module.id}/course`}
                        fullWidth
                      >
                        Course
                      </Button>
                      <Button 
                        size="xs"
                        variant="filled"
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
              Capstone Project
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