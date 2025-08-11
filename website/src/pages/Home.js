import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Group, Stack, Box, Divider } from '@mantine/core';
import { Link } from 'react-router-dom';
import { 
  IconBrandGithub, 
  IconBrandLinkedin, 
  IconSchool,
  IconBrain,
  IconRocket
} from '@tabler/icons-react';

const Home = () => {
  return (
    <Container size="xl" py="xl">
      {/* Profile Section */}
      <Stack align="center" spacing="xl" mb={60}>
        <Title order={1} size={48}>Raphaël Cousin</Title>
        
        <Text size="xl" fw={500} c="blue">
          Research Engineer at SCAI - Sorbonne Université
        </Text>
        
        <Text size="lg" c="dimmed" maw={900} ta="center">
          Data Engineer and Scientist with a strong background in mathematics and software development. 
          Experienced in deploying and managing large-scale production projects across the entire data chain. 
          Proficient in cloud architectures, data workflows, and machine learning.
        </Text>
        
        <Group gap="md">
          <Button
            leftSection={<IconBrandLinkedin size={20} />}
            variant="default"
            component="a"
            href="https://www.linkedin.com/in/raphael-cousin/"
            target="_blank"
          >
            LinkedIn
          </Button>
          <Button
            leftSection={<IconBrandGithub size={20} />}
            variant="default"
            component="a"
            href="https://github.com/racousin"
            target="_blank"
          >
            GitHub
          </Button>
          <Button
            leftSection={<IconBrain size={20} />}
            variant="default"
            component="a"
            href="https://scai.sorbonne-universite.fr/"
            target="_blank"
          >
            SCAI Lab
          </Button>
        </Group>
      </Stack>

      <Divider my="xl" />

      {/* Main Sections */}
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="xl">
        
        {/* Teaching Section */}
        <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
          <Stack h="100%">
            <Group gap="xs">
              <IconSchool size={24} color="var(--mantine-color-blue-6)" />
              <Title order={3}>Teaching</Title>
            </Group>
            
            <Text c="dimmed" size="sm" mb="md">
              Comprehensive courses on data science and deep learning with practical exercises and real-world applications.
            </Text>
            
            <Stack gap="sm" mt="auto">
              <Button
                fullWidth
                variant="filled"
                color="blue"
                component={Link}
                to="/courses"
                leftSection={<IconSchool size={18} />}
              >
                Browse Courses
              </Button>
            </Stack>
          </Stack>
        </Card>

        {/* Projects & Research Section */}
        <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
          <Stack h="100%">
            <Group gap="xs">
              <IconRocket size={24} color="var(--mantine-color-green-6)" />
              <Title order={3}>Projects & Research</Title>
            </Group>
            
            <Text c="dimmed" size="sm" mb="md">
              Working on innovative AI solutions and research projects bridging the gap between academic research and practical implementations.
            </Text>
            
            <Stack gap="sm" mt="auto">
              <Button
                fullWidth
                variant="filled"
                color="green"
                component={Link}
                to="/projects"
                leftSection={<IconRocket size={18} />}
              >
                View Projects
              </Button>
            </Stack>
          </Stack>
        </Card>
      </SimpleGrid>
    </Container>
  );
};

export default Home;