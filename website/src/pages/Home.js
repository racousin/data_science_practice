import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Group, Stack, Box, Divider } from '@mantine/core';
import { Link } from 'react-router-dom';
import { 
  IconBrandGithub, 
  IconBrandLinkedin, 
  IconSchool,
  IconBrain,
  IconRocket,
  IconUsers,
  IconCode,
  IconDatabase,
  IconCloud,
  IconChartBar,
  IconMessage
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
      <SimpleGrid cols={{ base: 1, md: 3 }} spacing="xl">
        
        {/* Research Section */}
        <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
          <Stack h="100%">
            <Group gap="xs">
              <IconBrain size={24} color="var(--mantine-color-blue-6)" />
              <Title order={3}>Research</Title>
            </Group>
            
            <Text c="dimmed" size="sm" mb="md">
              Working on innovative AI solutions bridging the gap between business needs and practical implementations.
            </Text>
            
            <Stack gap="sm" mt="auto">
              <Button
                fullWidth
                variant="light"
                color="blue"
                component="a"
                href="https://scai.sorbonne-universite.fr/"
                target="_blank"
                leftSection={<IconBrain size={18} />}
              >
                SCAI Research Lab
              </Button>
            </Stack>
          </Stack>
        </Card>

        {/* Projects Section */}
        <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
          <Stack h="100%">
            <Group gap="xs">
              <IconRocket size={24} color="var(--mantine-color-green-6)" />
              <Title order={3}>Projects</Title>
            </Group>
            
            <Text c="dimmed" size="sm" mb="md">
              Building robust, scalable, and efficient data systems with focus on practical AI applications.
            </Text>
            
            <Stack gap="sm" mt="auto">
              <Button
                fullWidth
                variant="light"
                color="green"
                component="a"
                href="https://bnfchat.isir.upmc.fr/login"
                target="_blank"
                leftSection={<IconMessage size={18} />}
              >
                BNF Chat
              </Button>
              <Button
                fullWidth
                variant="light"
                color="green"
                component="a"
                href="https://ml-arena.com"
                target="_blank"
                leftSection={<IconChartBar size={18} />}
              >
                ML Arena
              </Button>
            </Stack>
          </Stack>
        </Card>

        {/* Teaching Section */}
        <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
          <Stack h="100%">
            <Group gap="xs">
              <IconSchool size={24} color="var(--mantine-color-orange-6)" />
              <Title order={3}>Teaching</Title>
            </Group>
            
            <Text c="dimmed" size="sm" mb="md">
              Comprehensive courses on data science and deep learning with practical exercises and real-world applications.
            </Text>
            
            <Stack gap="sm" mt="auto">
              <Button
                fullWidth
                variant="gradient"
                gradient={{ from: 'orange', to: 'red' }}
                component={Link}
                to="/courses"
                leftSection={<IconSchool size={18} />}
              >
                Browse Courses
              </Button>
            </Stack>
          </Stack>
        </Card>
      </SimpleGrid>

      {/* Expertise Section */}
      <Box mt={60}>
        <Title order={2} mb="xl" ta="center">Areas of Expertise</Title>
        
        <SimpleGrid cols={{ base: 2, sm: 3, md: 6 }} spacing="md">
          <Stack align="center" gap="xs">
            <IconCode size={32} color="var(--mantine-color-blue-6)" />
            <Text size="sm" fw={500}>Software Development</Text>
          </Stack>
          <Stack align="center" gap="xs">
            <IconDatabase size={32} color="var(--mantine-color-green-6)" />
            <Text size="sm" fw={500}>Data Engineering</Text>
          </Stack>
          <Stack align="center" gap="xs">
            <IconBrain size={32} color="var(--mantine-color-purple-6)" />
            <Text size="sm" fw={500}>Machine Learning</Text>
          </Stack>
          <Stack align="center" gap="xs">
            <IconCloud size={32} color="var(--mantine-color-cyan-6)" />
            <Text size="sm" fw={500}>Cloud Architecture</Text>
          </Stack>
          <Stack align="center" gap="xs">
            <IconChartBar size={32} color="var(--mantine-color-orange-6)" />
            <Text size="sm" fw={500}>Data Analysis</Text>
          </Stack>
          <Stack align="center" gap="xs">
            <IconUsers size={32} color="var(--mantine-color-red-6)" />
            <Text size="sm" fw={500}>Team Leadership</Text>
          </Stack>
        </SimpleGrid>
      </Box>
    </Container>
  );
};

export default Home;