import React from 'react';
import { Container, Title, Text, SimpleGrid, Button, Group, Stack, Divider } from '@mantine/core';
import { Link } from 'react-router-dom';
import { 
  IconBrandGithub, 
  IconBrandLinkedin, 
  IconBrain
} from '@tabler/icons-react';

const Home = () => {
  return (
    <Container size="xl" py="xl">
      {/* Profile Section */}
      <Stack align="center" spacing="xl" mb={60}>
        <Title order={1} size={48}>Raphael Cousin</Title>
        
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
            leftSection={<IconBrandGithub size={20} />}
            variant="default"
            component="a"
            href="https://github.com/racousin"
            target="_blank"
          >
            GitHub
          </Button>
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
            leftSection={<IconBrain size={20} />}
            variant="default"
            component="a"
            href="https://scai.sorbonne-universite.fr/"
            target="_blank"
          >
            SCAI
          </Button>
        </Group>
      </Stack>

      <Divider my="xl" />

      {/* Main Sections */}
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="xl">
        
        {/* Teaching Section */}
        <Stack spacing="md">
          <Title order={3}>
            <Text component={Link} to="/courses" style={{ textDecoration: 'none', color: 'inherit' }}>
              Teaching
            </Text>
          </Title>
          
          <Text c="dimmed">
            Comprehensive courses on data science and deep learning with practical exercises and real-world applications.
          </Text>
        </Stack>

        {/* Projects & Research Section */}
        <Stack spacing="md">
          <Title order={3}>
            <Text component={Link} to="/projects" style={{ textDecoration: 'none', color: 'inherit' }}>
              Projects & Research
            </Text>
          </Title>
          
          <Text c="dimmed">
            Working on innovative AI solutions and research projects bridging the gap between academic research and practical implementations.
          </Text>
        </Stack>
      </SimpleGrid>
    </Container>
  );
};

export default Home;