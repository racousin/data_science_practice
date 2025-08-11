import React from 'react';
import { Container, Title, Text, SimpleGrid, Card, Button, Stack, Badge, Group, Box, Anchor } from '@mantine/core';
import { 
  IconExternalLink, 
  IconBrandGithub,
  IconUsers,
  IconRocket,
  IconMusic,
  IconMessage,
  IconMap,
  IconChartBar,
  IconBrain
} from '@tabler/icons-react';

const projects = [
  {
    id: 'ml-arena',
    title: 'ML Arena',
    description: 'A platform for machine learning competition and algorithm evaluation, providing comprehensive benchmarking tools for ML practitioners.',
    url: 'https://ml-arena.com/',
    icon: IconChartBar,
    color: 'blue',
    type: 'Platform',
    status: 'Active',
    technologies: ['Machine Learning', 'Web Platform', 'Algorithm Evaluation']
  },
  {
    id: 'bnf-chat',
    title: 'BNF Chat',
    description: 'A chatbot RAG experiment designed to challenge the Gallica search engine, exploring advanced information retrieval techniques.',
    url: 'https://bnfchat.isir.upmc.fr/login',
    icon: IconMessage,
    color: 'green',
    type: 'Research',
    status: 'Active',
    collaborators: [
      { name: 'Anfu Tang', linkedin: 'https://www.linkedin.com/in/anfu-t-9a64b8171/' }
    ],
    technologies: ['RAG', 'NLP', 'Information Retrieval', 'Chatbot']
  },
  {
    id: 'netmob25',
    title: 'Realistic IDF Trajectories',
    description: 'Generating realistic Île-de-France trajectories for mobility research and urban planning applications.',
    url: 'https://gitlab.lip6.fr/trnguyen/netmob25',
    icon: IconMap,
    color: 'orange',
    type: 'Research',
    status: 'Active',
    technologies: ['Mobility Modeling', 'Data Generation', 'Urban Analytics']
  },
  {
    id: 'bachgen',
    title: 'BachGen',
    description: 'An XML music generator that creates Bach-style compositions using machine learning techniques.',
    url: 'https://github.com/gomar0801/BachGen',
    githubUrl: 'https://github.com/gomar0801/BachGen',
    icon: IconMusic,
    color: 'purple',
    type: 'Research',
    status: 'Completed',
    collaborators: [
      { name: 'Margot Utrera', linkedin: 'https://www.linkedin.com/in/margot-utrera-1901a5230' }
    ],
    technologies: ['Music Generation', 'XML', 'Machine Learning', 'Bach Style']
  },
  {
    id: 'geoarches',
    title: 'GeoArches',
    description: 'A collaborative research project with INRIA focusing on geospatial data processing and archaeological site analysis.',
    url: 'https://github.com/INRIA/geoarches',
    githubUrl: 'https://github.com/INRIA/geoarches',
    icon: IconBrain,
    color: 'teal',
    type: 'Research',
    status: 'Active',
    collaborators: [
      { name: 'INRIA Team', url: 'https://www.inria.fr/' }
    ],
    technologies: ['Geospatial Analysis', 'Archaeology', 'Data Processing']
  }
];

const Projects = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl" mb={50}>
        <Title order={1} align="center">Projects & Research</Title>
        <Text size="lg" c="dimmed" align="center" maw={800} mx="auto">
          A collection of research projects and practical applications bridging academic research with real-world implementations
        </Text>
      </Stack>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="xl">
        {projects.map((project) => {
          const Icon = project.icon;
          return (
            <Card key={project.id} shadow="sm" padding="lg" radius="md" withBorder>
              <Stack spacing="md">
                <Group justify="space-between" align="flex-start">
                  <Group gap="sm">
                    <Icon size={32} color={`var(--mantine-color-${project.color}-6)`} />
                    <Box>
                      <Title order={3}>{project.title}</Title>
                      <Group gap="xs" mt={4}>
                        <Badge variant="light" color={project.color}>{project.type}</Badge>
                        <Badge 
                          variant="outline" 
                          color={project.status === 'Active' ? 'green' : 'gray'}
                        >
                          {project.status}
                        </Badge>
                      </Group>
                    </Box>
                  </Group>
                </Group>

                <Text size="sm" c="dimmed">
                  {project.description}
                </Text>

                <Group gap="xs" wrap="wrap">
                  {project.technologies.map((tech, index) => (
                    <Badge key={index} variant="dot" color="gray" size="sm">
                      {tech}
                    </Badge>
                  ))}
                </Group>

                {project.collaborators && project.collaborators.length > 0 && (
                  <Box>
                    <Text size="xs" fw={500} mb="xs" c="dimmed">
                      <IconUsers size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                      Collaborators:
                    </Text>
                    <Group gap="xs">
                      {project.collaborators.map((collaborator, index) => (
                        <Anchor
                          key={index}
                          href={collaborator.linkedin || collaborator.url}
                          target="_blank"
                          size="xs"
                          c={project.color}
                        >
                          {collaborator.name}
                        </Anchor>
                      ))}
                    </Group>
                  </Box>
                )}

                <Group gap="sm" mt="auto">
                  <Button
                    variant="filled"
                    color={project.color}
                    component="a"
                    href={project.url}
                    target="_blank"
                    leftSection={<IconExternalLink size={16} />}
                    flex={1}
                  >
                    Visit Project
                  </Button>
                  {project.githubUrl && (
                    <Button
                      variant="outline"
                      color="dark"
                      component="a"
                      href={project.githubUrl}
                      target="_blank"
                      leftSection={<IconBrandGithub size={16} />}
                    >
                      GitHub
                    </Button>
                  )}
                </Group>
              </Stack>
            </Card>
          );
        })}
      </SimpleGrid>

      <Box mt={60}>
        <Title order={2} mb="md" align="center">Research Focus</Title>
        <Text align="center" c="dimmed" maw={800} mx="auto">
          My research interests span across machine learning applications, natural language processing, 
          mobility modeling, and the intersection of AI with creative domains like music generation. 
          I'm particularly interested in building practical tools that bridge academic research with 
          real-world applications.
        </Text>
      </Box>
    </Container>
  );
};

export default Projects;