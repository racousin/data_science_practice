import React from 'react';
import { Container, Title, Text, Stack, Anchor } from '@mantine/core';

const projects = [
  {
    title: 'ML Arena',
    description: 'A platform for machine learning competition and algorithm evaluation, providing comprehensive benchmarking tools for ML practitioners.',
    url: 'https://ml-arena.com/',
    type: 'Platform'
  },
  {
    title: 'BNF Chat',
    description: 'A chatbot RAG experiment designed to challenge the Gallica search engine, exploring advanced information retrieval techniques.',
    url: 'https://bnfchat.isir.upmc.fr/login',
    type: 'Research',
    collaborators: [
      { name: 'Anfu Tang', url: 'https://www.linkedin.com/in/anfu-t-9a64b8171/' }
    ]
  },
  {
    title: 'Realistic IDF Trajectories',
    description: 'Generating realistic Île-de-France trajectories for mobility research and urban planning applications.',
    url: 'https://gitlab.lip6.fr/trnguyen/netmob25',
    type: 'Research'
  },
  {
    title: 'BachGen',
    description: 'An XML music generator that creates Bach-style compositions using machine learning techniques.',
    url: 'https://github.com/gomar0801/BachGen',
    type: 'Research',
    collaborators: [
      { name: 'Margot Utrera', url: 'https://www.linkedin.com/in/margot-utrera-1901a5230' }
    ]
  },
  {
    title: 'GeoArches',
    description: 'A collaborative research project with INRIA focusing on geospatial data processing and archaeological site analysis.',
    url: 'https://github.com/INRIA/geoarches',
    type: 'Research',
    collaborators: [
      { name: 'INRIA Team', url: 'https://www.inria.fr/' }
    ]
  }
];

const Projects = () => {
  return (
    <Container size="lg" py="xl">
      <Stack spacing="xl" mb={50}>
        <Title order={1}>Projects & Research</Title>
        <Text c="dimmed" maw={800}>
          A collection of research projects and practical applications bridging academic research with real-world implementations
        </Text>
      </Stack>

      <Stack spacing="xl">
        {projects.map((project, index) => (
          <Stack key={index} spacing="xs">
            <Title order={3}>
              <Anchor href={project.url} target="_blank" style={{ textDecoration: 'none', color: 'inherit' }}>
                {project.title}
              </Anchor>
            </Title>
            <Text c="dimmed" size="sm">
              {project.description}
            </Text>
            {project.collaborators && (
              <Text size="xs" c="dimmed">
                Collaborators: {project.collaborators.map((collaborator, i) => (
                  <span key={i}>
                    <Anchor href={collaborator.url} target="_blank" size="xs">
                      {collaborator.name}
                    </Anchor>
                    {i < project.collaborators.length - 1 && ', '}
                  </span>
                ))}
              </Text>
            )}
          </Stack>
        ))}
      </Stack>

      <Stack mt={60} spacing="md">
        <Title order={2}>Research Focus</Title>
        <Text c="dimmed">
          My research interests span across machine learning applications, natural language processing, 
          mobility modeling, and the intersection of AI with creative domains like music generation. 
          I'm particularly interested in building practical tools that bridge academic research with 
          real-world applications.
        </Text>
      </Stack>
    </Container>
  );
};

export default Projects;