import React from 'react';
import { Container, Title, Text, List, Button, Group, Badge } from '@mantine/core';
import { Link } from 'react-router-dom';
import { IconBrain, IconRobot } from '@tabler/icons-react';

const ProjectPage2025 = () => {
  return (
    <Container size="lg" py="xl">
      <Title order={1} mb="lg">Project 2025</Title>

      <Title order={2} mb="md">1. Project Objectives</Title>
      <Text mb="md">
        This project constitutes <strong>50% of the total course grade</strong> (the remaining 50% comes from continuous assessment).
        The objective is to develop a machine learning package that:
      </Text>
      <List spacing="sm" mb="xl">
        <List.Item><strong>Maximizes performance</strong> on a specific ML challenge</List.Item>
        <List.Item><strong>Follows best development practices</strong> for production-ready code</List.Item>
      </List>

      <Title order={2} mb="md">2. Project Options</Title>
      <Text mb="md">Choose between two machine learning challenges:</Text>

      <Group spacing="lg" mb="xl">
        <Button
          component={Link}
          to="/data-science-practice/project-pages/permuted-mnist"
          leftIcon={<IconBrain size={16} />}
          variant="light"
          size="lg"
        >
          Option A: Permuted MNIST
        </Button>
        <Button
          component={Link}
          to="/data-science-practice/project-pages/bipedal-walker"
          leftIcon={<IconRobot size={16} />}
          variant="light"
          size="lg"
        >
          Option B: Bipedal Walker
        </Button>
      </Group>

      <Title order={2} mb="md">3. Project Organization & Timeline</Title>

      <Title order={3} mb="sm">Team Structure</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>Team Size:</strong> 1-2 students per team</List.Item>
        <List.Item><strong>Repository:</strong> Create private GitHub repository, add <code>racousin</code> as collaborator</List.Item>
      </List>

      <Title order={3} mb="sm">Key Deadlines</Title>
      <List spacing="sm" mb="xl">
        <List.Item>
          <Badge color="blue" variant="light" mr="sm">October 14th</Badge>
        Send email to raphaelcousin.education@gmail.com with:
          <List withPadding mt="xs">
            <List.Item>Team members</List.Item>
            <List.Item>Chosen project</List.Item>
            <List.Item>GitHub repository link</List.Item>
          </List>
        </List.Item>
        <List.Item>
          <Badge color="red" variant="light" mr="sm">November 11th, midnight (French time)</Badge>
          Project freeze date (send your agent name in ml-arena leaderboard)
        </List.Item>
      </List>

      <Title order={2} mb="md">4. Evaluation Criteria</Title>

      <Title order={3} mb="sm">4.1 Performance (50% of project grade)</Title>
      <List spacing="sm" mb="md">
        <List.Item>Model/algorithm performance on ml-arena.com</List.Item>

      </List>

      <Title order={3} mb="sm">4.2 Deliverables (50% of project grade)</Title>

      <Title order={4} mb="sm">Mandatory Requirements</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>README:</strong> Project overview, installation instructions, usage examples, repository organization</List.Item>
        <List.Item><strong>Clean Python Package:</strong> Well-structured, modular code with proper documentation</List.Item>
        <List.Item><strong>Jupyter Notebook</strong> (1-10 pages): Package usage, methodology, benchmarks, theoretical aspects</List.Item>
        <List.Item><strong>Reproducibility:</strong> Clear installation and reproducible experiments</List.Item>
      </List>

      <Title order={4} mb="sm">Bonus Features</Title>
      <List spacing="sm">
        <List.Item>Paper implementation</List.Item>
        <List.Item>GitHub Actions with CI tests</List.Item>
        <List.Item>Docker deployment</List.Item>
      </List>
    </Container>
  );
};

export default ProjectPage2025;