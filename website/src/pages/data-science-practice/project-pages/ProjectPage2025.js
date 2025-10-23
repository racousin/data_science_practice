import React from 'react';
import { Container, Title, Text, List, Button, Group, Badge, Anchor } from '@mantine/core';
import { Link } from 'react-router-dom';
import { IconBrain, IconRobot, IconChartBar } from '@tabler/icons-react';

const ProjectPage2025 = () => {
  return (
    <Container size="lg" py="xl">
      <Group spacing="md" mb="xl" position="right">
        <Button
          component="a"
          href="/courses/data-science-practice/students/data_science_practice_2025"
          leftIcon={<IconChartBar size={16} />}
          variant="light"
          color="blue"
        >
          View Your Project Evaluation
        </Button>
      </Group>

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
          to="/courses/data-science-practice/project/permuted-mnist"
          leftIcon={<IconBrain size={16} />}
          variant="light"
          size="lg"
        >
          Option A: Permuted MNIST
        </Button>
        <Button
          component={Link}
          to="/courses/data-science-practice/project/bipedal-walker"
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

      <Title order={2} mb="md">4. Project Requirements</Title>
      <Text mb="md">
        Your project consists of three main components:
      </Text>

      <Title order={3} mb="sm">4.1 ML-Arena Competition Performance (50% of project grade)</Title>
      <Text mb="md">
        Submit your agent to ML-Arena and achieve the best possible performance:
      </Text>
      <List spacing="sm" mb="md" type="ordered">
        <List.Item>
          <strong>Create Account:</strong> Go to <Anchor href="https://ml-arena.com" target="_blank">ml-arena.com</Anchor> and
          create an account or connect with your GitHub account
        </List.Item>
        <List.Item>
          <strong>Select Competition:</strong> Navigate to your chosen competition (Permuted MNIST or Bipedal Walker)
        </List.Item>
        <List.Item>
          <strong>Submit Agent:</strong> Click "Submit Agent" at the top of the competition page:
          <List withPadding mt="xs">
            <List.Item>Upload your <code>agent.py</code> file following the required class format</List.Item>
            <List.Item>Add any additional Python files your agent needs</List.Item>
            <List.Item>Select the appropriate kernel (sklearn, pytorch, tensorflow, etc.)</List.Item>
            <List.Item>Your agent will be evaluated and ranked on the leaderboard</List.Item>
          </List>
        </List.Item>
        <List.Item>
          <strong>Performance:</strong> Your grade will be based on your agent's ranking and score
        </List.Item>
      </List>

      <Title order={3} mb="sm">4.2 GitHub Package (50% of project grade)</Title>
      <Text mb="md">
        Create a clean, professional GitHub repository containing:
      </Text>

      <Title order={4} mb="sm">A. Benchmark Algorithms Package</Title>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Modular Code Structure:</strong> Organized Python package with clear separation of concerns
        </List.Item>
        <List.Item>
          <strong>Multiple Algorithms:</strong> Implement and benchmark different approaches (baseline, optimized, advanced)
        </List.Item>
        <List.Item>
          <strong>Evaluation Package:</strong> Clear evaluation module to measure and compare algorithm performance:
          <List withPadding mt="xs">
            <List.Item>Automated evaluation scripts</List.Item>
            <List.Item>Performance metrics (accuracy, speed, resource usage)</List.Item>
            <List.Item>Comparison tools between different algorithms</List.Item>
            <List.Item>Results logging and visualization</List.Item>
          </List>
        </List.Item>
        <List.Item>
          <strong>README:</strong> Installation instructions, usage examples, repository structure
        </List.Item>
        <List.Item>
          <strong>Requirements:</strong> Complete <code>requirements.txt</code> or <code>pyproject.toml</code>
        </List.Item>
      </List>

      <Title order={4} mb="sm">B. Resume Notebook (resume.ipynb)</Title>
      <Text mb="md">
        A concise Jupyter notebook (maximum 10 pages) containing:
      </Text>
      <List spacing="sm" mb="md" type="ordered">
        <List.Item>
          <strong>Methodology:</strong> Explain your approach, algorithms used, and design decisions
        </List.Item>
        <List.Item>
          <strong>Results:</strong> Present performance metrics, comparisons, and visualizations
        </List.Item>
        <List.Item>
          <strong>Package Usage:</strong> Demonstrate how to use your package to reproduce the results:
          <List withPadding mt="xs">
            <List.Item>Installation steps</List.Item>
            <List.Item>Running experiments</List.Item>
            <List.Item>Reproducing benchmark results</List.Item>
          </List>
        </List.Item>
        <List.Item>
          <strong>Conclusion:</strong> Summarize findings and state the name of your best agent submitted to ML-Arena
        </List.Item>
        <List.Item>
          <strong>Next Steps:</strong> Discuss potential improvements and future work
        </List.Item>
      </List>

      <Title order={3} mb="sm">4.3 Bonus Features (Extra Credit)</Title>
      <Text mb="md">
        Enhance your project with advanced features:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Research Paper Implementation:</strong> Implement algorithms from recent research papers
        </List.Item>
        <List.Item>
          <strong>GitHub Actions CI/CD:</strong> Automated testing and validation pipeline
        </List.Item>
        <List.Item>
          <strong>Docker Deployment:</strong> Containerized environment for reproducibility
        </List.Item>
        <List.Item>
          <strong>Advanced Visualization:</strong> Interactive dashboards for results analysis
        </List.Item>
        <List.Item>
          <strong>Documentation Website:</strong> Sphinx or similar documentation hosting
        </List.Item>
      </List>
    </Container>
  );
};

export default ProjectPage2025;