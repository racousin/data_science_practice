import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Container, Grid, Card, Button, Text, Title, Alert, Anchor, Badge, Group } from '@mantine/core';
import { IconCalendar, IconUsers, IconExternalLink } from '@tabler/icons-react';

const RepositoriesList = () => {
  const [repositories, setRepositories] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/repositories/repositories.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then((data) => {
        const formattedData = Object.entries(data).map(([name, details]) => ({
          name,
          ...details,
        }));
        setRepositories(formattedData);
      })
      .catch((error) => {
        console.error('Error fetching Sessions Results:', error);
        setError('Failed to fetch repository data.');
      });
  }, []);

  return (
    <Container size="xl" py="xl">
      <Title order={1} align="center" mb="xl">Sessions Results</Title>
      {error && <Alert color="red" mb="lg">{error}</Alert>}
      <Grid gutter="lg">
        {repositories.map((repo) => (
          <Grid.Col key={repo.name} span={{ base: 12, sm: 6, lg: 4 }}>
            <Card shadow="sm" padding="lg" radius="md" withBorder>
              <Group position="apart" mb="md">
                <Text fw={500} size="lg">{repo.name}</Text>
                <Badge color="blue" variant="light">
                  {repo.number_of_students} Students
                </Badge>
              </Group>
              <Group spacing="xs" mb="md">
                <IconCalendar size={16} />
                <Text size="sm">
                  {repo.start_date} - {repo.end_date}
                </Text>
              </Group>
              <Group spacing="xs" mb="md">
                <IconUsers size={16} />
                <Text size="sm">{repo.number_of_students} Students</Text>
              </Group>
              <Anchor href={repo.url} target="_blank" rel="noopener noreferrer" mb="md" display="flex" alignItems="center">
                <IconExternalLink size={16} style={{ marginRight: '0.5rem' }} />
                Repository URL
              </Anchor>
              <Button
                component={Link}
                to={`/courses/data-science-practice/students/${repo.name}`}
                variant="light"
                color="blue"
                fullWidth
                mt="md"
                radius="md"
              >
                View Students
              </Button>
            </Card>
          </Grid.Col>
        ))}
      </Grid>
    </Container>
  );
};

export default RepositoriesList;