import React from 'react';
import { Card, Text, Title, Table, Group, ThemeIcon } from '@mantine/core';
import { CheckCircle, XCircle } from 'lucide-react';

// Updated API endpoint details
const apiEndpoints = [
  {
    method: 'GET',
    description: 'Authenticate to retrieve password.',
    url: 'https://www.raphaelcousin.com/api/course/auth',
    available: true,
  },
  {
    method: 'GET',
    description: 'Retrieve volumes data based.',
    url: 'https://www.raphaelcousin.com/api/course/:password/volumes',
    available: true,
  },
  {
    method: 'POST',
    description: 'Create a new volume entry (not implemented).',
    url: 'https://www.raphaelcousin.com/api/course/:password/volume',
    available: false,
  },
  {
    method: 'GET',
    description: 'Retrieve volume data by ID (not implemented).',
    url: 'https://www.raphaelcousin.com/api/course/:password/volume/:id',
    available: false,
  },
  {
    method: 'PUT',
    description: 'Update volume data by ID (not implemented).',
    url: 'https://www.raphaelcousin.com/api/course/:password/volume/:id',
    available: false,
  },
  {
    method: 'DELETE',
    description: 'Delete volume data by ID (not implemented).',
    url: 'https://www.raphaelcousin.com/api/course/:password/volume/:id',
    available: false,
  },
];

const ApiDoc = () => {
  return (
    <div style={{ padding: '2rem' }}>
      <Title order={2} align="center" mb="lg">
        API Documentation
      </Title>

      <Card shadow="sm" p="lg" radius="md" withBorder>
        <Text size="lg" weight={500} mb="md">
          API Base URL: <Text component="span" weight={700}>https://www.raphaelcousin.com/api</Text>
        </Text>
        <Text size="sm" color="dimmed" mb="xl">
          This page provides information about the available API endpoints.
        </Text>

        <Table highlightOnHover>
          <thead>
            <tr>
              <th>Method</th>
              <th>Endpoint</th>
              <th>Description</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {apiEndpoints.map((endpoint, index) => (
              <tr key={index}>
                <td>{endpoint.method}</td>
                <td>{endpoint.url}</td>
                <td>{endpoint.description}</td>
                <td>
                  <Group position="center">
                    {endpoint.available ? (
                      <ThemeIcon color="green" variant="light">
                        <CheckCircle size={18} />
                      </ThemeIcon>
                    ) : (
                      <ThemeIcon color="red" variant="light">
                        <XCircle size={18} />
                      </ThemeIcon>
                    )}
                  </Group>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  );
};

export default ApiDoc;
