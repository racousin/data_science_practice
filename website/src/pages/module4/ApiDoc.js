import React from 'react';
import { Card, Text, Title, Table, Group, ThemeIcon, Accordion, Badge } from '@mantine/core';
import { CheckCircle, XCircle } from 'lucide-react';

// API endpoint details for course
const courseApiEndpoints = [
  {
    method: 'GET',
    description: 'Authenticate to retrieve password.',
    url: 'https://www.raphaelcousin.com/api/course/auth',
    available: true,
  },
  {
    method: 'GET',
    description: 'Retrieve volumes data.',
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

// API endpoint details for exercise
const exerciseApiEndpoints = [
    {
      method: 'GET',
      description: 'Authenticate to retrieve password.',
      url: 'https://www.raphaelcousin.com/api/exercise/auth',
      available: true,
    },
    {
      method: 'GET',
      description: 'Retrieve volumes data.',
      url: 'https://www.raphaelcousin.com/api/exercise/:password/prices',
      available: true,
    },
    {
      method: 'POST',
      description: 'Create a new volume entry (not implemented).',
      url: 'https://www.raphaelcousin.com/api/exercise/:password/prices',
      available: false,
    },
    {
      method: 'GET',
      description: 'Retrieve volume data by ID (not implemented).',
      url: 'https://www.raphaelcousin.com/api/exercise/:password/prices/:id',
      available: false,
    },
    {
      method: 'PUT',
      description: 'Update volume data by ID (not implemented).',
      url: 'https://www.raphaelcousin.com/api/exercise/:password/prices/:id',
      available: false,
    },
    {
      method: 'DELETE',
      description: 'Delete volume data by ID (not implemented).',
      url: 'https://www.raphaelcousin.com/api/exercise/:password/prices/:id',
      available: false,
    },
  ];

// Render API rows for table
const renderApiRows = (endpoints) => {
  return endpoints.map((endpoint, index) => (
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
  ));
};

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
          This page provides detailed information about the available API endpoints.
        </Text>

        <Accordion>
          <Accordion.Item value="course">
            <Accordion.Control>
              <Text size="lg" weight={500}>Course API</Text>
            </Accordion.Control>
            <Accordion.Panel>
              <Table highlightOnHover>
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Endpoint</th>
                    <th>Description</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>{renderApiRows(courseApiEndpoints)}</tbody>
              </Table>
            </Accordion.Panel>
          </Accordion.Item>

          <Accordion.Item value="exercise">
            <Accordion.Control>
              <Text size="lg" weight={500}>Exercise API</Text>
            </Accordion.Control>
            <Accordion.Panel>
              <Table highlightOnHover>
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Endpoint</th>
                    <th>Description</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>{renderApiRows(exerciseApiEndpoints)}</tbody>
              </Table>
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>

      </Card>
    </div>
  );
};

export default ApiDoc;
