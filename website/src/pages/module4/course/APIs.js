import React from 'react';
import { Container, Title, Text, Accordion, Stack, List, Paper, Table } from '@mantine/core';
import { Globe, Lock, ArrowLeftRight, Code } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const SectionIcon = ({ type }) => {
  const iconProps = { size: 24, strokeWidth: 1.5 };
  switch (type) {
    case 'overview': return <Globe {...iconProps} />;
    case 'auth': return <Lock {...iconProps} />;
    case 'requests': return <ArrowLeftRight {...iconProps} />;
    case 'implementation': return <Code {...iconProps} />;
    default: return null;
  }
};

const APISection = ({ type, title, content }) => (
  <Accordion.Item value={type}>
    <Accordion.Control icon={<SectionIcon type={type} />}>
      {title}
    </Accordion.Control>
    <Accordion.Panel>
      <Stack gap="md">
        {content}
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const APIs = () => {
  const sections = [
    {
      type: 'overview',
      title: 'What are APIs?',
      content: (
        <>
          <Text>
            APIs (Application Programming Interfaces) provide standardized ways for different software systems to communicate. 
            In data science, they serve as crucial tools for accessing and collecting data from various sources.
          </Text>
          <Paper p="md" className="bg-slate-50">
            <Stack gap="sm">
              <Title order={4}>Core API Concepts</Title>
              <List>
                <List.Item><strong>Endpoints:</strong> URLs that represent specific resources (e.g., /users, /data)</List.Item>
                <List.Item><strong>HTTP Methods:</strong> Actions to perform (GET, POST, PUT, DELETE)</List.Item>
                <List.Item><strong>Data Format:</strong> Usually JSON for data exchange</List.Item>
                <List.Item><strong>Status Codes:</strong> Response indicators (200 Success, 404 Not Found)</List.Item>
              </List>
            </Stack>
          </Paper>
        </>
      )
    },
    {
      type: 'auth',
      title: 'Authentication Methods',
      content: (
        <>
          <Text>
            API authentication verifies client identity and controls access to resources. Different methods offer varying levels of security and complexity.
          </Text>
          <Table striped highlightOnHover>
            <thead>
              <tr>
                <th>Method</th>
                <th>Description</th>
                <th>Best For</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>API Keys</td>
                <td>Simple key included in request header</td>
                <td>Basic applications, public APIs</td>
              </tr>
              <tr>
                <td>OAuth 2.0</td>
                <td>Token-based authorization protocol</td>
                <td>Complex permissions, third-party access</td>
              </tr>
              <tr>
                <td>JWT</td>
                <td>Encoded tokens with user information</td>
                <td>Stateless authentication, modern web APIs</td>
              </tr>
            </tbody>
          </Table>
        </>
      )
    },
    {
      type: 'requests',
      title: 'API Communication',
      content: (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Paper p="md" className="bg-slate-50">
              <Stack gap="sm">
                <Title order={4}>Request Components</Title>
                <List>
                  <List.Item><strong>Method:</strong> GET, POST, PUT, DELETE</List.Item>
                  <List.Item><strong>URL:</strong> API endpoint address</List.Item>
                  <List.Item><strong>Headers:</strong> Metadata, authentication</List.Item>
                  <List.Item><strong>Body:</strong> Data for POST/PUT requests</List.Item>
                </List>
              </Stack>
            </Paper>
            <Paper p="md" className="bg-slate-50">
              <Stack gap="sm">
                <Title order={4}>Response Elements</Title>
                <List>
                  <List.Item><strong>Status:</strong> Success/failure code</List.Item>
                  <List.Item><strong>Headers:</strong> Response metadata</List.Item>
                  <List.Item><strong>Body:</strong> Requested data (JSON)</List.Item>
                  <List.Item><strong>Errors:</strong> Error messages if failed</List.Item>
                </List>
              </Stack>
            </Paper>
          </div>
        </>
      )
    },
    {
      type: 'implementation',
      title: 'Python Implementation',
      content: (
        <>
          <Text>
            The requests library provides a straightforward way to interact with APIs in Python. Here's a comprehensive example:
          </Text>
          <CodeBlock
            language="python"
            code={`import requests

# Basic GET request
def get_data(url, api_key):
    """Fetch data from an API endpoint."""
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        # Make request with authentication
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Basic POST request
def create_resource(url, api_key, data):
    """Create a new resource via API."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error creating resource: {e}")
        return None

# Example usage
API_KEY = 'your_api_key'
BASE_URL = 'https://api.example.com'

# Get data
users = get_data(f'{BASE_URL}/users', API_KEY)
if users:
    print(f"Retrieved {len(users)} users")

# Create new resource
new_user = {
    'name': 'John Doe',
    'email': 'john@example.com'
}
result = create_resource(f'{BASE_URL}/users', API_KEY, new_user)
if result:
    print(f"Created user with ID: {result['id']}")`}
          />
          <Text mt="md" size="sm" c="dimmed">
            Note: Replace 'your_api_key' and 'https://api.example.com' with actual API credentials and endpoints.
          </Text>
        </>
      )
    }
  ];

  return (
    <Container fluid>
      <Stack gap="xl">
        <div>
          <Title order={1}>APIs (Application Programming Interfaces)</Title>
          <Text mt="md">
            APIs are essential tools in data science for accessing and collecting data from various sources. They provide structured methods for data retrieval and manipulation, enabling efficient data collection pipelines.
          </Text>
        </div>

        <Accordion>
          {sections.map(section => (
            <APISection key={section.type} {...section} />
          ))}
        </Accordion>
      </Stack>
    </Container>
  );
};

export default APIs;