import React from 'react';
import { Container, Title, Text, Accordion, Stack, List, Paper, Table } from '@mantine/core';
import { Globe, Lock, ArrowLeftRight, Code, Clock } from 'lucide-react';
import CodeBlock from "components/CodeBlock";

const SectionIcon = ({ type }) => {
  const iconProps = { size: 24, strokeWidth: 1.5 };
  switch (type) {
    case 'overview': return <Globe {...iconProps} />;
    case 'auth': return <Lock {...iconProps} />;
    case 'requests': return <ArrowLeftRight {...iconProps} />;
    case 'python': return <Code {...iconProps} />;
    case 'rate': return <Clock {...iconProps} />;
    default: return null;
  }
};

const APISection = ({ type, title, content }) => (
  <Accordion.Item value={type}>
    <Accordion.Control icon={<SectionIcon type={type} />}>
      {title}
    </Accordion.Control>
    <Accordion.Panel>
      <Stack spacing="md">
        {content}
      </Stack>
    </Accordion.Panel>
  </Accordion.Item>
);

const APIs = () => {
  const sections = [
    {
      type: 'overview',
      title: 'High-level Overview',
      content: (
        <>
          <Text>
            APIs (Application Programming Interfaces) are sets of protocols and tools for building software applications. They define how different components should interact, allowing different systems to communicate with each other.
          </Text>
          <Title order={4}>Key Concepts:</Title>
          <List>
            <List.Item>Endpoints: Specific URLs that represent objects or collections of objects</List.Item>
            <List.Item>HTTP Methods: GET, POST, PUT, DELETE, etc., defining the action to be performed</List.Item>
            <List.Item>Data Formats: Usually JSON or XML for data exchange</List.Item>
            <List.Item>Status Codes: Indicate the result of the API request (e.g., 200 OK, 404 Not Found)</List.Item>
          </List>
          <Text>
            In data science, APIs are crucial for accessing data from various sources, integrating different services, and building data pipelines.
          </Text>
        </>
      )
    },
    {
      type: 'auth',
      title: 'Authentication',
      content: (
        <>
          <Text>
            Authentication is the process of verifying the identity of a client attempting to access the API. It's crucial for securing data and controlling access to resources.
          </Text>
          <Title order={4}>Common Authentication Methods:</Title>
          <Table>
            <thead>
              <tr>
                <th>Method</th>
                <th>Description</th>
                <th>Use Case</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>API Keys</td>
                <td>A unique identifier sent with each request</td>
                <td>Simple, low-security applications</td>
              </tr>
              <tr>
                <td>OAuth</td>
                <td>Token-based protocol for delegated authorization</td>
                <td>Third-party access, complex permissions</td>
              </tr>
              <tr>
                <td>JWT (JSON Web Tokens)</td>
                <td>Encoded tokens containing claims about the user</td>
                <td>Stateless authentication, microservices</td>
              </tr>
              <tr>
                <td>Basic Auth</td>
                <td>Username and password sent in request header</td>
                <td>Simple systems, not recommended for production</td>
              </tr>
            </tbody>
          </Table>
          <Text mt="md">
            Always use HTTPS when implementing authentication to encrypt data in transit.
          </Text>
        </>
      )
    },
    {
      type: 'requests',
      title: 'Requests and Responses',
      content: (
        <>
          <Text>
            API communication involves sending requests to specific endpoints and receiving responses. Understanding this process is crucial for effective API usage.
          </Text>
          <Title order={4}>Request Components:</Title>
          <List>
            <List.Item>Method (GET, POST, PUT, DELETE, etc.)</List.Item>
            <List.Item>URL (API endpoint)</List.Item>
            <List.Item>Headers (metadata about the request, including authentication)</List.Item>
            <List.Item>Body (data sent with the request, typically for POST or PUT methods)</List.Item>
          </List>
          <Title order={4}>Response Components:</Title>
          <List>
            <List.Item>Status Code (indicates success or failure of the request)</List.Item>
            <List.Item>Headers (metadata about the response)</List.Item>
            <List.Item>Body (requested data or error information)</List.Item>
          </List>
          <Text mt="md">
            Most modern APIs use JSON (JavaScript Object Notation) for data exchange due to its simplicity and readability.
          </Text>
        </>
      )
    },
    {
      type: 'python',
      title: "Using Python's requests Library",
      content: (
        <>
          <Text>
            The requests library is a popular choice for making HTTP requests in Python. It simplifies the process of interacting with APIs.
          </Text>
          <CodeBlock
            language="python"
            code={`import requests

# Making a GET request
response = requests.get('https://api.example.com/data')
print(response.status_code)
print(response.json())

# Making a POST request
data = {'key': 'value'}
response = requests.post('https://api.example.com/create', json=data)
print(response.status_code)
print(response.json())

# Using authentication
api_key = 'your_api_key'
headers = {'Authorization': f'Bearer {api_key}'}
response = requests.get('https://api.example.com/secure', headers=headers)
print(response.status_code)

# Handling errors
response = requests.get('https://api.example.com/data')
response.raise_for_status()  # Raises an HTTPError for bad responses
data = response.json()

# Session for multiple requests
session = requests.Session()
session.headers.update({'Authorization': f'Bearer {api_key}'})
response1 = session.get('https://api.example.com/data1')
response2 = session.get('https://api.example.com/data2')
`}
          />
          <Text mt="md">
            The requests library handles many low-level details, making it easy to work with APIs in Python.
          </Text>
        </>
      )
    },
    {
      type: 'rate',
      title: 'Rate Limiting and Efficient Handling',
      content: (
        <>
          <Text>
            Rate limiting is a strategy used by APIs to control the number of requests a client can make within a specified time period. Handling rate limits efficiently is crucial for data collection at scale.
          </Text>
          <Title order={4}>Strategies for Handling Rate Limits:</Title>
          <List>
            <List.Item>Respect rate limits specified in API documentation</List.Item>
            <List.Item>Implement exponential backoff for retries</List.Item>
            <List.Item>Use asynchronous requests for better performance</List.Item>
            <List.Item>Cache responses to reduce unnecessary API calls</List.Item>
          </List>
          <CodeBlock
            language="python"
            code={`import time
import requests
from requests.exceptions import RequestException

def make_api_request(url, max_retries=3, backoff_factor=0.1):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            wait = backoff_factor * (2 ** retries)
            print(f"Request failed. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)
            retries += 1
    
    raise Exception("Max retries reached. Request failed.")

# Usage
try:
    data = make_api_request('https://api.example.com/data')
    print(data)
except Exception as e:
    print(f"Error: {e}")
`}
          />
          <Text mt="md">
            This example implements exponential backoff, which increases the wait time between retries, reducing the load on the API server and improving the chances of a successful request.
          </Text>
        </>
      )
    }
  ];

  return (
    <Container fluid>
      <Title order={1}>APIs (Application Programming Interfaces)</Title>
      <Text mt="md">
        APIs play a crucial role in data science by providing standardized ways to access and manipulate data from various sources. Understanding how to work with APIs is essential for efficient data collection and integration in data science workflows.
      </Text>
      
      <Accordion mt="xl">
        {sections.map(section => (
          <APISection key={section.type} {...section} />
        ))}
      </Accordion>

      <Text mt="xl">
        Mastering API interactions is a valuable skill in data science. It allows you to access a wide range of data sources, integrate different services, and build robust data pipelines. As you work with APIs, always consider authentication, rate limiting, and efficient data handling to ensure smooth and reliable data collection processes.
      </Text>
    </Container>
  );
};

export default APIs;