import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Cpu } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise3 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module3/exercises/exercise3.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module3/exercises/exercise3.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module3/exercises/exercise3.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3">
            <Cpu size={32} className="text-blue-600" />
            <Title order={1} className="text-2xl font-bold">Exercise 3.3: Data & Optimization</Title>
          </div>

          <Stack spacing="lg">
            <div className="border rounded-lg p-6 bg-gray-50">
              <Title order={2} className="text-xl font-semibold mb-4">Data Pipeline Optimization</Title>
              <Text className="text-gray-700 mb-4">
                Optimize data loading and processing for efficient training:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Profile data loading bottlenecks</List.Item>
                <List.Item>Implement custom transforms</List.Item>
                <List.Item>Create custom metrics and callbacks</List.Item>
                <List.Item>Track and compare multiple experiments</List.Item>
              </List>
            </div>
          </Stack>

          <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
            className="mt-6"
          />
        </Stack>
      </Container>
    </>
  );
};

export default Exercise3;