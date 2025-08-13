import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise1 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module13/exercise/module13_exercise1.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module13/exercise/module13_exercise1.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module13/exercise/module13_exercise1.ipynb";

  return (
    <Container fluid>
      <Stack spacing="xl" className="p-6">
        <div className="flex items-center gap-3">
          <Title order={1} className="text-2xl font-bold">Exercise 1: Q-Learning with FrozenLake</Title>
        </div>

        <Stack spacing="lg">
          {/* Implementation Section */}
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Part A: Q-Learning Implementation</Title>
            <Text className="text-gray-700 mb-4">
              Implement the core components of a Q-Learning agent:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Q-table initialization and management</List.Item>
              <List.Item>Epsilon-greedy action selection strategy</List.Item>
              <List.Item>Q-value updates using the Q-Learning algorithm</List.Item>
              <List.Item>Learning rate and discount factor implementation</List.Item>
            </List>
          </div>

          {/* Environment Section */}
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Part B: FrozenLake Environment</Title>
            <Text className="text-gray-700 mb-4">
              Work with the FrozenLake-v1 environment from Gymnasium:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Understanding the state and action spaces</List.Item>
              <List.Item>Handling environment dynamics and transitions</List.Item>
              <List.Item>Managing episode termination conditions</List.Item>
              <List.Item>Implementing reward collection and processing</List.Item>
            </List>
          </div>

          {/* Training Section */}
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Part C: Training and Analysis</Title>
            <Text className="text-gray-700 mb-4">
              Train and evaluate the Q-Learning agent:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Setting up the training loop and hyperparameters</List.Item>
              <List.Item>Implementing exploration vs exploitation balance</List.Item>
              <List.Item>Tracking learning progress and performance metrics</List.Item>
              <List.Item>Visualizing Q-value convergence and policy behavior</List.Item>
            </List>
          </div>
        </Stack>


      </Stack>
      <DataInteractionPanel
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          className="mt-6"
        />
    </Container>
  );
};

export default Exercise1;