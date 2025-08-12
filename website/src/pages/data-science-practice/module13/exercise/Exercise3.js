import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise1 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module13/exercise/module13_exercise3.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module13/exercise/module13_exercise3.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module13/exercise/module13_exercise3.ipynb";

  return (
    <Container fluid>
      <Stack spacing="xl" className="p-6">
        <div className="flex items-center gap-3">
          <Title order={1} className="text-2xl font-bold">Exercise 1: Deep Q-Learning with Cart Pole</Title>
        </div>

        <Stack spacing="lg">
          {/* Implementation Section */}
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Part A: Deep Q-Learning Implementation</Title>
            <Text className="text-gray-700 mb-4">
              Implement the core components of a Q-Learning agent:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Torch Q Model initialization</List.Item>
              <List.Item>Epsilon-greedy action selection strategy</List.Item>
            </List>
          </div>


          {/* Training Section */}
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Part C: Training and improve</Title>
            <Text className="text-gray-700 mb-4">
              Train and evaluate the Deep Q-Learning agent:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Setting up the training loop and hyperparameters</List.Item>
              <List.Item>Tracking learning progress and performance metrics</List.Item>
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