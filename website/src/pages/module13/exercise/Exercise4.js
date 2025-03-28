import React from 'react';
import { Container, Text, Title, Stack, List, Alert, Code } from '@mantine/core';
import { AlertTriangle } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise2 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/module13/exercise/module13_exercise4.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/module13/exercise/module13_exercise4.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/module13/exercise/module13_exercise4.ipynb";

  return (
    <Container fluid>
      <Stack spacing="xl" className="p-6">
        <div className="flex items-center gap-3">
          <Title order={1} className="text-2xl font-bold">Exercise 4: LunarLander Competition Challenge</Title>
        </div>

        <Text className="text-gray-700">
          In this exercise, you'll train an agent for the LunarLander environment and submit it to the ml-arena.com platform
          for evaluation. Your agent will need to achieve a mean reward of at least 50 over runs to be validated.
        </Text>

        <Stack spacing="lg">
          {/* Training Section */}
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Part A: Agent Development</Title>
            <Text className="text-gray-700 mb-4">
              Train your agent on the gymnasium LunarLander-v2 environment:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Implement and optimize a Deep Q-Learning (or another algo)</List.Item>
              <List.Item>Test your agent's performance locally</List.Item>
              <List.Item>Ensure consistent performance across multiple episodes</List.Item>
              <List.Item>Optimize hyperparameters for better stability</List.Item>
            </List>
          </div>

          {/* Submission Process */}
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Part B: Competition Submission</Title>
            <Text className="text-gray-700 mb-4">
              Follow these steps to submit your agent:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Create an account on ml-arena.com using your GitHub credentials</List.Item>
              <List.Item>Navigate to the LunarLander competition page: https://ml-arena.com/viewcompetition/1</List.Item>
              <List.Item>Name and upload your agent following the platform guidelines</List.Item>
              <List.Item>Deploy your submission and verify it appears on the leaderboard</List.Item>
            </List>
          </div>
          <div>
          <Title order={2} id="expected-output">Save your training notebook in github</Title>
          <Text>Create a pull request with your Jupyter Notebook (<Code>exercise4.ipynb</Code>) containing:</Text>
          <List>
            <List.Item>Implementation and training</List.Item>
          </List>
          </div>
          {/* Requirements Section */}
          <div>
            <Alert 
              icon={<AlertTriangle className="w-5 h-5" />}
              title="Validation Requirements"
              className="bg-blue-50 text-blue-900 border-blue-200"
            >
              <Text className="text-sm">
                Your agent must achieve a mean reward of at least 50 over runs to be validated. 
                The platform will automatically evaluate your agent's performance.
              </Text>
              <Text className="text-sm mt-2">
                If you're using a different username than your GitHub account, please email: 
                raphaelcousin.teaching@gmail.com
              </Text>
            </Alert>
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

export default Exercise2;