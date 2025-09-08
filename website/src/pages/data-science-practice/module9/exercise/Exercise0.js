import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Brain } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise0 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module9/exercise/module13_exercise0.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module9/exercise/module13_exercise0.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module9/exercise/module13_exercise0.ipynb";

  return (
    <>
      <Container fluid>
        <Stack spacing="xl" className="p-6">
          <div className="flex items-center gap-3"> 
            <Title order={1} className="text-2xl font-bold">Exercise 0: Introduction to Reinforcement Learning</Title>
          </div>

          <Stack spacing="lg">
            {/* Environment Section */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part A: Understanding RL Environments</Title>
              <Text className="text-gray-700 mb-4">
                Learn the fundamentals of reinforcement learning environments and their implementation:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Introduction to Gymnasium (formerly OpenAI Gym) framework</List.Item>
                <List.Item>Creating custom environments with state spaces and action spaces</List.Item>
                <List.Item>Implementing reward functions and transition dynamics</List.Item>
                <List.Item>Understanding environment reset and step functions</List.Item>
              </List>
            </div>

            {/* Agent Section */}
            <div >
              <Title order={2} className="text-xl font-semibold mb-4">Part B: Building RL Agents</Title>
              <Text className="text-gray-700 mb-4">
                Implement and understand the core components of reinforcement learning agents:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Agent architecture and decision-making process</List.Item>
                <List.Item>Policy implementation (random and basic deterministic)</List.Item>
                <List.Item>State representation and action selection</List.Item>
                <List.Item>Experience collection and storage</List.Item>
              </List>
            </div>

            {/* Experimentation Section */}
            <div>
              <Title order={2} className="text-xl font-semibold mb-4">Part C: Running RL Experiments</Title>
              <Text className="text-gray-700 mb-4">
                Learn to conduct and analyze reinforcement learning experiments:
              </Text>
              <List spacing="sm" className="ml-6">
                <List.Item>Setting up training loops and episode structure</List.Item>
                <List.Item>Computing and tracking cumulative rewards</List.Item>
                <List.Item>Implementing basic dynamic programming methods</List.Item>
                <List.Item>Visualizing agent performance and learning progress</List.Item>
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
    </>
  );
};

export default Exercise0;