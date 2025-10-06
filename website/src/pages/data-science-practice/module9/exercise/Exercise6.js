import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import DataInteractionPanel from 'components/DataInteractionPanel';
import { InlineMath } from 'react-katex';

const Exercise6 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module9/exercise/module9_exercise6.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module9/exercise/module9_exercise6.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module9/exercise/module9_exercise6.ipynb";

  return (
    <Container fluid>
      <Stack spacing="xl" className="p-6">
        <div className="flex items-center gap-3">
          <Title order={1} className="text-2xl font-bold">Exercise 6: Actor-Critic Algorithm</Title>
        </div>

        <Text className="text-gray-700 mb-4">
          Implement an Actor-Critic algorithm, combining policy gradient methods with value function estimation for efficient reinforcement learning.
        </Text>

        <Stack spacing="lg">
          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Actor Network</Title>
            <Text className="text-gray-700 mb-4">
              Implement the actor network for policy representation:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Design network architecture for discrete and continuous action spaces</List.Item>
              <List.Item>For discrete actions: output probability distribution using softmax</List.Item>
              <List.Item>For continuous actions: output mean and standard deviation for Gaussian policy</List.Item>
              <List.Item>Implement forward pass with appropriate output transformations</List.Item>
            </List>
          </div>

          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Critic Network</Title>
            <Text className="text-gray-700 mb-4">
              Implement the critic network for value estimation:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Design network to estimate state values <InlineMath>{'V(s)'}</InlineMath></List.Item>
              <List.Item>Configure hidden layers and activation functions</List.Item>
              <List.Item>Implement value function prediction</List.Item>
              <List.Item>Handle value normalization and scaling</List.Item>
            </List>
          </div>

          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Generalized Advantage Estimation</Title>
            <Text className="text-gray-700 mb-4">
              Implement GAE for advantage computation:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Compute temporal difference errors using value estimates</List.Item>
              <List.Item>Calculate advantages with GAE parameter <InlineMath>{'\u03bb'}</InlineMath></List.Item>
              <List.Item>Normalize advantages for stable training</List.Item>
              <List.Item>Compute discounted returns for value function targets</List.Item>
            </List>
          </div>

          <div>
            <Title order={2} className="text-xl font-semibold mb-4">Training Loop</Title>
            <Text className="text-gray-700 mb-4">
              Implement the actor-critic training algorithm:
            </Text>
            <List spacing="sm" className="ml-6">
              <List.Item>Collect trajectories using current policy</List.Item>
              <List.Item>Compute policy loss using log probabilities and advantages</List.Item>
              <List.Item>Compute value loss using MSE between predictions and returns</List.Item>
              <List.Item>Add entropy bonus for exploration</List.Item>
              <List.Item>Update both networks using combined loss</List.Item>
              <List.Item>Apply gradient clipping for stability</List.Item>
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

export default Exercise6;