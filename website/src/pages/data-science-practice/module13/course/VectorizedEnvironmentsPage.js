import React from 'react';
import { Title, Text, Container, Paper, Alert, List, Grid } from '@mantine/core';
import { Info } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';

const ParallelTrainingSection = () => {
  return (
    <div style={{ maxWidth: '64rem', margin: '0 auto' }}>
      <Title order={1} mb="md">Parallel Environment Training</Title>
      
      <section>
        <Title order={2} mb="sm">Asynchronous Environment Execution</Title>
        <Text size="lg" mb="lg">
          In parallel RL training, environments operate independently with different termination times. 
          AsyncVectorEnv manages this through non-blocking execution, while SyncVectorEnv waits for all 
          environments to complete their steps.
        </Text>
      </section>

      <Alert icon={<Info size={16} />} mb="lg">
        Vector environments return batched observations (num_envs × obs_dim) and expect batched actions,
        enabling parallel policy evaluation and environment stepping.
      </Alert>

      <section>
        <Title order={2} mb="md">Episode Length Management</Title>
        <Paper p="md" radius="md" withBorder mb="lg">
          <Title order={3} mb="md">Handling Variable Lengths</Title>
          <List spacing="sm">
            <List.Item icon={
              <Text component="span" px="xs" bg="blue.1" style={{ fontFamily: 'monospace' }}>
                Masking
              </Text>
            }>
              Use boolean masks to track active episodes and filter terminated environments
            </List.Item>
            <List.Item icon={
              <Text component="span" px="xs" bg="blue.1" style={{ fontFamily: 'monospace' }}>
                Padding
              </Text>
            }>
              Fill terminated episodes with zero rewards and forward-pass last observation
            </List.Item>
            <List.Item icon={
              <Text component="span" px="xs" bg="blue.1" style={{ fontFamily: 'monospace' }}>
                Bootstrapping
              </Text>
            }>
              Estimate value functions for truncated episodes using current policy
            </List.Item>
          </List>
        </Paper>
      </section>

      <section>
        <Title order={2} mb="md">Memory Management</Title>
        <Grid>
          <Grid.Col span={6}>
            <Paper p="md" radius="md" withBorder>
              <Title order={3} size="h4" mb="sm">Buffer Structure</Title>
              <Text mb="xs">Pre-allocated numpy arrays for:</Text>
              <List>
                <List.Item>Observations: (num_envs × obs_dim)</List.Item>
                <List.Item>Actions: (num_envs × action_dim)</List.Item>
                <List.Item>Rewards: (num_envs,)</List.Item>
                <List.Item>Done flags: (num_envs,)</List.Item>
              </List>
            </Paper>
          </Grid.Col>
          <Grid.Col span={6}>
            <Paper p="md" radius="md" withBorder>
              <Title order={3} size="h4" mb="sm">Optimization</Title>
              <List>
                <List.Item>Circular buffers for trajectory storage</List.Item>
                <List.Item>Vectorized operations for batch processing</List.Item>
                <List.Item>Shared memory for multi-process communication</List.Item>
              </List>
            </Paper>
          </Grid.Col>
        </Grid>
      </section>
    </div>
  );
};


const VectorizedEnvironmentsPage = () => {
  return (
    <>
    
      <Title order={1} className="mb-6">Understanding Vectorized Environments</Title>
      
      <Text className="mb-6 text-lg">
        Vectorized environments are a powerful technique in reinforcement learning that allows
        parallel execution of multiple environment instances. This approach significantly
        speeds up training and improves learning stability.
      </Text>

      <Title order={2} id="basic-concept" className="mb-4 mt-8">Basic Concept</Title>
      <Text className="mb-4">
        Think of vectorized environments like running multiple simulator instances in parallel.
        Instead of having one agent interact with one environment sequentially, you have a
        single agent interacting with multiple copies of the environment simultaneously.
      </Text>

      <Title order={2} id="implementation" className="mb-4 mt-8">Implementation Deep Dive</Title>
      
      <CodeBlock 
        language="python" 
        code={`from gymnasium.vector import make
import numpy as np
import torch

# Create 4 parallel environments
envs = make("CartPole-v1", num_envs=4)

# Key shapes to understand:
# observations: (num_envs, observation_space_dim)
# actions: (num_envs, action_space_dim)
# rewards: (num_envs,)
# terminateds: (num_envs,)
# truncateds: (num_envs,)`} 
      />

      <Title order={3} className="mb-4 mt-6">Vectorized Agent Implementation</Title>
      
      <CodeBlock 
        language="python" 
        code={`class VectorizedAgent:
    def __init__(self, observation_space, action_space, num_envs):
        self.num_envs = num_envs
        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space.n)
        )
    
    def get_actions(self, observations):
        # observations shape: (num_envs, obs_dim)
        with torch.no_grad():
            logits = self.network(torch.FloatTensor(observations))
            # Return actions for all environments
            return torch.argmax(logits, dim=1).numpy()`}
      />


      <Title order={2} id="key-advantages" className="mb-4 mt-8">Key Advantages</Title>

      <Title order={3} className="mb-4">1. Faster Data Collection</Title>
      <CodeBlock 
        language="python" 
        code={`# Single environment:
for _ in range(1000):
    action = agent.get_action(obs)  # Process one observation
    next_obs, reward, done, info = env.step(action)  # One step

# Vectorized (4 environments):
for _ in range(250):  # 4x fewer iterations needed
    actions = agent.get_actions(obs)  # Process 4 observations at once
    next_obs, rewards, dones, infos = envs.step(actions)  # 4 steps at once`}
      />

      <Title order={3} className="mb-4 mt-6">2. Better Gradient Estimation</Title>
      <CodeBlock 
        language="python" 
        code={`def compute_loss_vectorized(experiences, num_envs):
    # Experiences from multiple environments provide better gradient estimates
    states = torch.FloatTensor([e['state'] for e in experiences])  # Shape: (num_envs, state_dim)
    actions = torch.LongTensor([e['action'] for e in experiences])  # Shape: (num_envs,)
    rewards = torch.FloatTensor([e['reward'] for e in experiences])  # Shape: (num_envs,)
    
    # More stable gradients due to diverse experiences
    loss = policy_gradient_loss(states, actions, rewards)
    return loss.mean()  # Average across environments`}
      />

<ParallelTrainingSection/>
      </>
  );
};

export default VectorizedEnvironmentsPage;