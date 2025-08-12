import React from 'react';
import { Title, Text, Code, Stack, Anchor } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import VectorizedEnvironmentsPage from './VectorizedEnvironmentsPage'

const GymnasiumGuide = () => {
  return (
    <Stack spacing="md">
      <Title order={2} id="gymnasium-environments" className="mb-2">
        Training with Gymnasium Environments
      </Title>

      <Text>
        <Anchor href="https://gymnasium.farama.org/" target="_blank" rel="noopener noreferrer">
          Gymnasium
        </Anchor> is the standard API for single-agent reinforcement learning environments,
        maintained by the Farama Foundation. It's a fork and successor of OpenAI Gym.
      </Text>

      <Title order={3} className="mb-2">Installation</Title>
      <CodeBlock language="bash" code={`
# Base installation
pip install gymnasium

# Additional environment dependencies
pip install gymnasium[box2d]    # For environments like LunarLander
pip install gymnasium[atari]    # For Atari environments
pip install gymnasium[accept-rom-license] # Accept Atari ROM license`} />

      <Title order={3} className="mb-2">Basic Usage</Title>
      <CodeBlock language="python" code={`
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Environment information
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)

# Basic interaction loop
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()`} />

      <Title order={3} className="mb-2">Vectorized Environments</Title>
      <Text className="mb-4">
        Vectorized environments allow parallel simulation for faster training and more
        stable gradient updates:
      </Text>

      <VectorizedEnvironmentsPage/>
      <Title order={3} className="mb-2">Custom Environment Creation</Title>
      <Text className="mb-4">
        You can create custom environments by subclassing gymnasium.Env:
      </Text>

      <CodeBlock language="python" code={`
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize state
        self.state = self.observation_space.sample()
        return self.state, {}
    
    def step(self, action):
        # Implement environment dynamics
        next_state = self.state + np.random.randn(4) * 0.1
        reward = 1.0 if np.all(np.abs(next_state) < 1.0) else 0.0
        terminated = False
        truncated = False
        self.state = next_state
        return self.state, reward, terminated, truncated, {}
        
    def render(self):
        pass  # Implement visualization if needed`} />
    </Stack>
  );
};

export default GymnasiumGuide;