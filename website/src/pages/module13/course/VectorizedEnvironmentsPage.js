import React from 'react';
import { Title, Text, Container, Paper, Alert  } from '@mantine/core';
import { Info } from 'lucide-react';
import CodeBlock from 'components/CodeBlock';

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

      <Alert className="my-6 bg-blue-50">
        <div className="flex items-start">
          <Info className="w-5 h-5 mt-1 mr-2" />
          <div>
            <Text className="font-medium mb-2">Pro Tip</Text>
            <Text>
              When implementing vectorized environments, pay special attention to tensor shapes
              and batch processing. The key is to process all environment interactions simultaneously
              for maximum efficiency.
            </Text>
          </div>
        </div>
      </Alert>

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

      <Title order={2} id="common-challenges" className="mb-4 mt-8">Common Challenges and Solutions</Title>

      <Title order={3} className="mb-4">1. Handling Different Episode Lengths</Title>
      <CodeBlock 
        language="python" 
        code={`class EpisodeManager:
    def __init__(self, num_envs):
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs)
        
    def update(self, rewards, dones):
        self.episode_rewards += rewards
        self.episode_lengths += 1
        
        # Reset stats for completed episodes
        completed = []
        for env_idx in range(len(dones)):
            if dones[env_idx]:
                completed.append({
                    'reward': self.episode_rewards[env_idx],
                    'length': self.episode_lengths[env_idx]
                })
                self.episode_rewards[env_idx] = 0
                self.episode_lengths[env_idx] = 0
        
        return completed`}
      />

      <Title order={3} className="mb-4 mt-6">2. Resource Management</Title>
      <CodeBlock 
        language="python" 
        code={`def configure_vectorized_training(available_memory_gb=4):
    # Calculate optimal number of environments
    env_memory_estimate = 0.1  # GB per environment (example)
    optimal_num_envs = max(1, int(available_memory_gb / env_memory_estimate))
    
    # Create environments with resource constraints
    envs = make("CartPole-v1", 
                num_envs=optimal_num_envs,
                asynchronous=True,  # Better CPU utilization
                max_episode_steps=1000)  # Prevent infinite episodes
    
    return envs`}
      />

      <Title order={3} className="mb-4 mt-6">3. Synchronization and Determinism</Title>
      <CodeBlock 
        language="python" 
        code={`def create_deterministic_envs(num_envs, seed=42):
    def make_env(idx):
        def _init():
            env = gym.make("CartPole-v1")
            env.reset(seed=seed + idx)
            return env
        return _init
    
    # Create vectorized environment with different seeds
    return gym.vector.AsyncVectorEnv(
        [make_env(i) for i in range(num_envs)]
    )`}
      />

      <Alert className="my-6 bg-yellow-50">
        <div className="flex items-start">
          <Info className="w-5 h-5 mt-1 mr-2" />
          <div>
            <Text className="font-medium mb-2">Important Note</Text>
            <Text>
              When using vectorized environments in production, always monitor memory usage
              and CPU utilization. Start with a smaller number of environments and scale up
              based on your hardware capabilities.
            </Text>
          </div>
        </div>
      </Alert>
      </>
  );
};

export default VectorizedEnvironmentsPage;