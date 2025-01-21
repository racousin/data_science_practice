import React from 'react';
import { Title, Text, Stack, Anchor, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const RLFrameworks = () => {
  return (
    <Stack spacing="md">
      <Title order={2} id="rl-frameworks" className="mb-2">
        Popular RL Frameworks
      </Title>

      <Text className="mb-4">
        Modern RL frameworks provide robust implementations of popular algorithms and utilities
        for training agents efficiently. We'll explore two major frameworks: Stable-Baselines3
        and RLlib.
      </Text>

      <Title order={3} id="stable-baselines3" className="mb-2">
        Stable-Baselines3
      </Title>

      <Text>
        <Anchor href="https://stable-baselines3.readthedocs.io/" target="_blank" rel="noopener noreferrer">
          Stable-Baselines3
        </Anchor> provides reliable implementations of RL algorithms with a simple, unified API.
      </Text>

      <Title order={4} className="mb-2">Installation</Title>
      <CodeBlock language="bash" code={`
# Base installation
pip install stable-baselines3

# Optional dependencies
pip install stable-baselines3[extra]  # Includes additional algorithms
pip install sb3-contrib              # Community contributions`} />

      <Title order={4} className="mb-2">Basic Usage</Title>
      <CodeBlock language="python" code={`
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("CartPole-v1")

# Initialize agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

# Train agent
model.learn(total_timesteps=50000)

# Evaluate agent
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save & load model
model.save("ppo_cartpole")
del model
model = PPO.load("ppo_cartpole")`} />

      <Title order={3} id="rllib" className="mb-2">
        Ray RLlib
      </Title>

      <Text>
        <Anchor href="https://docs.ray.io/en/latest/rllib/index.html" target="_blank" rel="noopener noreferrer">
          RLlib
        </Anchor> is a highly scalable library for RL that supports distributed training
        and numerous algorithms.
      </Text>

      <Title order={4} className="mb-2">Installation</Title>
      <CodeBlock language="bash" code={`
# Base installation
pip install "ray[rllib]"

# Optional dependencies
pip install "ray[rllib,tune]"  # Includes hyperparameter tuning
pip install tensorflow         # For TF-based algorithms
pip install torch             # For PyTorch-based algorithms`} />

      <Title order={4} className="mb-2">Basic Usage</Title>
      <CodeBlock language="python" code={`
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPO

# Initialize Ray (for distributed computing)
ray.init()

# Configure training
config = {
    "env": "CartPole-v1",
    "framework": "torch",
    "num_workers": 2,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 30,
    "lambda": 0.95,
    "clip_param": 0.2,
    "lr": 0.0003,
}

# Create trainer
trainer = PPO(config=config)

# Training loop
for i in range(10):
    result = trainer.train()
    print(f"Iteration {i}: mean_reward={result['episode_reward_mean']}")

# Save model
checkpoint_dir = trainer.save()
print(f"Model saved in {checkpoint_dir}")

# Clean up
trainer.stop()
ray.shutdown()`} />

      <Title order={4} className="mb-2">Distributed Training</Title>
      <CodeBlock language="python" code={`
import ray
from ray import tune

# Configure distributed training
config = {
    "env": "CartPole-v1",
    "framework": "torch",
    "num_workers": 4,
    "num_gpus": 1,
    "train_batch_size": 4000,
    
    # Hyperparameter search space
    "lambda": tune.uniform(0.9, 1.0),
    "clip_param": tune.uniform(0.1, 0.3),
    "lr": tune.loguniform(1e-4, 1e-3),
}

# Run distributed training with hyperparameter tuning
results = tune.run(
    "PPO",
    config=config,
    num_samples=10,  # Number of trials
    stop={"training_iteration": 100},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max",
)`} />

      <Title order={3} className="mb-2">Framework Comparison</Title>
      <Stack spacing="xs">
        <Text weight={500}>Stable-Baselines3:</Text>
        <List>
          <List.Item>Easier to use and get started with</List.Item>
          <List.Item>Well-documented implementations of popular algorithms</List.Item>
          <List.Item>Good for single-machine training</List.Item>
          <List.Item>Excellent for prototyping and research</List.Item>
        </List>

        <Text weight={500} mt="sm">RLlib:</Text>
        <List>
          <List.Item>Highly scalable with built-in distributed training</List.Item>
          <List.Item>Supports more algorithms and frameworks</List.Item>
          <List.Item>Integrated with Ray ecosystem for tuning and serving</List.Item>
          <List.Item>Better for production deployments</List.Item>
        </List>
      </Stack>
    </Stack>
  );
};

export default RLFrameworks;