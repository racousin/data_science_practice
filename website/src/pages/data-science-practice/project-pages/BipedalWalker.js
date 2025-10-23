import React from 'react';
import { Container, Title, Text, List, Alert, Anchor, Paper, Group, Stack, Button } from '@mantine/core';
import { IconRobot, IconInfoCircle, IconTrophy, IconClock, IconCpu, IconDatabase, IconChartBar, IconArrowLeft, IconBrain } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import CodeBlock from 'components/CodeBlock';

const BipedalWalker = () => {
  return (
    <Container size="lg" py="xl">
      <Group spacing="md" mb="xl">
        <Button
          component={Link}
          to="/courses/data-science-practice/project/2025"
          leftIcon={<IconArrowLeft size={16} />}
          variant="subtle"
        >
          Back to Project 2025
        </Button>
        <Button
          component={Link}
          to="/courses/data-science-practice/project/permuted-mnist"
          leftIcon={<IconBrain size={16} />}
          variant="light"
        >
          Option A: Permuted MNIST
        </Button>
        <Button
          component="a"
          href="/courses/data-science-practice/students/data_science_practice_2025"
          leftIcon={<IconChartBar size={16} />}
          variant="light"
          color="blue"
          ml="auto"
        >
          View Your Project Evaluation
        </Button>
      </Group>

      <Title order={1} mb="lg">
        <IconRobot size={32} style={{ marginRight: '10px', verticalAlign: 'middle' }} />
        Bipedal Walker Challenge
      </Title>

      <Title order={2} mb="md">Challenge Overview</Title>
      <Text mb="md">
        The Bipedal Walker challenge focuses on training a reinforcement learning agent to successfully
        walk in a simulated 2D physics environment. Your task is to develop an RL agent that demonstrates
        stable walking behavior and efficient learning progression.
      </Text>

      <Alert icon={<IconTrophy />} color="blue" mb="md">
        <Text weight={600} mb="sm">Competition Details</Text>
        <Text size="sm">
          Test your agent against others on the ML Arena platform:
        </Text>
        <Anchor href="https://ml-arena.com/viewcompetition/10" target="_blank" size="sm">
          https://ml-arena.com/viewcompetition/10
        </Anchor>
      </Alert>

      <Paper p="md" withBorder mb="xl">
        <Group spacing="xl">
          <Stack spacing="xs">
            <Group spacing="xs">
              <IconClock size={20} />
              <Text weight={500}>Training Time</Text>
            </Group>
            <Text size="sm">Flexible (days to weeks)</Text>
          </Stack>
          <Stack spacing="xs">
            <Group spacing="xs">
              <IconDatabase size={20} />
              <Text weight={500}>Environment</Text>
            </Group>
            <Text size="sm">OpenAI Gym / Gymnasium</Text>
          </Stack>
          <Stack spacing="xs">
            <Group spacing="xs">
              <IconCpu size={20} />
              <Text weight={500}>Compute</Text>
            </Group>
            <Text size="sm">CPU or GPU supported</Text>
          </Stack>
        </Group>
      </Paper>

      <Title order={2} mb="md">How the Challenge Works</Title>
      <Text mb="md">
        The Bipedal Walker environment simulates a 2D robot that must learn to walk:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>2D Physics Simulation:</strong> A bipedal robot with 4 joints (hip and knee for each leg)
          must learn to coordinate movement in a Box2D physics environment.
        </List.Item>
        <List.Item>
          <strong>Continuous Control:</strong> The agent outputs torque values for each joint,
          requiring smooth, coordinated control policies.
        </List.Item>
        <List.Item>
          <strong>Sparse Rewards:</strong> The agent receives rewards for forward progress but
          is heavily penalized for falling, making exploration challenging.
        </List.Item>
        <List.Item>
          <strong>Generalization Required:</strong> The terrain is randomly generated each episode,
          requiring robust locomotion strategies.
        </List.Item>
      </List>

      <Alert icon={<IconInfoCircle />} color="yellow" mb="xl">
        <Text weight={500}>Key Challenge</Text>
        <Text size="sm">
          Unlike supervised learning, the agent must learn through trial and error. Early training
          will involve many falls and failures before discovering stable walking gaits.
        </Text>
      </Alert>

      <Title order={2} mb="md">Getting Started</Title>

      <Title order={3} mb="sm">1. Set Up the Environment</Title>
      <Text mb="md">
        Install Gymnasium and create the BipedalWalker environment:
      </Text>
      <CodeBlock
        code={`# Install dependencies
pip install gymnasium[box2d]
pip install stable-baselines3  # Optional: for RL algorithms

# Test the environment
import gymnasium as gym

env = gym.make('BipedalWalker-v3', render_mode='human')
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">2. Understand the Observation and Action Spaces</Title>
      <Text mb="md">
        The environment provides rich sensory information:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Observation Space:</strong> 24-dimensional vector including hull angle, velocity, joint positions, and LIDAR readings</List.Item>
        <List.Item><strong>Action Space:</strong> 4 continuous values [-1, 1] for hip and knee torques</List.Item>
        <List.Item><strong>Reward Signal:</strong> Forward progress minus energy costs and fall penalties</List.Item>
      </List>

      <Title order={3} mb="sm">3. Implement Your Agent</Title>
      <Text mb="md">
        Your RL agent should follow a standard interface:
      </Text>
      <CodeBlock
        code={`class RLAgent:
    def __init__(self, observation_space, action_space):
        """Initialize your agent with networks and hyperparameters"""
        self.observation_space = observation_space
        self.action_space = action_space
        # Initialize policy network, value network, etc.

    def get_action(self, observation, training=True):
        """Select action given observation"""
        # Return action in [-1, 1]^4
        pass

    def train(self, buffer):
        """Update policy using collected experience"""
        pass

    def save(self, path):
        """Save model weights"""
        pass

    def load(self, path):
        """Load model weights"""
        pass`}
        language="python"
      />

      <Title order={2} mb="md">Strategic Advice</Title>

      <Title order={3} mb="sm">Start Simple</Title>
      <Text mb="md">
        Begin with proven baseline algorithms:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>PPO (Proximal Policy Optimization):</strong> Stable and sample-efficient,
          great starting point. Use Stable-Baselines3 implementation.
        </List.Item>
        <List.Item>
          <strong>SAC (Soft Actor-Critic):</strong> Good for continuous control,
          handles exploration well.
        </List.Item>
        <List.Item>
          <strong>Monitor Everything:</strong> Track episode rewards, episode lengths,
          and learning curves.
        </List.Item>
      </List>

      <Title order={3} mb="sm" mt="lg">Optimize Systematically</Title>
      <Text mb="md">
        Once you have a baseline, improve methodically:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Hyperparameter Tuning:</strong> Learning rate, batch size, and network architecture
          significantly impact performance.
        </List.Item>
        <List.Item>
          <strong>Reward Shaping:</strong> Consider adding intermediate rewards for
          maintaining balance or lifting legs.
        </List.Item>
        <List.Item>
          <strong>Curriculum Learning:</strong> Start with easier terrain or shorter episodes,
          gradually increase difficulty.
        </List.Item>
        <List.Item>
          <strong>Observation Normalization:</strong> Normalize inputs for stable training.
        </List.Item>
      </List>

      <Title order={3} mb="sm" mt="lg">Advanced Approaches</Title>
      <Text mb="md">
        For cutting-edge performance:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Evolutionary Strategies:</strong> Population-based training can find
          robust solutions.
        </List.Item>
        <List.Item>
          <strong>Imitation Learning:</strong> Bootstrap from expert demonstrations
          or successful trajectories.
        </List.Item>
        <List.Item>
          <strong>Domain Randomization:</strong> Train on varied physics parameters
          for robustness.
        </List.Item>
        <List.Item>
          <strong>Hierarchical RL:</strong> Separate high-level gait planning from
          low-level control.
        </List.Item>
      </List>

      <Alert icon={<IconInfoCircle />} color="green" mb="xl">
        <Text weight={500}>Pro Tips</Text>
        <List spacing="xs" size="sm" mt="xs">
          <List.Item>Use vectorized environments for faster training</List.Item>
          <List.Item>Implement early stopping to prevent overfitting</List.Item>
          <List.Item>Save checkpoints regularly during training</List.Item>
          <List.Item>Visualize learned behavior to debug issues</List.Item>
        </List>
      </Alert>

      <Title order={2} mb="md">Testing Your Agent</Title>

      <Title order={3} mb="sm">Local Testing</Title>
      <Text mb="md">
        Evaluate your trained agent:
      </Text>
      <CodeBlock
        code={`import gymnasium as gym
import numpy as np
from your_package.agent import RLAgent

# Load environment and agent
env = gym.make('BipedalWalker-v3')
agent = RLAgent.load('path/to/model')

# Evaluate over multiple episodes
rewards = []
for episode in range(100):
    observation, info = env.reset()
    episode_reward = 0

    while True:
        action = agent.get_action(observation, training=False)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    rewards.append(episode_reward)

print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"Success Rate: {np.mean(np.array(rewards) > 250):.1%}")`}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Competition Submission</Title>
      <Text mb="md">
        Submit your agent to ML-Arena for evaluation:
      </Text>
      <List spacing="sm" mb="md" type="ordered">
        <List.Item>
          <strong>Competition Page:</strong> Visit{' '}
          <Anchor href="https://ml-arena.com/viewcompetition/10" target="_blank">
            ML-Arena Bipedal Walker Competition
          </Anchor>
        </List.Item>
        <List.Item>
          <strong>Create Account:</strong> Sign up or connect with your GitHub account
        </List.Item>
        <List.Item>
          <strong>Submit Agent:</strong> Click "Submit Agent" at the top:
          <List withPadding mt="xs">
            <List.Item>Upload your <code>agent.py</code> file with the required class interface</List.Item>
            <List.Item>Add trained model weights and any additional Python files</List.Item>
            <List.Item>Include a <code>requirements.txt</code> with all dependencies</List.Item>
            <List.Item>Select the appropriate kernel (sklearn, pytorch, tensorflow, etc.)</List.Item>
          </List>
        </List.Item>
        <List.Item>
          <strong>Evaluation Process:</strong>
          <List withPadding mt="xs">
            <List.Item>Your agent will be evaluated on multiple episodes with different random seeds</List.Item>
            <List.Item>Performance is measured by average reward across episodes</List.Item>
            <List.Item>View your score and rank on the leaderboard</List.Item>
          </List>
        </List.Item>
      </List>

      <Title order={2} mb="md">Expected Performance</Title>
      <Text mb="md">
        Typical performance ranges after training:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Random Policy:</strong> -100 to -50 (falls immediately)</List.Item>
        <List.Item><strong>Basic PPO (1M steps):</strong> 200-250 average reward</List.Item>
        <List.Item><strong>Tuned PPO/SAC:</strong> 280-320 average reward</List.Item>
        <List.Item><strong>State-of-the-art:</strong> 330+ with consistent walking</List.Item>
      </List>

      <Title order={2} mb="md">Building a Reproducible Project</Title>
      <Text mb="md">
        Structure your project for reproducibility and experimentation:
      </Text>

      <Title order={3} mb="sm">Recommended Structure</Title>
      <CodeBlock
        code={`bipedal_walker_rl/
├── report.ipynb         # Results and analysis notebook
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
├── requirements.txt    # Dependencies
├── configs/
│   ├── ppo_config.yaml # PPO hyperparameters
│   └── sac_config.yaml # SAC hyperparameters
├── src/
│   ├── agents/
│   │   ├── ppo.py     # PPO implementation
│   │   ├── sac.py     # SAC implementation
│   │   └── networks.py # Neural network architectures
│   ├── utils/
│   │   ├── buffer.py   # Experience replay buffer
│   │   ├── logger.py   # Training metrics logging
│   │   └── wrappers.py # Environment wrappers
│   └── training/
│       ├── trainer.py  # Training loop
│       └── evaluator.py # Evaluation utilities
├── experiments/
│   └── logs/           # Training logs and metrics
├── models/
│   ├── checkpoints/    # Saved model weights
│   └── best_model.zip  # Best performing model
└── notebooks/
    ├── exploration.ipynb # Environment exploration
    └── analysis.ipynb    # Results visualization`}
        language="text"
      />

      <Title order={3} mb="sm" mt="lg">Key Components</Title>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Reproducible Training:</strong> Set random seeds and log all hyperparameters
        </List.Item>
        <List.Item>
          <strong>Modular Design:</strong> Separate algorithms, networks, and utilities
        </List.Item>
        <List.Item>
          <strong>Experiment Tracking:</strong> Use tensorboard or wandb for metrics
        </List.Item>
        <List.Item>
          <strong>Checkpointing:</strong> Save models regularly during training
        </List.Item>
      </List>

      <Title order={2} mb="md">Success Criteria</Title>
      <Text mb="md">A successful RL submission will demonstrate:</Text>
      <List spacing="sm" mb="xl">
        <List.Item>Consistent walking behavior across multiple test episodes</List.Item>
        <List.Item>Stable learning progression with convergence analysis</List.Item>
        <List.Item>Robust performance on different terrain configurations</List.Item>
        <List.Item>Clear documentation of hyperparameter choices and training process</List.Item>
        <List.Item>Comparative analysis with baseline algorithms and ablation studies</List.Item>
        <List.Item>Reproducible training pipeline with proper random seeding</List.Item>
      </List>

      <Alert icon={<IconRobot />} color="blue">
        <Text weight={500}>Remember</Text>
        <Text size="sm">
          Reinforcement learning requires patience and systematic experimentation. Early attempts
          will likely fail, but each failure provides valuable information about what doesn't work.
          Focus on understanding why agents fail before trying complex solutions.
        </Text>
      </Alert>

      <Title order={2} mb="md" mt="xl">Additional Resources</Title>
      <List spacing="sm">
        <List.Item>
          <strong>Competition:</strong>{' '}
          <Anchor href="https://ml-arena.com/viewcompetition/10" target="_blank">
            ML-Arena Bipedal Walker Competition
          </Anchor> - Submit and track your progress
        </List.Item>
        <List.Item>
          <strong>Environment Documentation:</strong>{' '}
          <Anchor href="https://gymnasium.farama.org/environments/box2d/bipedal_walker/" target="_blank">
            Gymnasium BipedalWalker-v3
          </Anchor> - Official environment documentation, observation/action spaces, and details
        </List.Item>
        <List.Item>
          <strong>Gymnasium API:</strong>{' '}
          <Anchor href="https://gymnasium.farama.org/" target="_blank">
            Gymnasium Library
          </Anchor> - Complete RL environment framework
        </List.Item>
        <List.Item>
          <strong>Course Materials:</strong>{' '}
          <Anchor href="https://www.raphaelcousin.com/courses/data-science-practice/module9/course/model-free-methods" target="_blank">
            Module 9: Model-Free Methods
          </Anchor> - Reinforcement learning theory and algorithms
        </List.Item>
        <List.Item>
          <strong>Stable-Baselines3:</strong>{' '}
          <Anchor href="https://stable-baselines3.readthedocs.io/" target="_blank">
            SB3 Documentation
          </Anchor> - RL algorithm implementations (PPO, SAC, TD3)
        </List.Item>
      </List>
    </Container>
  );
};

export default BipedalWalker;