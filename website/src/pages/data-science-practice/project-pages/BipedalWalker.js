import React from 'react';
import { Container, Title, Text, List, Code, Alert } from '@mantine/core';
import { IconRobot, IconInfoCircle, IconCalendarEvent } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';

const BipedalWalker = () => {
  return (
    <Container size="lg" py="xl">
      <Title order={1} mb="lg">
        <IconRobot size={32} style={{ marginRight: '10px', verticalAlign: 'middle' }} />
        Option B: Bipedal Walker - Reinforcement Learning
      </Title>

      <Alert icon={<IconCalendarEvent />} color="orange" mb="xl">
        <strong>Important:</strong> RL session scheduled for October 6th. Attendance recommended for this project option.
      </Alert>

      <Title order={2} mb="md">Challenge Overview</Title>
      <Text mb="md">
        The Bipedal Walker challenge focuses on training a reinforcement learning agent to successfully
        walk in a simulated 2D physics environment. Your task is to develop an RL agent that demonstrates
        stable walking behavior and efficient learning progression.
      </Text>

      <Alert icon={<IconInfoCircle />} color="blue" mb="xl">
        <strong>Goal:</strong> Train an RL agent to achieve consistent walking performance with stable
        learning convergence and robust locomotion strategies.
      </Alert>

      <Title order={2} mb="md">Environment Description</Title>
      <Text mb="md">
        The Bipedal Walker environment (OpenAI Gym) presents the following characteristics:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Agent:</strong> 2D bipedal robot with 4 joints (hip and knee for each leg)</List.Item>
        <List.Item><strong>Observation Space:</strong> 24-dimensional continuous state vector</List.Item>
        <List.Item><strong>Action Space:</strong> 4-dimensional continuous control (torque for each joint)</List.Item>
        <List.Item><strong>Physics:</strong> Box2D physics simulation with realistic dynamics</List.Item>
        <List.Item><strong>Terrain:</strong> Randomly generated ground with varying difficulty</List.Item>
      </List>

      <Title order={2} mb="md">State and Action Spaces</Title>

      <Title order={3} mb="sm">Observation Space (24 dimensions)</Title>
      <List spacing="sm" mb="md">
        <List.Item>Hull angle and angular velocity</List.Item>
        <List.Item>Hull position and velocity (x, y)</List.Item>
        <List.Item>Joint positions and velocities for both legs</List.Item>
        <List.Item>Ground contact sensors for both legs</List.Item>
        <List.Item>10 LIDAR sensors for terrain perception</List.Item>
      </List>

      <Title order={3} mb="sm">Action Space (4 dimensions)</Title>
      <List spacing="sm" mb="xl">
        <List.Item>Hip joint torque for left and right legs</List.Item>
        <List.Item>Knee joint torque for left and right legs</List.Item>
        <List.Item>Actions are continuous values in range [-1, 1]</List.Item>
      </List>

      <Title order={2} mb="md">Reward Structure</Title>
      <Text mb="md">
        Understanding the reward function is crucial for successful training:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Forward Progress:</strong> +1.3 * forward_velocity for moving right</List.Item>
        <List.Item><strong>Energy Penalty:</strong> -0.00035 * torque^2 for each joint</List.Item>
        <List.Item><strong>Termination Penalty:</strong> -100 if the agent falls (hull touches ground)</List.Item>
        <List.Item><strong>Success Bonus:</strong> Additional reward for completing the course</List.Item>
      </List>

      <CodeBlock
        code={`import gym

# Initialize environment
env = gym.make('BipedalWalker-v3')
observation = env.reset()

# Episode loop
for step in range(1000):
    action = agent.get_action(observation)  # Your RL agent
    observation, reward, done, info = env.step(action)

    if done:
        break

env.close()`}
        language="python"
      />

      <Title order={2} mb="md">Evaluation Metrics</Title>
      <Text mb="md">
        Your RL agent will be evaluated on multiple performance criteria:
      </Text>

      <Title order={3} mb="sm">Performance Metrics</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>Average Episode Reward:</strong> Mean reward over 100 test episodes</List.Item>
        <List.Item><strong>Success Rate:</strong> Percentage of episodes where agent completes the course</List.Item>
        <List.Item><strong>Walking Stability:</strong> Consistency in forward progress without falling</List.Item>
        <List.Item><strong>Sample Efficiency:</strong> Episodes required to achieve stable performance</List.Item>
      </List>

      <Title order={3} mb="sm">Learning Metrics</Title>
      <List spacing="sm" mb="xl">
        <List.Item><strong>Convergence Speed:</strong> Time to reach stable performance</List.Item>
        <List.Item><strong>Learning Stability:</strong> Variance in performance during training</List.Item>
        <List.Item><strong>Final Performance:</strong> Best sustained performance level achieved</List.Item>
      </List>

      <Title order={2} mb="md">Recommended Algorithms</Title>
      <Text mb="md">
        Consider these RL algorithms suitable for continuous control tasks:
      </Text>

      <Title order={3} mb="sm">Policy Gradient Methods</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>PPO (Proximal Policy Optimization):</strong> Stable, sample-efficient, widely used</List.Item>
        <List.Item><strong>SAC (Soft Actor-Critic):</strong> Off-policy, good for exploration</List.Item>
        <List.Item><strong>TD3 (Twin Delayed DDPG):</strong> Improved DDPG with better stability</List.Item>
      </List>

      <Title order={3} mb="sm">Implementation Frameworks</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>Stable-Baselines3:</strong> High-quality implementations of RL algorithms</List.Item>
        <List.Item><strong>Ray RLlib:</strong> Scalable RL library with distributed training</List.Item>
        <List.Item><strong>Custom Implementation:</strong> PyTorch/TensorFlow from scratch</List.Item>
      </List>

      <Title order={2} mb="md">Training Considerations</Title>

      <Title order={3} mb="sm">Hyperparameter Tuning</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>Learning Rate:</strong> Critical for stable policy updates</List.Item>
        <List.Item><strong>Network Architecture:</strong> Hidden layer sizes and activation functions</List.Item>
        <List.Item><strong>Batch Size:</strong> Balance between stability and sample efficiency</List.Item>
        <List.Item><strong>Exploration Strategy:</strong> Noise parameters for action exploration</List.Item>
      </List>

      <Title order={3} mb="sm">Training Strategies</Title>
      <List spacing="sm" mb="xl">
        <List.Item><strong>Curriculum Learning:</strong> Start with easier terrain, gradually increase difficulty</List.Item>
        <List.Item><strong>Reward Shaping:</strong> Additional intermediate rewards for learning guidance</List.Item>
        <List.Item><strong>Experience Replay:</strong> Efficient use of collected experience data</List.Item>
        <List.Item><strong>Regularization:</strong> Prevent overfitting to specific terrain patterns</List.Item>
      </List>

      <Title order={2} mb="md">Baseline Performance</Title>
      <Text mb="md">
        Your solution should exceed these baseline benchmarks:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Random Policy:</strong> ~-100 average reward (falls immediately)</List.Item>
        <List.Item><strong>Basic PPO:</strong> ~200-250 average reward after training</List.Item>
        <List.Item><strong>Tuned Algorithm:</strong> ~300+ average reward with consistent walking</List.Item>
      </List>

      <Title order={2} mb="md">Package Structure</Title>
      <Text mb="md">Your RL package should include:</Text>
      <List spacing="sm" mb="md">
        <List.Item><Code>src/</Code> - Core RL algorithm implementations</List.Item>
        <List.Item><Code>agents/</Code> - Agent architectures and neural network models</List.Item>
        <List.Item><Code>environments/</Code> - Environment wrappers and modifications</List.Item>
        <List.Item><Code>training/</Code> - Training loops and hyperparameter configurations</List.Item>
        <List.Item><Code>evaluation/</Code> - Testing and performance analysis tools</List.Item>
        <List.Item><Code>notebooks/</Code> - Training progress visualization and analysis</List.Item>
        <List.Item><Code>saved_models/</Code> - Trained model checkpoints</List.Item>
      </List>

      <Title order={2} mb="md">Success Criteria</Title>
      <Text mb="md">A successful RL submission will demonstrate:</Text>
      <List spacing="sm">
        <List.Item>Consistent walking behavior across multiple test episodes</List.Item>
        <List.Item>Stable learning progression with convergence analysis</List.Item>
        <List.Item>Robust performance on different terrain configurations</List.Item>
        <List.Item>Clear documentation of hyperparameter choices and training process</List.Item>
        <List.Item>Comparative analysis with baseline algorithms and ablation studies</List.Item>
        <List.Item>Reproducible training pipeline with proper random seeding</List.Item>
      </List>
    </Container>
  );
};

export default BipedalWalker;