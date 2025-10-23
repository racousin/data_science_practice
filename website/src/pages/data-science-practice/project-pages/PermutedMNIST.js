import React from 'react';
import { Container, Title, Text, List, Alert, Anchor, Paper, Group, Stack, Button } from '@mantine/core';
import { IconBrain, IconInfoCircle, IconClock, IconCpu, IconDatabase, IconTrophy, IconChartBar, IconArrowLeft, IconRobot } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import CodeBlock from 'components/CodeBlock';

const PermutedMNIST = () => {
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
          to="/courses/data-science-practice/project/bipedal-walker"
          leftIcon={<IconRobot size={16} />}
          variant="light"
        >
          Option B: Bipedal Walker
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
        <IconBrain size={32} style={{ marginRight: '10px', verticalAlign: 'middle' }} />
        Permuted MNIST Challenge
      </Title>

      <Title order={2} mb="md">Challenge Overview</Title>
      <Text mb="md">
        The Permuted MNIST challenge is a fast adaptation meta-learning competition where agents must quickly
        learn to classify MNIST digits that have random permutations. The challenge tests your ability
        to create efficient learning algorithms that can adapt to new tasks within strict resource constraints.
      </Text>

      <Alert icon={<IconTrophy />} color="blue" mb="md">
        <Text weight={600} mb="sm">Competition Details</Text>
        <Text size="sm">
          Test your agent against others on the ML Arena platform:
        </Text>
        <Anchor href="https://ml-arena.com/viewcompetition/8" target="_blank" size="sm">
          https://ml-arena.com/viewcompetition/8
        </Anchor>
      </Alert>

      <Paper p="md" withBorder mb="xl">
        <Group spacing="xl">
          <Stack spacing="xs">
            <Group spacing="xs">
              <IconClock size={20} />
              <Text weight={500}>Time Limit</Text>
            </Group>
            <Text size="sm">1 minute per task</Text>
          </Stack>
          <Stack spacing="xs">
            <Group spacing="xs">
              <IconDatabase size={20} />
              <Text weight={500}>Memory Limit</Text>
            </Group>
            <Text size="sm">4 GB RAM</Text>
          </Stack>
          <Stack spacing="xs">
            <Group spacing="xs">
              <IconCpu size={20} />
              <Text weight={500}>CPU Limit</Text>
            </Group>
            <Text size="sm">2 CPU cores (no GPU)</Text>
          </Stack>
        </Group>
      </Paper>

      <Title order={2} mb="md">How the Challenge Works</Title>
      <Text mb="md">
        In each task, the environment applies two types of permutations that completely scramble the data:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Pixel Permutation:</strong> All 784 pixels (28×28) are randomly shuffled in a consistent way across all images.
          This destroys the spatial structure of the digits.
        </List.Item>
        <List.Item>
          <strong>Label Permutation:</strong> The digit labels are randomly remapped (e.g., all 3s might become 7s, all 7s become 1s).
          This means you cannot rely on prior knowledge of which digit is which.
        </List.Item>
        <List.Item>
          <strong>Fresh Start Required:</strong> Each task uses different permutations, requiring your agent to learn from scratch.
        </List.Item>
        <List.Item>
          <strong>Fast Adaptation:</strong> You have only 1 minute to both train and predict on 10,000 test samples.
        </List.Item>
      </List>

      <Alert icon={<IconInfoCircle />} color="yellow" mb="xl">
        <Text weight={500}>Key Challenge</Text>
        <Text size="sm">
          Traditional pre-trained models and convolutional approaches will struggle here. The permutations destroy
          spatial patterns, requiring algorithms that can quickly discover new structure in the data.
        </Text>
      </Alert>

      <Title order={2} mb="md">Getting Started</Title>

      <Title order={3} mb="sm">1. Set Up the Environment</Title>
      <Text mb="md">
        Clone the official repository and install the environment:
      </Text>
      <CodeBlock
        code={`# Clone the repository
git clone https://github.com/ml-arena/permuted_mnist/
cd permuted_mnist

# Install the package
pip install -e .`}
        language="bash"
      />

      <Title order={3} mb="sm" mt="lg">2. Understand the Data Flow</Title>
      <Text mb="md">
        Each task provides:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Training set:</strong> 60,000 samples with labels</List.Item>
        <List.Item><strong>Test set:</strong> 10,000 samples for prediction</List.Item>
        <List.Item><strong>Format:</strong> Images as (N, 28, 28) numpy arrays</List.Item>
        <List.Item><strong>Labels:</strong> Integers 0-9 (but permuted!)</List.Item>
      </List>

      <Title order={3} mb="sm">3. Implement Your Agent</Title>
      <Text mb="md">
        Your agent must follow this interface:
      </Text>
      <CodeBlock
        code={`class Agent:
    def __init__(self, output_dim: int = 10, seed: int = None):
        """Initialize your agent"""
        pass

    def reset(self):
        """Reset for a new task (new permutation)"""
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train on the permuted training data"""
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predictions for test data"""
        pass`}
        language="python"
      />

      <Title order={2} mb="md">Strategic Advice</Title>

      <Title order={3} mb="sm">Start Simple</Title>
      <Text mb="md">
        Begin with basic approaches to establish baselines:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Linear Models:</strong> Try logistic regression or simple linear classifiers first.
          They are fast and can achieve ~80% accuracy.
        </List.Item>
        <List.Item>
          <strong>Shallow Networks:</strong> Small MLPs with 1-2 hidden layers can reach 85-90% accuracy
          while training quickly.
        </List.Item>
        <List.Item>
          <strong>Monitor Everything:</strong> Track time, memory, and accuracy for each experiment.
        </List.Item>
      </List>

      <Title order={3} mb="sm" mt="lg">Optimize Systematically</Title>
      <Text mb="md">
        Once you have a baseline, improve methodically:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Hyperparameter Tuning:</strong> Experiment with learning rates, batch sizes, and epochs.
          Balance speed vs. accuracy.
        </List.Item>
        <List.Item>
          <strong>Model Architecture:</strong> Try different layer configurations.
          <Anchor href="https://www.raphaelcousin.com/courses/python-deep-learning/module3/course/essential-layers" target="_blank">
            Learn about essential layers
          </Anchor>
        </List.Item>
        <List.Item>
          <strong>Optimizers:</strong> Compare SGD, Adam, RMSprop. Some converge faster than others.
        </List.Item>
        <List.Item>
          <strong>Early Stopping:</strong> Stop training when validation accuracy plateaus to save time.
        </List.Item>
      </List>

      <Title order={3} mb="sm" mt="lg">Advanced Approaches</Title>
      <Text mb="md">
        When you reach the resource limits, explore more sophisticated techniques:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Feature Engineering:</strong> Modify the input features to provide more insights.
        </List.Item>
        <List.Item>
          <strong>Ensemble Methods:</strong> Combine multiple weak learners trained on data subsets.
        </List.Item>
        <List.Item>
          <strong>Model Compression:</strong> Use knowledge distillation or pruning to reduce model size.
        </List.Item>
          <List.Item>
          <strong>Meta-Learning:</strong> Explore MAML, Reptile, or other fast adaptation algorithms.
        </List.Item>
      </List>

      <Alert icon={<IconInfoCircle />} color="green" mb="xl">
        <Text weight={500}>Pro Tips</Text>
        <List spacing="xs" size="sm" mt="xs">
          <List.Item>Profile your code to identify bottlenecks</List.Item>
          <List.Item>Use vectorized operations (NumPy) instead of loops</List.Item>
          <List.Item>Precompute what you can outside the training loop</List.Item>
          <List.Item>Consider using subset of training data if time is tight</List.Item>
        </List>
      </Alert>

      <Title order={2} mb="md">Testing Your Agent</Title>

      <Title order={3} mb="sm">Local Testing</Title>
      <Text mb="md">
        Test your agent locally with the provided environment:
      </Text>
      <CodeBlock
        code={`from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
from your_package.agent import Agent
import time

# Create environment
env = PermutedMNISTEnv(number_episodes=10)
env.set_seed(42)

# Initialize your agent
agent = Agent()

# Track performance
total_time = 0
accuracies = []

for episode in range(10):
    task = env.get_next_task()
    if task is None:
        break

    agent.reset()

    start = time.time()
    agent.train(task['X_train'], task['y_train'])
    predictions = agent.predict(task['X_test'])
    elapsed = time.time() - start

    accuracy = env.evaluate(predictions, task['y_test'])
    accuracies.append(accuracy)
    total_time += elapsed

    print(f"Task {episode+1}: {accuracy:.2%} in {elapsed:.2f}s")

print(f"Average: {np.mean(accuracies):.2%}")
print(f"Total time: {total_time:.2f}s")
print(f"Status: {'PASS' if total_time < 600 else 'FAIL'}") `}
        language="python"
      />

      <Title order={3} mb="sm" mt="lg">Competition Submission</Title>
      <Text mb="md">
        Submit your agent to ML-Arena for evaluation:
      </Text>
      <List spacing="sm" mb="md" type="ordered">
        <List.Item>
          <strong>Competition Page:</strong> Visit{' '}
          <Anchor href="https://ml-arena.com/viewcompetition/8" target="_blank">
            ML-Arena Permuted MNIST Competition
          </Anchor>
        </List.Item>
        <List.Item>
          <strong>Create Account:</strong> Sign up or connect with your GitHub account
        </List.Item>
        <List.Item>
          <strong>Submit Agent:</strong> Click "Submit Agent" at the top:
          <List withPadding mt="xs">
            <List.Item>Upload your <code>agent.py</code> file with the required class interface</List.Item>
            <List.Item>Add any additional Python files (model weights, utilities, etc.)</List.Item>
            <List.Item>Select the appropriate kernel (sklearn, pytorch, tensorflow, etc.)</List.Item>
          </List>
        </List.Item>
        <List.Item>
          <strong>Evaluation Process:</strong>
          <List withPadding mt="xs">
            <List.Item>Initial evaluation on 10 random permutations</List.Item>
            <List.Item>Top performers undergo additional evaluations for final ranking</List.Item>
            <List.Item>View your score and rank on the leaderboard</List.Item>
          </List>
        </List.Item>
      </List>

      <Title order={2} mb="md">Building a Reproducible Project</Title>
      <Text mb="md">
        Structure your project for reproducibility and experimentation:
      </Text>

      <Title order={3} mb="sm">Example Structure</Title>
      <CodeBlock
        code={`your_repo/
├── report.ipynb          # Your main results in less than 10pages explaining methodology and results
├── agent.py              # Your main agent class
├── requirements.txt      # Dependencies
├── pyproject.toml           # Package setup
├── experiments/
│   ├── baseline.py      # Baseline experiments
│   ├── hyperparameter_search.py
│   └── results/         # Experiment logs (hyperparams, accuracy, time, weights)
├── models/
│   ├── linear.py        # Different model architectures
│   ├── mlp.py
│   └── ensemble.py
├── utils/
│   ├── monitoring.py    # Resource monitoring
│   ├── visualization.py # Performance plots
│   └── optimization.py  # Training utilities
├── notebooks/
│   ├── exploration.ipynb # Data analysis
│   └── evaluation.ipynb  # Results analysis
└── tests/
    └── test_agent.py    # Unit tests`}
        language="text"
      />

      <Title order={3} mb="sm" mt="lg">Key Components</Title>
      <List spacing="sm" mb="md">
        <List.Item>
          <strong>Experiment Tracking:</strong> Log all experiments with hyperparameters and results
        </List.Item>
        <List.Item>
          <strong>Resource Monitoring:</strong> Track memory and time usage for each run
        </List.Item>
        <List.Item>
          <strong>Reproducibility:</strong> Set random seeds and document exact configurations
        </List.Item>
        <List.Item>
          <strong>Modular Design:</strong> Separate models, training logic, and utilities
        </List.Item>
      </List>

      <Title order={2} mb="md">Success Criteria</Title>
      <Text mb="md">A successful project will demonstrate:</Text>
      <List spacing="sm" mb="md">
        <List.Item>Working agent that meets the 1-minute constraint</List.Item>
        <List.Item>Clear progression from simple to complex approaches</List.Item>
        <List.Item>Systematic experimentation with logged results</List.Item>
        <List.Item>Understanding of the accuracy vs. speed trade-off</List.Item>
        <List.Item>Clean, documented, and reproducible code</List.Item>
        <List.Item>Thoughtful analysis of what works and why</List.Item>
      </List>

      <Alert icon={<IconBrain />} color="blue">
        <Text weight={500}>Remember</Text>
        <Text size="sm">
          This challenge is about finding the sweet spot between model complexity and training efficiency.
          The best solution is not necessarily the most complex one, but the one that makes optimal use
          of the limited resources available.
        </Text>
      </Alert>

      <Title order={2} mb="md" mt="xl">Additional Resources</Title>
      <List spacing="sm">
        <List.Item>
          <strong>Competition:</strong>{' '}
          <Anchor href="https://ml-arena.com/viewcompetition/8" target="_blank">
            ML-Arena Permuted MNIST Competition
          </Anchor> - Submit and track your progress
        </List.Item>
        <List.Item>
          <strong>Environment Package:</strong>{' '}
          <Anchor href="https://github.com/ml-arena/permuted_mnist" target="_blank">
            Official GitHub Repository
          </Anchor> - Complete environment code, starter files, and examples
        </List.Item>
        <List.Item>
          <strong>Environment Source:</strong>{' '}
          <Anchor href="https://github.com/ml-arena/permuted_mnist/blob/main/permuted_mnist/env/permuted_mnist.py" target="_blank">
            permuted_mnist.py
          </Anchor> - View the environment implementation
        </List.Item>
        <List.Item>
          <strong>Course Materials:</strong>{' '}
          <Anchor href="https://www.raphaelcousin.com/courses/data-science-practice/module6/course" target="_blank">
            Module 6: Advanced ML Techniques
          </Anchor>
        </List.Item>
        <List.Item>
          <strong>Deep Learning Guide:</strong>{' '}
          <Anchor href="https://www.raphaelcousin.com/courses/python-deep-learning/module3/course/essential-layers" target="_blank">
            Essential Layers
          </Anchor> - Neural network components
        </List.Item>
      </List>
    </Container>
  );
};

export default PermutedMNIST;