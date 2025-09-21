import React from 'react';
import { Container, Title, Text, List, Code, Alert } from '@mantine/core';
import { IconBrain, IconInfoCircle } from '@tabler/icons-react';
import CodeBlock from 'components/CodeBlock';

const PermutedMNIST = () => {
  return (
    <Container size="lg" py="xl">
      <Title order={1} mb="lg">
        <IconBrain size={32} style={{ marginRight: '10px', verticalAlign: 'middle' }} />
        Option A: Permuted MNIST - "Do More with Less"
      </Title>

      <Title order={2} mb="md">Challenge Overview</Title>
      <Text mb="md">
        The Permuted MNIST challenge focuses on efficiency and resource optimization. Your task is to solve
        the Permuted MNIST classification problem while minimizing computational resources.
      </Text>

      <Alert icon={<IconInfoCircle />} color="blue" mb="xl">
        <strong>Goal:</strong> Achieve the highest accuracy with the most efficient resource utilization
        (RAM usage, CPU time, training duration).
      </Alert>

      <Title order={2} mb="md">Problem Description</Title>
      <Text mb="md">
        Permuted MNIST is a variant of the classic MNIST digit recognition task where:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item>Each pixel in the 28×28 image is randomly permuted using a fixed permutation</List.Item>
        <List.Item>The same permutation is applied to all images in the dataset</List.Item>
        <List.Item>This breaks spatial locality, making convolutional approaches less effective</List.Item>
        <List.Item>The challenge tests model efficiency rather than architectural complexity</List.Item>
      </List>

      <Title order={2} mb="md">Dataset Characteristics</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>Input:</strong> 784-dimensional vectors (flattened 28×28 images)</List.Item>
        <List.Item><strong>Output:</strong> 10 classes (digits 0-9)</List.Item>
        <List.Item><strong>Training set:</strong> 60,000 samples</List.Item>
        <List.Item><strong>Test set:</strong> 10,000 samples</List.Item>
        <List.Item><strong>Permutation:</strong> Fixed random permutation applied consistently</List.Item>
      </List>

      <Title order={2} mb="md">Evaluation Metrics</Title>
      <Text mb="md">
        Your solution will be evaluated on multiple criteria to balance performance and efficiency:
      </Text>

      <Title order={3} mb="sm">Primary Metrics</Title>
      <List spacing="sm" mb="md">
        <List.Item><strong>Test Accuracy:</strong> Classification accuracy on the test set</List.Item>
        <List.Item><strong>Training Time:</strong> Total time required for model training</List.Item>
        <List.Item><strong>Memory Usage:</strong> Peak RAM consumption during training and inference</List.Item>
        <List.Item><strong>Model Size:</strong> Number of parameters in the final model</List.Item>
      </List>

      <Title order={3} mb="sm">Efficiency Score</Title>
      <Text mb="md">
        Teams will be ranked based on a composite efficiency score that considers:
      </Text>
      <List spacing="sm" mb="xl">
        <List.Item>Accuracy-to-parameter ratio</List.Item>
        <List.Item>Accuracy-to-training-time ratio</List.Item>
        <List.Item>Memory efficiency during inference</List.Item>
      </List>

      <Title order={2} mb="md">Technical Requirements</Title>

      <Title order={3} mb="sm">Data Loading</Title>
      <Text mb="md">Your package should handle data loading and permutation:</Text>
      <CodeBlock
        code={`import torch
from torchvision import datasets, transforms

# Apply fixed permutation to MNIST
def create_permuted_mnist(seed=42):
    torch.manual_seed(seed)
    permutation = torch.randperm(784)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28))
    ])`}
        language="python"
      />

      <Title order={3} mb="sm">Model Architecture</Title>
      <Text mb="md">
        Consider efficient architectures suitable for this task:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Fully Connected Networks:</strong> Simple MLPs with dropout and batch normalization</List.Item>
        <List.Item><strong>Linear Models:</strong> Logistic regression with feature engineering</List.Item>
        <List.Item><strong>Efficient Neural Networks:</strong> MobileNet-inspired architectures</List.Item>
        <List.Item><strong>Ensemble Methods:</strong> Lightweight model combinations</List.Item>
      </List>

      <Title order={3} mb="sm">Optimization Strategies</Title>
      <List spacing="sm" mb="xl">
        <List.Item><strong>Model Compression:</strong> Pruning, quantization, knowledge distillation</List.Item>
        <List.Item><strong>Training Efficiency:</strong> Learning rate scheduling, early stopping</List.Item>
        <List.Item><strong>Data Efficiency:</strong> Data augmentation, transfer learning</List.Item>
        <List.Item><strong>Hardware Optimization:</strong> Vectorization, batch processing</List.Item>
      </List>

      <Title order={2} mb="md">Baseline Performance</Title>
      <Text mb="md">
        Your solution should exceed these baseline benchmarks:
      </Text>
      <List spacing="sm" mb="md">
        <List.Item><strong>Simple MLP:</strong> ~95% accuracy, 100K parameters</List.Item>
        <List.Item><strong>Logistic Regression:</strong> ~85% accuracy, 7K parameters</List.Item>
        <List.Item><strong>Random Forest:</strong> ~90% accuracy, variable complexity</List.Item>
      </List>

      <Title order={2} mb="md">Package Structure</Title>
      <Text mb="md">Your package should include:</Text>
      <List spacing="sm" mb="md">
        <List.Item><Code>src/</Code> - Main source code with modular components</List.Item>
        <List.Item><Code>models/</Code> - Model architectures and training logic</List.Item>
        <List.Item><Code>utils/</Code> - Data loading, preprocessing, evaluation utilities</List.Item>
        <List.Item><Code>experiments/</Code> - Training scripts and configuration files</List.Item>
        <List.Item><Code>notebooks/</Code> - Jupyter notebook demonstrating usage</List.Item>
        <List.Item><Code>tests/</Code> - Unit tests for reproducibility</List.Item>
      </List>

      <Title order={2} mb="md">Success Criteria</Title>
      <Text mb="md">A successful submission will demonstrate:</Text>
      <List spacing="sm">
        <List.Item>Clear trade-offs between accuracy and efficiency</List.Item>
        <List.Item>Systematic experimentation and optimization</List.Item>
        <List.Item>Reproducible results with proper documentation</List.Item>
        <List.Item>Clean, modular code following best practices</List.Item>
        <List.Item>Thorough analysis of model behavior and performance</List.Item>
      </List>
    </Container>
  );
};

export default PermutedMNIST;