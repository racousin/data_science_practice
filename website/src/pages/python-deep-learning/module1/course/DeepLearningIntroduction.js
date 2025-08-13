import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const DeepLearningIntroduction = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        {/* What is Deep Learning */}
        <div id="what-is-deep-learning">
          <Title order={1} className="mb-6">
            Deep Learning Introduction
          </Title>
          <Text size="xl" className="mb-6">
            Understanding Deep Learning: Historical Context and Evolution
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">What is Deep Learning?</Title>
            <Text size="lg" className="mb-4">
              Deep Learning is a subset of machine learning that uses artificial neural networks 
              with multiple layers (hence "deep") to model and understand complex patterns in data.
            </Text>
            <List>
              <List.Item><strong>Neural Networks:</strong> Computational models inspired by biological neural networks</List.Item>
              <List.Item><strong>Deep Architecture:</strong> Multiple hidden layers enabling hierarchical feature learning</List.Item>
              <List.Item><strong>Representation Learning:</strong> Automatic feature extraction from raw data</List.Item>
              <List.Item><strong>End-to-End Learning:</strong> Direct optimization from input to output</List.Item>
            </List>
          </Paper>

          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Historical Milestones</Title>
                <List size="sm">
                  <List.Item><strong>1943:</strong> McCulloch-Pitts neuron model</List.Item>
                  <List.Item><strong>1957:</strong> Perceptron algorithm (Rosenblatt)</List.Item>
                  <List.Item><strong>1986:</strong> Backpropagation popularized</List.Item>
                  <List.Item><strong>2006:</strong> Deep learning renaissance (Hinton)</List.Item>
                  <List.Item><strong>2012:</strong> AlexNet breaks ImageNet records</List.Item>
                  <List.Item><strong>2017:</strong> Transformer architecture ("Attention is All You Need")</List.Item>
                  <List.Item><strong>2020s:</strong> Large Language Models (GPT, BERT)</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">Key Breakthroughs</Title>
                <List size="sm">
                  <List.Item><strong>Universal Approximation:</strong> Neural networks can approximate any continuous function</List.Item>
                  <List.Item><strong>Gradient-Based Learning:</strong> Efficient optimization through backpropagation</List.Item>
                  <List.Item><strong>GPU Acceleration:</strong> Parallel computation enables large-scale training</List.Item>
                  <List.Item><strong>Big Data:</strong> Large datasets fuel model performance</List.Item>
                  <List.Item><strong>Architectural Innovations:</strong> CNNs, RNNs, Transformers</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* PyTorch Ecosystem */}
        <div id="pytorch-ecosystem">
          <Title order={2} className="mb-6">PyTorch Ecosystem Overview</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">PyTorch Philosophy</Title>
            <Text className="mb-4">
              PyTorch was designed with three core principles: <strong>Pythonic</strong>, <strong>Dynamic</strong>, and <strong>Fast</strong>.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={4}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} className="mb-3">Pythonic</Title>
                  <List size="sm">
                    <List.Item>Natural Python syntax</List.Item>
                    <List.Item>Easy debugging with standard tools</List.Item>
                    <List.Item>Seamless integration with Python ecosystem</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-4 bg-green-50">
                  <Title order={4} className="mb-3">Dynamic</Title>
                  <List size="sm">
                    <List.Item>Dynamic computation graphs</List.Item>
                    <List.Item>Define-by-run execution</List.Item>
                    <List.Item>Flexible model architectures</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-4 bg-yellow-50">
                  <Title order={4} className="mb-3">Fast</Title>
                  <List size="sm">
                    <List.Item>Optimized C++ backend</List.Item>
                    <List.Item>GPU acceleration</List.Item>
                    <List.Item>JIT compilation</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Title order={3} className="mb-4">PyTorch Ecosystem Components</Title>
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} className="mb-3">Core Libraries</Title>
                <CodeBlock language="python" code={`# Core PyTorch
import torch                    # Tensor operations, autograd
import torch.nn as nn          # Neural network modules
import torch.optim as optim    # Optimization algorithms
import torch.nn.functional as F # Function API

# Data utilities
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Computer vision
import torchvision
import torchvision.transforms as transforms

# Audio processing  
import torchaudio

# Text processing
import torchtext`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} className="mb-3">Extended Ecosystem</Title>
                <List size="sm">
                  <List.Item><strong>PyTorch Lightning:</strong> High-level framework for research</List.Item>
                  <List.Item><strong>Transformers (Hugging Face):</strong> Pre-trained transformer models</List.Item>
                  <List.Item><strong>PyTorch Geometric:</strong> Graph neural networks</List.Item>
                  <List.Item><strong>Captum:</strong> Model interpretability</List.Item>
                  <List.Item><strong>TorchServe:</strong> Model serving and deployment</List.Item>
                  <List.Item><strong>PyTorch Mobile:</strong> Mobile deployment</List.Item>
                  <List.Item><strong>FairScale:</strong> Large-scale training utilities</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Framework Comparison */}
        <div id="comparison-frameworks">
          <Title order={2} className="mb-6">Comparison with Other Frameworks</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">PyTorch vs TensorFlow vs JAX</Title>
            
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8f9fa' }}>
                    <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Aspect</th>
                    <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>PyTorch</th>
                    <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>TensorFlow</th>
                    <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>JAX</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Execution Model</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Dynamic (Eager)</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Static + Eager</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Functional + JIT</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Learning Curve</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Moderate</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Steep</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Steep</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Research Friendliness</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Excellent</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Good</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Excellent</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Production Deployment</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Good</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Excellent</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Limited</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Community</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Large (Research)</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Large (Industry)</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Growing</td>
                  </tr>
                  <tr>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}><strong>Debugging</strong></td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Easy</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Moderate</td>
                    <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>Moderate</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Paper>

          <Grid gutter="lg">
            <Grid.Col span={4}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">When to Choose PyTorch</Title>
                <List size="sm">
                  <List.Item>Research and experimentation</List.Item>
                  <List.Item>Rapid prototyping</List.Item>
                  <List.Item>Complex model architectures</List.Item>
                  <List.Item>Educational purposes</List.Item>
                  <List.Item>Dynamic computation needs</List.Item>
                  <List.Item>Python-first development</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={4}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">When to Choose TensorFlow</Title>
                <List size="sm">
                  <List.Item>Large-scale production deployment</List.Item>
                  <List.Item>Mobile and edge deployment</List.Item>
                  <List.Item>Established MLOps pipelines</List.Item>
                  <List.Item>JavaScript integration (TensorFlow.js)</List.Item>
                  <List.Item>Google Cloud ecosystem</List.Item>
                  <List.Item>Mature tooling requirements</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={4}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">When to Choose JAX</Title>
                <List size="sm">
                  <List.Item>High-performance computing</List.Item>
                  <List.Item>Scientific computing</List.Item>
                  <List.Item>Functional programming preference</List.Item>
                  <List.Item>Advanced differentiation needs</List.Item>
                  <List.Item>XLA optimization benefits</List.Item>
                  <List.Item>NumPy-like interface</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Summary */}
        <div>
          <Title order={2} className="mb-8">Summary: Deep Learning Introduction</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} className="mb-4">Key Takeaways</Title>
                <List spacing="md">
                  <List.Item>Deep learning uses multi-layer neural networks for complex pattern recognition</List.Item>
                  <List.Item>Historical evolution from perceptrons to modern transformers</List.Item>
                  <List.Item>PyTorch emphasizes research flexibility and Python integration</List.Item>
                  <List.Item>Framework choice depends on research vs production needs</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} className="mb-4">Why PyTorch for This Course</Title>
                <List spacing="md">
                  <List.Item>Intuitive Python-like syntax for mathematical concepts</List.Item>
                  <List.Item>Dynamic graphs enable flexible experimentation</List.Item>
                  <List.Item>Strong research community and recent innovations</List.Item>
                  <List.Item>Excellent debugging and visualization tools</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6 text-center">
            <Text size="lg" className="font-semibold">
              Deep learning represents a paradigm shift in how we approach complex data problems
            </Text>
            <Text className="mt-2">
              PyTorch provides the mathematical foundation and practical tools to master these concepts
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default DeepLearningIntroduction;