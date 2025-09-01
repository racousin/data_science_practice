import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const PytorchDataAndModels = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Introduction */}
        <div data-slide>
          <Title order={1} mb="xl">
            PyTorch: Data Management and Neural Networks
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 mb-6">
            <Title order={2} className="mb-4">From Data to Models</Title>
            <Text size="lg" mb="md">
              Building effective deep learning systems requires mastering three core components: 
              efficient data loading and preprocessing, powerful data transformations, and 
              well-architected neural network models. This section covers PyTorch's tools 
              for each of these essential elements.
            </Text>
            
            <Flex direction="column" align="center" mb="md">
              <Image
                src="/assets/python-deep-learning/module1/data_pipeline.png"
                alt="PyTorch Data Pipeline"
                w={{ base: 400, sm: 600, md: 800 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              PyTorch data processing pipeline: from raw data to trained models
            </Text>
          </Paper>
        </div>

        {/* Datasets and DataLoaders */}
        <div data-slide>
          <Title order={2} className="mb-6" id="datasets-dataloaders">
            Datasets and DataLoaders
          </Title>
          
          <Paper className="p-6 bg-green-50 mb-6">
            <Title order={3} className="mb-4">The Dataset Abstraction</Title>
            <Text size="lg" mb="md">
              PyTorch's Dataset class provides a clean abstraction for working with data. 
              It handles indexing, loading, and preprocessing, making it easy to work with 
              any type of data - images, text, audio, or tabular data.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Dataset Interface</Title>
                  <Text size="sm" className="mb-3">
                    Every PyTorch Dataset must implement two key methods:
                  </Text>
                  <List size="sm">
                    <List.Item><strong>__len__():</strong> Return the size of the dataset</List.Item>
                    <List.Item><strong>__getitem__(idx):</strong> Return a sample given an index</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Built-in Datasets</Title>
                  <Text size="sm" className="mb-3">
                    PyTorch provides many pre-built datasets for common tasks:
                  </Text>
                  <List size="sm">
                    <List.Item><strong>Vision:</strong> MNIST, CIFAR-10/100, ImageNet</List.Item>
                    <List.Item><strong>Text:</strong> IMDb, AG News, WikiText</List.Item>
                    <List.Item><strong>Audio:</strong> LibriSpeech, CommonVoice</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`import torchvision.datasets as datasets
import torchvision.transforms as transforms

# CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print(f"Dataset size: {len(cifar10_train)}")
sample, label = cifar10_train[0]
print(f"Sample shape: {sample.shape}")  # [3, 32, 32]`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-6 bg-orange-50 mb-6">
            <Title order={3} className="mb-4">DataLoader: Efficient Batch Processing</Title>
            <Text className="mb-4">
              The DataLoader wraps a Dataset and provides powerful features for training: 
              batching, shuffling, parallel loading, and memory management.
            </Text>
            
            <Flex direction="column" align="center" mb="md">
              <Image
                src="/assets/python-deep-learning/module1/dataloader_process.png"
                alt="DataLoader Process"
                w={{ base: 400, sm: 600, md: 700 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              DataLoader workflow: from dataset to training batches
            </Text>
            
            <CodeBlock language="python" code={`from torch.utils.data import DataLoader
import torch

# Create a simple dataset
class NumbersDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)  # Random data
        self.labels = torch.randint(0, 2, (size,))  # Binary labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and dataloader
dataset = NumbersDataset(1000)
dataloader = DataLoader(
    dataset,
    batch_size=32,          # Process 32 samples at once
    shuffle=True,           # Shuffle data each epoch
    num_workers=4,          # Use 4 processes for loading
    pin_memory=True,        # Faster GPU transfer
    drop_last=True          # Drop incomplete final batch
)

# Training loop
for epoch in range(5):
    for batch_idx, (data, targets) in enumerate(dataloader):
        # data.shape: [32, 10]
        # targets.shape: [32]
        
        # Your training code here
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        if batch_idx == 0:  # First batch info
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}")
            print(f"  Data shape: {data.shape}")
            print(f"  Targets shape: {targets.shape}")`} />

            <Grid gutter="lg" className="mt-4">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Key DataLoader Parameters</Title>
                  <List size="sm">
                    <List.Item><strong>batch_size:</strong> Number of samples per batch</List.Item>
                    <List.Item><strong>shuffle:</strong> Randomize sample order</List.Item>
                    <List.Item><strong>num_workers:</strong> Parallel data loading processes</List.Item>
                    <List.Item><strong>pin_memory:</strong> Faster GPU memory transfer</List.Item>
                    <List.Item><strong>drop_last:</strong> Handle incomplete final batch</List.Item>
                    <List.Item><strong>collate_fn:</strong> Custom batch creation function</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Performance Tips</Title>
                  <List size="sm">
                    <List.Item>Use num_workers=4-8 for faster loading</List.Item>
                    <List.Item>Enable pin_memory for GPU training</List.Item>
                    <List.Item>Larger batch sizes improve GPU utilization</List.Item>
                    <List.Item>Cache preprocessed data when possible</List.Item>
                    <List.Item>Use SSDs for better I/O performance</List.Item>
                    <List.Item>Monitor CPU and GPU usage balance</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Linear Algebra Operations */}
        <div data-slide>
          <Title order={2} className="mb-6" id="linear-algebra">
            Essential Linear Algebra Operations
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-green-50 to-teal-50 mb-6">
            <Title order={3} className="mb-4">Fundamental Operations for Deep Learning</Title>
            <Text size="lg" mb="md">
              Deep learning operates on multi-dimensional arrays. Understanding their properties and operations 
              is crucial for implementing and debugging neural networks.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} mb="sm">Matrix Multiplication</Title>
                  <BlockMath>{`C = AB \\text{ where } C_{ij} = \\sum_k A_{ik}B_{kj}`}</BlockMath>
                  <Text size="sm" className="mb-2">Dimensions: <InlineMath>{`(m \\times n) \\cdot (n \\times p) = (m \\times p)`}</InlineMath></Text>
                  
                  <CodeBlock language="python" code={`# Matrix multiplication
A = torch.randn(10, 5)
B = torch.randn(5, 3)
C = torch.matmul(A, B)  # or A @ B
print(C.shape)  # torch.Size([10, 3])`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-green-50">
                  <Title order={4} mb="sm">Element-wise Operations</Title>
                  <BlockMath>{`C = A \\odot B \\text{ where } C_{ij} = A_{ij} \\cdot B_{ij}`}</BlockMath>
                  <Text size="sm" className="mb-2">Hadamard product (element-wise multiplication)</Text>
                  
                  <CodeBlock language="python" code={`# Element-wise operations
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Element-wise multiplication
C = A * B  # [[5, 12], [21, 32]]

# Broadcasting example
v = torch.tensor([1, 2])
E = A + v  # [[2, 4], [4, 6]]`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Broadcasting and Norms</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Broadcasting Rules</Title>
                  <Text size="sm" className="mb-3">
                    PyTorch automatically broadcasts tensors for element-wise operations:
                  </Text>
                  <List size="sm">
                    <List.Item>Compare shapes element-wise from right to left</List.Item>
                    <List.Item>Dimensions are compatible if equal or one is 1</List.Item>
                    <List.Item>Missing dimensions treated as 1</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Broadcasting examples
A = torch.randn(5, 3)     # Shape: [5, 3]
b = torch.randn(3)         # Shape: [3]
C = A + b                  # Shape: [5, 3]`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Vector Norms</Title>
                  <div className="space-y-3">
                    <div>
                      <Text fw="bold" size="sm">L2 Norm (Euclidean):</Text>
                      <BlockMath>{`||x||_2 = \\sqrt{\\sum_{i=1}^n x_i^2}`}</BlockMath>
                    </div>
                    
                    <div>
                      <Text fw="bold" size="sm">L1 Norm (Manhattan):</Text>
                      <BlockMath>{`||x||_1 = \\sum_{i=1}^n |x_i|`}</BlockMath>
                    </div>
                  </div>
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Transforms */}
        <div data-slide>
          <Title order={2} className="mb-6" id="transforms">
            Data Transforms and Preprocessing
          </Title>
          
          <Paper className="p-6 bg-purple-50 mb-6">
            <Title order={3} className="mb-4">The Power of Data Augmentation</Title>
            <Text size="lg" mb="md">
              Data transforms are essential for preprocessing and augmenting your data. 
              They can normalize inputs, apply augmentations for better generalization, 
              and convert between data formats. PyTorch transforms are composable and efficient.
            </Text>
            
            <Flex direction="column" align="center" mb="md">
              <Image
                src="/assets/python-deep-learning/module1/data_augmentation.png"
                alt="Data Augmentation Examples"
                w={{ base: 400, sm: 600, md: 800 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              Data augmentation techniques: rotation, flipping, cropping, and color changes
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Basic Transforms</Title>
                  
                  <CodeBlock language="python" code={`import torchvision.transforms as transforms
from PIL import Image

# Basic image transforms
basic_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to 224x224
    transforms.ToTensor(),              # Convert to tensor [0,1]
    transforms.Normalize(               # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply to an image
image = Image.open('example.jpg')
tensor = basic_transforms(image)
print(f"Transformed shape: {tensor.shape}")  # [3, 224, 224]`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Augmentation Transforms</Title>
                  
                  <CodeBlock language="python" code={`# Data augmentation for training
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),   # Random crop and resize
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance flip
    transforms.RandomRotation(10),       # Rotate ±10 degrees
    transforms.ColorJitter(              # Color variations
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-6 bg-teal-50 mb-6">
            <Title order={3} className="mb-4">Custom Transforms</Title>
            <Text className="mb-4">
              You can create custom transforms for specific preprocessing needs. Custom transforms 
              should be callable objects that take an input and return a transformed output.
            </Text>
            
            <CodeBlock language="python" code={`import torch
import random

class AddGaussianNoise:
    """Add Gaussian noise to tensor"""
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

class RandomCutout:
    """Randomly mask out a square region"""
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p
    
    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
            
        h, w = tensor.shape[-2:]
        y = random.randint(0, h - self.size)
        x = random.randint(0, w - self.size)
        
        # Create a copy and mask the region
        tensor = tensor.clone()
        tensor[:, y:y+self.size, x:x+self.size] = 0
        return tensor

# Use custom transforms
custom_transforms = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(std=0.01),
    RandomCutout(size=32, p=0.3),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Transform composition for different phases
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    custom_transforms
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Use with datasets
train_dataset = CustomDataset(train_data, train_labels, train_transform)
val_dataset = CustomDataset(val_data, val_labels, val_transform)`} />

            <Grid gutter="lg" className="mt-4">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Why Normalization Matters</Title>
                  <Text size="sm" className="mb-3">
                    Normalization helps with training stability and convergence:
                  </Text>
                  <List size="sm">
                    <List.Item>Centers data around zero mean</List.Item>
                    <List.Item>Scales variance to unit range</List.Item>
                    <List.Item>Prevents gradient explosion/vanishing</List.Item>
                    <List.Item>Enables higher learning rates</List.Item>
                    <List.Item>Improves training speed</List.Item>
                  </List>
                  
                  <BlockMath>{`x_{normalized} = \\frac{x - \\mu}{\\sigma}`}</BlockMath>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Augmentation Benefits</Title>
                  <Text size="sm" className="mb-3">
                    Data augmentation improves model generalization:
                  </Text>
                  <List size="sm">
                    <List.Item>Increases effective dataset size</List.Item>
                    <List.Item>Reduces overfitting</List.Item>
                    <List.Item>Teaches invariance to transformations</List.Item>
                    <List.Item>Improves robustness</List.Item>
                    <List.Item>Better test performance</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Building Neural Networks */}
        <div data-slide>
          <Title order={2} className="mb-6" id="neural-networks">
            Building Neural Networks with torch.nn
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-rose-50 to-pink-50 mb-6">
            <Title order={3} className="mb-4">PyTorch Neural Network Module</Title>
            <Text size="lg" mb="md">
              PyTorch provides the torch.nn module for building neural networks. Understanding 
              the basic building blocks and how they connect is essential for creating models.
            </Text>
            
            <Flex direction="column" align="center" mb="md">
              <Image
                src="/assets/python-deep-learning/module1/simple_mlp.png"
                alt="Simple Multi-Layer Perceptron"
                w={{ base: 400, sm: 600, md: 700 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              Simple Multi-Layer Perceptron architecture
            </Text>
          </Paper>

          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Understanding Linear Layers</Title>
            <Text className="mb-4">
              Linear layers are the fundamental building blocks of neural networks. They perform 
              matrix multiplication followed by bias addition.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Mathematical Operation</Title>
                  <BlockMath>{`y = xW^T + b`}</BlockMath>
                  <Text size="sm" className="mb-3">
                    Where:
                  </Text>
                  <List size="sm">
                    <List.Item><InlineMath>{`x \\in \\mathbb{R}^{n \\times d_{in}}`}</InlineMath> - input tensor</List.Item>
                    <List.Item><InlineMath>{`W \\in \\mathbb{R}^{d_{out} \\times d_{in}}`}</InlineMath> - weight matrix</List.Item>
                    <List.Item><InlineMath>{`b \\in \\mathbb{R}^{d_{out}}`}</InlineMath> - bias vector</List.Item>
                    <List.Item><InlineMath>{`y \\in \\mathbb{R}^{n \\times d_{out}}`}</InlineMath> - output tensor</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Dimension Example</Title>
                  <Text size="sm" className="mb-2">
                    For a linear layer with 784 inputs and 128 outputs:
                  </Text>
                  <List size="sm">
                    <List.Item>Input: <InlineMath>{`[32, 784]`}</InlineMath> (batch_size=32)</List.Item>
                    <List.Item>Weight: <InlineMath>{`[128, 784]`}</InlineMath></List.Item>
                    <List.Item>Bias: <InlineMath>{`[128]`}</InlineMath></List.Item>
                    <List.Item>Output: <InlineMath>{`[32, 128]`}</InlineMath></List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Linear layer example
layer = nn.Linear(784, 128)
print(f"Weight shape: {layer.weight.shape}")  # [128, 784]
print(f"Bias shape: {layer.bias.shape}")      # [128]

# Forward pass
x = torch.randn(32, 784)  # Batch of 32 samples
y = layer(x)              # Output: [32, 128]`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-6 bg-green-50 mb-6">
            <Title order={3} className="mb-4">Simple Multi-Layer Perceptron</Title>
            
            <CodeBlock language="python" code={`import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Create model instance
model = NeuralNetwork()
print(model)

# Test with random input
X = torch.rand(1, 28, 28)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")`} />
          </Paper>

          <Paper className="p-6 bg-amber-50 mb-6">
            <Title order={3} className="mb-4">Model Parameters and Components</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Accessing Parameters</Title>
                  <CodeBlock language="python" code={`# Iterate through model parameters
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")
    
# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Access specific layer
first_linear = model.linear_relu_stack[0]
print(f"First layer weight: {first_linear.weight.shape}")
print(f"First layer bias: {first_linear.bias.shape}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Common Activation Functions</Title>
                  <CodeBlock language="python" code={`# ReLU: most common activation
relu = nn.ReLU()

# Sigmoid: for binary classification output
sigmoid = nn.Sigmoid()

# Softmax: for multi-class classification
softmax = nn.Softmax(dim=1)

# Example usage
x = torch.randn(4, 5)
relu_output = relu(x)        # Negative values become 0
sigmoid_output = sigmoid(x)  # Values between 0 and 1
softmax_output = softmax(x)  # Probabilities sum to 1`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Next Steps */}
        <div data-slide>
          <Title order={2} className="mb-6" id="next-steps">
            Next Steps: Complete Training Pipeline
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-emerald-50 to-teal-50">
            <Title order={3} className="mb-4">What's Coming Next</Title>
            <Text size="lg" mb="md">
              You now understand the fundamental components of PyTorch: datasets, transforms, 
              and neural network building blocks. In the next sections, we'll cover:
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Automatic Differentiation</Title>
                  <List size="sm">
                    <List.Item>Mathematical foundations of autograd</List.Item>
                    <List.Item>Computational graphs and backpropagation</List.Item>
                    <List.Item>Advanced gradient computation techniques</List.Item>
                    <List.Item>Custom gradient functions</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Complete Training Pipeline</Title>
                  <List size="sm">
                    <List.Item>Optimization algorithms and schedulers</List.Item>
                    <List.Item>Training loops and best practices</List.Item>
                    <List.Item>Model evaluation and validation</List.Item>
                    <List.Item>Saving and loading models</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
            
            <Flex direction="column" align="center" className="mt-6">
              <Image
                src="/assets/python-deep-learning/module1/training_roadmap.png"
                alt="Training Pipeline Roadmap"
                w={{ base: 400, sm: 600, md: 700 }}
                h="auto"
                fluid
              />
            </Flex>
            <Text component="p" ta="center" mt="xs">
              The complete PyTorch learning journey: from foundations to deployment
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default PytorchDataAndModels;