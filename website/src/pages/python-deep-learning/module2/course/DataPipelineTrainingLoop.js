import React from 'react';
import { Container, Title, Text, List, Paper, Alert, Stack, Grid, Flex, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const DataPipelineTrainingLoop = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">Data Pipeline & Training Loop</Title>
      
      <Text>
        Training neural networks requires efficient data loading and a systematic training process. 
        PyTorch provides powerful tools to handle datasets and implement training loops.
      </Text>

      <Title order={2} mt="xl">1. Dataset and DataLoader</Title>
      
      <Title order={3} mt="md">Creating a Custom Dataset</Title>
      
      <Text>
        PyTorch's <code>Dataset</code> class provides an interface for accessing data. You need to implement 
        two methods: <code>__len__</code> and <code>__getitem__</code>:
      </Text>
      
      <CodeBlock language="python" code={`import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        # Generate random data
        self.X = torch.randn(size, 10)  # 10 features
        self.y = torch.randn(size, 1)   # 1 target
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]`} />

      <Text mt="md">
        Create an instance and access data:
      </Text>

      <CodeBlock language="python" code={`# Create dataset
dataset = SimpleDataset(size=1000)
print(f"Dataset size: {len(dataset)}")

# Access single sample
x_sample, y_sample = dataset[0]
print(f"Sample input shape: {x_sample.shape}")
print(f"Sample target shape: {y_sample.shape}")`} />

      <Title order={3} mt="xl">DataLoader for Batching</Title>
      
      <Text>
        <code>DataLoader</code> handles batching, shuffling, and parallel data loading:
      </Text>

      <CodeBlock language="python" code={`# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,      # Process 32 samples at once
    shuffle=True,       # Randomize order each epoch
    num_workers=2       # Parallel data loading
)`} />

      <Text mt="md">
        Iterate through batches:
      </Text>

      <CodeBlock language="python" code={`# Get one batch
for batch_x, batch_y in train_loader:
    print(f"Batch X shape: {batch_x.shape}")  # [32, 10]
    print(f"Batch y shape: {batch_y.shape}")  # [32, 1]
    break  # Just show first batch`} />

      <Title order={2} mt="xl">2. Building a Simple Model</Title>
      
      <Text>
        Let's create a simple neural network for our data:
      </Text>

      <CodeBlock language="python" code={`import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x`} />

      <Text mt="md">
        Initialize the model:
      </Text>

      <CodeBlock language="python" code={`model = SimpleModel()
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")`} />

      <Title order={2} mt="xl">3. Loss Function and Optimizer</Title>
      
      <Title order={3} mt="md">Choosing a Loss Function</Title>
      
      <Text>
        The loss function measures how wrong our predictions are:
      </Text>

      <CodeBlock language="python" code={`# Mean Squared Error for regression
loss_fn = nn.MSELoss()

# Example loss calculation
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss = loss_fn(predictions, targets)
print(f"Loss: {loss.item():.4f}")`} />

      <Title order={3} mt="xl">Setting up the Optimizer</Title>
      
      <Text>
        The optimizer updates model parameters based on gradients:
      </Text>

      <CodeBlock language="python" code={`# Stochastic Gradient Descent
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,        # Learning rate
    momentum=0.9    # Momentum for faster convergence
)`} />

      <Text mt="md">
        Alternative optimizers:
      </Text>

      <CodeBlock language="python" code={`# Adam optimizer (adaptive learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Different learning rates for different layers
optimizer = torch.optim.Adam([
    {'params': model.fc1.parameters(), 'lr': 0.001},
    {'params': model.fc2.parameters(), 'lr': 0.01}
])`} />

      <Title order={2} mt="xl">4. The Training Loop</Title>
      
      <Title order={3} mt="md">Basic Training Structure</Title>
      
      <Text>
        A training loop consists of four main steps: forward pass, loss computation, backward pass, and parameter update:
      </Text>

      <CodeBlock language="python" code={`def train_one_epoch(model, data_loader, loss_fn, optimizer):
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # 1. Zero gradients from previous step
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(inputs)
        
        # 3. Compute loss
        loss = loss_fn(outputs, targets)
        
        # 4. Backward pass (compute gradients)
        loss.backward()
        
        # 5. Update parameters
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss`} />

      <Title order={3} mt="xl">Complete Training Script</Title>
      
      <Text>
        Putting it all together with training and validation:
      </Text>

      <CodeBlock language="python" code={`# Prepare data
train_dataset = SimpleDataset(size=800)
val_dataset = SimpleDataset(size=200)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss, optimizer
model = SimpleModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 10
train_losses = []
val_losses = []`} />

      <Text mt="md">
        The main training loop:
      </Text>

      <CodeBlock language="python" code={`for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()  # Set to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Don't compute gradients
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    
    # Average losses
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")`} />

      <Title order={2} mt="xl">5. Device Management (CPU/GPU)</Title>
      
      <Text>
        Move model and data to GPU for faster training:
      </Text>

      <CodeBlock language="python" code={`# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Update training loop to use device
for inputs, targets in train_loader:
    # Move data to device
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()`} />

      <Title order={2} mt="xl">6. Monitoring Training</Title>
      
      <Title order={3} mt="md">Tracking Metrics</Title>
      
      <Text>
        Monitor training progress with additional metrics:
      </Text>

      <CodeBlock language="python" code={`from torch.utils.tensorboard import SummaryWriter

# Initialize tensorboard writer
writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # ... training code ...
    
    # Log metrics
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    
    # Log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning_rate', current_lr, epoch)

writer.close()`} />

      <Title order={3} mt="xl">Early Stopping</Title>
      
      <Text>
        Stop training when validation loss stops improving:
      </Text>

      <CodeBlock language="python" code={`class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0`} />

      <Text mt="md">
        Use early stopping in training:
      </Text>

      <CodeBlock language="python" code={`early_stopping = EarlyStopping(patience=5)

for epoch in range(num_epochs):
    # ... training code ...
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break`} />

      <Title order={2} mt="xl">7. Saving and Loading Models</Title>
      
      <Text>
        Save model checkpoints during training:
      </Text>

      <CodeBlock language="python" code={`# Save model state
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load model state
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']`} />

      <Title order={2} mt="xl">8. Complete Training Template</Title>
      
      <Text>
        Here's a reusable training function:
      </Text>

      <CodeBlock language="python" code={`def train_model(model, train_loader, val_loader, num_epochs=10, 
                device='cpu', lr=0.001):
    """Complete training function with best practices"""
    
    # Setup
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} [{batch_idx*len(inputs)}/{len(train_loader.dataset)}]'
                      f' Loss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved best model!')
    
    return model`} />

      <Paper p="md" withBorder mt="xl">
        <Title order={3}>Key Concepts Summary</Title>
        <List mt="sm">
          <List.Item>
            <strong>Dataset</strong>: Provides interface to access data samples
          </List.Item>
          <List.Item>
            <strong>DataLoader</strong>: Handles batching, shuffling, and parallel loading
          </List.Item>
          <List.Item>
            <strong>Training Loop</strong>: Forward pass → Loss → Backward pass → Update
          </List.Item>
          <List.Item>
            <strong>Validation</strong>: Evaluate without gradients using <code>torch.no_grad()</code>
          </List.Item>
          <List.Item>
            <strong>Device Management</strong>: Move model and data to GPU with <code>.to(device)</code>
          </List.Item>
          <List.Item>
            <strong>Monitoring</strong>: Track losses, use early stopping, save checkpoints
          </List.Item>
        </List>
      </Paper>

      <Alert icon="💡" title="Best Practices" color="blue" mt="xl">
        <List>
          <List.Item>Always set <code>model.train()</code> and <code>model.eval()</code> appropriately</List.Item>
          <List.Item>Zero gradients before each backward pass</List.Item>
          <List.Item>Use <code>torch.no_grad()</code> during validation to save memory</List.Item>
          <List.Item>Save best model based on validation performance</List.Item>
          <List.Item>Monitor for overfitting by comparing train and validation losses</List.Item>
        </List>
      </Alert>
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
              style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
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
          <Title order={2} mb="xl" id="datasets-dataloaders">
            Datasets and DataLoaders
          </Title>
          
          <Paper className="p-6 bg-green-50 mb-6">
            <Title order={3} mb="md">The Dataset Abstraction</Title>
            <Text size="lg" mb="md">
              PyTorch's Dataset class provides a clean abstraction for working with data. 
              It handles indexing, loading, and preprocessing, making it easy to work with 
              any type of data - images, text, audio, or tabular data.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Dataset Interface</Title>
                  <Text size="sm" mb="md">
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
                  <Text size="sm" mb="md">
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
            <Title order={3} mb="md">DataLoader: Efficient Batch Processing</Title>
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


        {/* Transforms */}
        <div data-slide>
          <Title order={2} mb="xl" id="transforms">
            Data Transforms and Preprocessing
          </Title>
          
          <Paper className="p-6 bg-purple-50 mb-6">
            <Title order={3} mb="md">The Power of Data Augmentation</Title>
            <Text size="lg" mb="md">
              Data transforms are essential for preprocessing and augmenting your data. 
              They can normalize inputs, apply augmentations for better generalization, 
              and convert between data formats. PyTorch transforms are composable and efficient.
            </Text>
            
            <Flex direction="column" align="center" mb="md">
              <Image
                src="/assets/python-deep-learning/module1/data_augmentation.png"
                alt="Data Augmentation Examples"
                style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
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
            <Title order={3} mb="md">Custom Transforms</Title>
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
                  <Text size="sm" mb="md">
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
                  <Text size="sm" mb="md">
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
          <Title order={2} mb="xl" id="neural-networks">
            Building Neural Networks with torch.nn
          </Title>
          
          <Paper className="p-6 bg-gradient-to-r from-rose-50 to-pink-50 mb-6">
            <Title order={3} mb="md">PyTorch Neural Network Module</Title>
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
            <Title order={3} mb="md">Understanding Linear Layers</Title>
            <Text className="mb-4">
              Linear layers are the fundamental building blocks of neural networks. They perform 
              matrix multiplication followed by bias addition.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Mathematical Operation</Title>
                  <BlockMath>{`y = xW^T + b`}</BlockMath>
                  <Text size="sm" mb="md">
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
                  <Text size="sm" mb="xs">
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
            <Title order={3} mb="md">Simple Multi-Layer Perceptron</Title>
            
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
            <Title order={3} mb="md">Model Parameters and Components</Title>
            
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

      </Stack>
    </Container>
  );
};

export default DataPipelineTrainingLoop;