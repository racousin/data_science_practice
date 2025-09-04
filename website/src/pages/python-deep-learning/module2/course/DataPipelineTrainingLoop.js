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

{/* add transform? */}
    </Container>
  );
};

export default DataPipelineTrainingLoop;