import React from 'react';
import { Container, Title, Text, List, Paper, Alert, Tabs } from '@mantine/core';
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

    </Container>
  );
};

export default DataPipelineTrainingLoop;