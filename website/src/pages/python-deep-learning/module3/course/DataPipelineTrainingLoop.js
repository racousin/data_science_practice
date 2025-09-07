import React from 'react';
import { Container, Title, Text, List, Code, Stack } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';


import WeightInitialization from './EssentialComponents/WeightInitialization';
import Optimization from './EssentialComponents/Optimization';
import EarlyStopping from './EssentialComponents/EarlyStopping';
import CustomLoss from './EssentialComponents/CustomLoss';
import ReduceLROnPlateau from './EssentialComponents/ReduceLROnPlateau';

const DataPipelineTrainingLoop = () => {
  return (
    <Container fluid>
      <Stack spacing="lg">
        <div data-slide>
        <Title order={1}>Data Pipeline Essential Components</Title>
        </div>
         <div data-slide>
          <Title order={2}>Core Components</Title>
          
          <Title order={3} mt="md">nn.Module Overview</Title>
          <Text>
            Base class for all neural network components. Handles parameters and gradients automatically.
          </Text>
          
          <Title order={3} mt="md">nn.Module Core Features</Title>
          <Text>
            The nn.Module class provides essential functionality for all neural network layers and models:
          </Text>
          <List>
            <List.Item><strong>Automatic Parameter Management:</strong> Registers all trainable parameters</List.Item>
            <List.Item><strong>GPU Movement:</strong> Move entire model to GPU with .to(device)</List.Item>
            <List.Item><strong>Mode Switching:</strong> Toggle between training and evaluation modes</List.Item>
            <List.Item><strong>Gradient Management:</strong> Automatic gradient computation and storage</List.Item>
            <List.Item><strong>State Dict:</strong> Save and load model parameters</List.Item>
          </List>
          </div>
          <div data-slide>
          <CodeBlock language="python" code={`class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

model = MyModel()
model.to('cuda')  # Move to GPU
model.train()     # Training mode
model.eval()      # Evaluation mode`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">nn.Module Methods</Title>
          <Text>
            Essential methods provided by nn.Module:
          </Text>
          <CodeBlock language="python" code={`# Parameter access
model.parameters()           # Iterator over all parameters
model.named_parameters()     # Iterator with parameter names
model.state_dict()          # Dictionary of all parameters

# Model manipulation
model.to(device)            # Move to device (CPU/GPU)
model.half()                # Convert to half precision
model.double()              # Convert to double precision
model.requires_grad_(False) # Freeze all parameters

# Mode control
model.train()               # Enable dropout, batch norm updates
model.eval()                # Disable dropout, freeze batch norm`}/>
          </div>
          
          
          <WeightInitialization/>
          <div data-slide>
          <Title order={3} mt="md">nn.Module Save and Load</Title>
          <Text>
            Saving and loading model state for checkpointing and deployment:
          </Text>
          <CodeBlock language="python" code={`# Save model state
torch.save(model.state_dict(), 'model_weights.pth')

# Save complete checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_accuracy': best_acc
}
torch.save(checkpoint, 'checkpoint.pth')

# Load model state
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Load complete checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']`}/>
          </div>
          
          
          <Optimization/>
          
          
          <CustomLoss/>
        

        <div data-slide>
          <Title order={2}>Data Pipeline Components</Title>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Dataset</Title>
          <Text>
            Container that defines how to access your data. Implement <Code>__len__</Code> and <Code>__getitem__</Code> methods.
          </Text>
          <CodeBlock language="python" code={`class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])`}/>
          
          <Text mt="sm">
            The <Code>__len__</Code> method returns the total number of samples:
          </Text>
          <CodeBlock language="python" code={`    def __len__(self):
        return len(self.data)`}/>
          
          <Text mt="sm">
            The <Code>__getitem__</Code> method retrieves a single sample and applies transformations:
          </Text>
          <CodeBlock language="python" code={`    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">DataLoader</Title>
          <Text>
            Handles batching, shuffling, and parallel loading efficiently. Key parameters control data loading behavior:
          </Text>
          <CodeBlock language="python" code={`# Basic DataLoader configuration
dataloader = DataLoader(
    dataset,
    batch_size=32,      # Samples per batch
    shuffle=True,       # Randomize order each epoch
    num_workers=4       # Parallel data loading processes
)`}/>
          
          <Title order={3} mt="md">Batch Size</Title>
          <Text>
            Number of samples processed together. Balance between memory usage and training speed:
          </Text>
          <List>
            <List.Item><strong>Large batch size:</strong> Better GPU utilization, more stable gradients, faster per epoch</List.Item>
            <List.Item><strong>Small batch size:</strong> Less memory, more gradient noise (can help generalization), more updates per epoch</List.Item>
          </List>
          </div>
          
        <div data-slide>
          <Title order={2}>Data Splits</Title>
          
          <Title order={3} mt="md">Training Set</Title>
          <Text>
            Model learns patterns from this data. Typically 60-80% of total data.
          </Text>
          
          <Title order={3} mt="md">Validation Set</Title>
          <Text>
            Tune hyperparameters, monitor overfitting, make early stopping decisions. Typically 10-20% of data.
          </Text>
          
          <Title order={3} mt="md">Test Set</Title>
          <Text>
            Final performance evaluation. Never touched during training. Typically 10-20% of data.
          </Text>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Random Split Implementation</Title>
          <CodeBlock language="python" code={`from torch.utils.data import random_split

# Define split sizes
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Create random splits
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)`}/></div>
          <div data-slide>
          <Title order={3} mt="md">Creating DataLoaders for Each Split</Title>
          <Text>
            Different configurations for training and evaluation:
          </Text>
          <CodeBlock language="python" code={`# Training loader - shuffle for better learning
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,       # Randomize each epoch
    num_workers=4,
    pin_memory=True     # Faster GPU transfer
)

# Validation loader - no shuffle needed
val_loader = DataLoader(
    val_dataset,
    batch_size=64,      # Can use larger batch (no gradients)
    shuffle=False,      # Keep order consistent
    num_workers=4,
    pin_memory=True
)

# Test loader - similar to validation
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4
)`}/>
        </div>

        <div data-slide>
          <Title order={2}>Training Concepts</Title>
          
          <Title order={3} mt="md">Epoch</Title>
          <Text>
            One complete pass through the entire training dataset.
          </Text>
          
          <Title order={3} mt="md">Training Loop</Title>
          <Text>
            Core training cycle: Forward pass → Calculate loss → Backward pass → Update weights
          </Text>
          <CodeBlock language="python" code={`for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Evaluation Mode</Title>
          <Text>
            Disable dropout and batch normalization updates during validation.
          </Text>
          <CodeBlock language="python" code={`model.eval()
with torch.no_grad():
    # validation code`}/>
        </div>

        <div data-slide>
          <Title order={2}>Callbacks</Title>
          
          <Text>
            Callbacks are functions executed at specific points during training to monitor progress, save checkpoints, or modify training behavior. They help automate common training patterns and improve model performance.
          </Text>
          
          <Title order={3} mt="md">Common Callback Patterns</Title>
          <Text>
            Implementing a simple callback system for training monitoring:
          </Text>
          <CodeBlock language="python" code={`class TrainingCallback:
    def on_epoch_start(self, epoch, model):
        pass
    
    def on_batch_end(self, batch_idx, loss):
        pass
    
    def on_epoch_end(self, epoch, train_loss, val_loss):
        pass`}/>
          </div>
          <div data-slide>
          <Text mt="sm">
            Using callbacks in your training loop:
          </Text>
          <CodeBlock language="python" code={`callbacks = [checkpoint_callback, logging_callback]

for epoch in range(num_epochs):
    for callback in callbacks:
        callback.on_epoch_start(epoch, model)
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_step(data, target)
        for callback in callbacks:
            callback.on_batch_end(batch_idx, loss)`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Model Checkpointing</Title>
          <Text>
            Save model weights periodically and keep best performing versions:
          </Text>
          <CodeBlock language="python" code={`class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss'):
        self.filepath = filepath
        self.best_score = float('inf')
        
    def on_epoch_end(self, epoch, model, val_loss):
        if val_loss < self.best_score:
            self.best_score = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'best_score': self.best_score
            }, self.filepath)
            print(f"Saved best model at epoch {epoch}")`}/>
          </div>
      </Stack>
      
    </Container>
  );
};

export default DataPipelineTrainingLoop;