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
          <Title order={2}>Pipeline Components</Title>
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
            <List.Item><strong>Large batch size:</strong> More stable gradients, faster per epoch</List.Item>
            <List.Item><strong>Small batch size:</strong> Less memory, more gradient noise (can help generalization), more updates per epoch</List.Item>
          </List>
          </div>
          

         <div data-slide>
          <Title order={2}>nn Components</Title>
          
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
          <Title order={3} mt="md">nn.Module Methods</Title>
          <Text>
            Essential methods provided by nn.Module:
          </Text>
          <CodeBlock language="python" code={`# Model manipulation
model.to(device)            # Move to device (CPU/GPU)
model.half()                # Convert to half precision
model.double()              # Convert to double precision
model.requires_grad_(False) # Freeze all parameters

# Mode control
model.train()               # Enable dropout, batch norm updates
model.eval()                # Disable dropout, freeze batch norm`}/>
</div>
<div data-slide>
<CodeBlock language="python" code={`
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)   # Named 'fc1'
        self.fc2 = nn.Linear(5, 2)    # Named 'fc2'

model = SimpleModel()

# Parameter access methods
model.parameters()           # Iterator of tensors only (no names)
model.named_parameters()     # Iterator of (name, tensor) pairs  
model.state_dict()          # OrderedDict with all parameters & buffers

# Example outputs:
list(model.parameters())    # [tensor([[...]]), tensor([...]), ...]

dict(model.named_parameters())  # {'fc1.weight': tensor([5, 10]),  <- layer_name.parameter_type
                                # 'fc1.bias': tensor([5]),
                                # 'fc2.weight': tensor([2, 5]),
                                # 'fc2.bias': tensor([2])}

model.state_dict().keys()   # odict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
model.state_dict()['fc1.weight']  # Access specific tensor by name`}/>
          </div>
          
          
          <WeightInitialization/>
         
          
          <Optimization/>
          
          
          <CustomLoss/>
         <div data-slide>
          <Title order={3} mt="md">nn.Module Save and Load</Title>
          
          <Text>
PyTorch uses <Code>.pth</Code> (or <Code>.pt</Code>) files to save tensors, models, and other Python objects.  </Text>
          </div>
          
          <div data-slide>
          <Title order={4} mt="md">Saving and Loading Weights</Title>
          <Text>
            The standard PyTorch workflow for model persistence:
          </Text>
          <CodeBlock language="python" code={`# Save only the state dictionary (weights and biases)
torch.save(model.state_dict(), 'model_weights.pth')

# Load the weights
model = MyModel()  # Need to instantiate the model first
model.load_state_dict(torch.load('model_weights.pth'))
# you can train it or eval`}/>
          </div>
          
          <div data-slide>
          <Title order={4} mt="md">Training Checkpoints</Title>
          <Text>
            Save complete training state to resume training later:
          </Text>
          <CodeBlock language="python" code={`# Create comprehensive checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_accuracy': best_acc,
    ...
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')`}/>
          </div>
          
          <div data-slide>
          <Title order={4} mt="md">Loading Checkpoints to Resume Training</Title>
          <Text>
            Restore complete training state to continue from where you left off:
          </Text>
          <CodeBlock language="python" code={`# Load checkpoint
checkpoint = torch.load('checkpoint.pth')

# Restore model and optimizer states
model = MyModel()
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
`}/>
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
          <div data-slide>
          <Title order={3} mt="md">Early Stopping</Title>
          <Text>
            Stop training when the model stops improving to prevent overfitting:
          </Text>
          <CodeBlock language="python" code={`class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def on_epoch_end(self, epoch, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered at epoch {epoch}")
        else:
            self.best_loss = val_loss
            self.counter = 0
            
    def should_stop_training(self):
        return self.should_stop`}/>
          </div>
          <div data-slide>
          <Title order={3} mt="md">Logging Callback</Title>
          <Text>
            Track and log training metrics for analysis and debugging:
          </Text>
          <CodeBlock language="python" code={`class LoggingCallback:
    def __init__(self, log_file='training.log', log_interval=10):
        self.log_file = log_file
        self.log_interval = log_interval
        self.batch_losses = []
        self.epoch_metrics = []
        
    def on_batch_end(self, batch_idx, loss):
        self.batch_losses.append(loss)
        if batch_idx % self.log_interval == 0:
            avg_loss = sum(self.batch_losses[-self.log_interval:]) / len(self.batch_losses[-self.log_interval:])
            with open(self.log_file, 'a') as f:
                f.write(f"Batch {batch_idx}: Loss = {avg_loss:.4f}\\n")
                
    def on_epoch_end(self, epoch, train_loss, val_loss, metrics=None):
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics or {}
        }
        self.epoch_metrics.append(log_entry)
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\\n")
            if metrics:
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.4f}\\n")`}/>
          </div>
      </Stack>
      
    </Container>
  );
};

export default DataPipelineTrainingLoop;