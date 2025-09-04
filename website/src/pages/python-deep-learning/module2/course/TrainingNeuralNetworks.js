import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const TrainingNeuralNetworks = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Training Neural Networks
          </Title>
          <Text size="xl" className="text-center mb-6">
            Optimization, Loss Functions, and Training Strategies
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" mb="md">
                Training neural networks involves finding optimal weights through iterative optimization.
                This process requires careful consideration of loss functions, optimizers, and training procedures.
              </Text>
              <List>
                <List.Item>Loss function design and selection</List.Item>
                <List.Item>Gradient-based optimization algorithms</List.Item>
                <List.Item>Training loops and validation strategies</List.Item>
                <List.Item>Regularization and generalization techniques</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Loss Functions */}
        <div data-slide className="min-h-[500px]" id="loss-functions">
          <Title order={2} mb="xl">Loss Functions</Title>
          
            <Paper mb="xl">
            <Text size="lg">
              Loss functions quantify how well our model's predictions match the true targets.
              The choice of loss function depends on the task and desired behavior.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Classification Losses</Title>
                <Code block language="python">{`import torch
import torch.nn.functional as F

# Cross-entropy loss
predictions = torch.randn(32, 10)  # Logits
targets = torch.randint(0, 10, (32,))
loss = F.cross_entropy(predictions, targets)

# Binary cross-entropy
sigmoid_output = torch.sigmoid(torch.randn(32, 1))
binary_targets = torch.randint(0, 2, (32, 1)).float()
loss = F.binary_cross_entropy(sigmoid_output, binary_targets)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Regression Losses</Title>
                <Code block language="python">{`# Mean squared error
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
mse_loss = F.mse_loss(predictions, targets)

# Mean absolute error
mae_loss = F.l1_loss(predictions, targets)

# Smooth L1 loss (Huber loss)
smooth_l1_loss = F.smooth_l1_loss(predictions, targets)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: Optimizers */}
        <div data-slide className="min-h-[500px]" id="optimizers">
          <Title order={2} mb="xl">Optimization Algorithms</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-purple-50 mb-4">
                <Title order={4} mb="sm">Common Optimizers in PyTorch</Title>
                <Code block language="python">{`import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (adaptive moment estimation)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper p="md">
                <Title order={4} mb="sm">Learning Rate Scheduling</Title>
                <Code block language="python">{`# Step learning rate decay
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10
)

# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} mb="sm">Optimizer Usage</Title>
                <Code block language="python">{`# Training step
optimizer.zero_grad()  # Clear gradients
loss = criterion(outputs, targets)
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters

# With scheduler
for epoch in range(num_epochs):
    # Training loop...
    scheduler.step()  # Update learning rate`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: Training Loop */}
        <div data-slide className="min-h-[500px]" id="training-loop">
          <Title order={2} mb="xl">Complete Training Loop</Title>
          
          <Paper className="p-4 bg-gray-50">
            <Code block language="python">{`def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}')
        print(f'  Val Loss: {val_losses[-1]:.4f}')
        print(f'  Val Accuracy: {100 * correct / total:.2f}%')
    
    return train_losses, val_losses`}</Code>
          </Paper>
        </div>

        {/* Slide 5: Regularization */}
        <div data-slide className="min-h-[500px]" id="regularization">
          <Title order={2} mb="xl">Regularization Techniques</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Dropout</Title>
                <Code block language="python">{`import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),  # 50% dropout
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # 30% dropout
            nn.Linear(hidden_size, output_size)
        )`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Weight Decay</Title>
                <Code block language="python">{`# L2 regularization via weight decay
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001, 
    weight_decay=1e-4
)

# Manual L2 regularization
def l2_regularization(model, lambda_reg):
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)**2
    return lambda_reg * l2_reg`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-blue-50 mt-4">
            <Title order={4} mb="sm">Early Stopping</Title>
            <Code block language="python">{`class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience`}</Code>
          </Paper>
        </div>

        {/* Slide 6: Best Practices */}
        <div data-slide className="min-h-[500px]" id="best-practices">
          <Title order={2} mb="xl">Training Best Practices</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">✅ Do's</Title>
                <List>
                  <List.Item>Monitor both training and validation metrics</List.Item>
                  <List.Item>Use appropriate learning rate scheduling</List.Item>
                  <List.Item>Implement early stopping to prevent overfitting</List.Item>
                  <List.Item>Save model checkpoints regularly</List.Item>
                  <List.Item>Use proper weight initialization</List.Item>
                  <List.Item>Normalize your input data</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">❌ Don'ts</Title>
                <List>
                  <List.Item>Don't use learning rates that are too high or too low</List.Item>
                  <List.Item>Don't ignore validation loss trends</List.Item>
                  <List.Item>Don't train for too many epochs without monitoring</List.Item>
                  <List.Item>Don't forget to set model.eval() during validation</List.Item>
                  <List.Item>Don't use the same data for training and testing</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default TrainingNeuralNetworks;