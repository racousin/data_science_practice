import React from 'react';
import { Container, Title, Text, Paper, Stack, Grid, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const PyTorchOptimizers = () => {
  return (
    <Container fluid>
      <Stack spacing="xl">
        
        <div data-slide>
          <Title order={1} mb="lg">PyTorch Optimizers: Automating Gradient Descent</Title>
          
          <Text size="lg">
            PyTorch optimizers automate the parameter update process, eliminating the need for manual gradient calculations and updates.
          </Text>
        </div>

        <div data-slide>
          <Title order={2} mt="xl">1. Manual Gradient Descent vs. Optimizers</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper p="md" className="bg-red-50">
                <Title order={4} mb="sm">Manual Gradient Descent</Title>
                <CodeBlock language="python" code={`# Manual gradient descent
x = torch.tensor([2.0], requires_grad=True)
learning_rate = 0.1

for step in range(10):
    y = x ** 2  # Function to minimize
    y.backward()  # Compute gradients
    
    # Manual update
    with torch.no_grad():
        x -= learning_rate * x.grad
        x.grad.zero_()  # Clear gradients`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper p="md" className="bg-green-50">
                <Title order={4} mb="sm">Using PyTorch Optimizer</Title>
                <CodeBlock language="python" code={`# Using PyTorch optimizer
x = torch.tensor([2.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(10):
    y = x ** 2  # Function to minimize
    
    optimizer.zero_grad()  # Clear gradients
    y.backward()          # Compute gradients
    optimizer.step()      # Update parameters`} />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        <div data-slide>
          <Title order={2} mt="xl">2. Basic Optimizer Workflow</Title>
          
          <Text mb="md">
            The standard optimization loop follows a simple pattern:
          </Text>
          
          <CodeBlock language="python" code={`import torch
import torch.optim as optim

# Step 1: Create parameters to optimize
x_params = torch.tensor([2.0], requires_grad=True)

# Step 2: Create optimizer
sgd_optimizer = torch.optim.SGD([x_params], lr=0.1)

# Step 3: Optimization loop
for iteration in range(20):
    # Define the function to minimize
    y = x_params ** 2
    
    # Clear previous gradients
    sgd_optimizer.zero_grad()
    
    # Compute gradients (dy/dx = 2x)
    y.backward()
    
    # Update parameters using gradients
    sgd_optimizer.step()
    
    print(f"Iteration {iteration}: x = {x_params.item():.4f}, y = {y.item():.4f}")`} />
          
          <Paper p="md" mt="md" className="bg-blue-50">
            <Text fw="bold">Key Steps:</Text>
            <List>
              <List.Item><strong>zero_grad():</strong> Clear gradients from previous iteration</List.Item>
              <List.Item><strong>backward():</strong> Compute gradients via backpropagation</List.Item>
              <List.Item><strong>step():</strong> Update parameters using computed gradients</List.Item>
            </List>
          </Paper>
        </div>

        <div data-slide>
          <Title order={2} mt="xl">3. Common Optimizers</Title>
          
          <Text mb="md">
            PyTorch provides various optimizers with different update strategies:
          </Text>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper p="md">
                <Title order={4} mb="sm">SGD (Stochastic Gradient Descent)</Title>
                <BlockMath>{`\\theta_{t+1} = \\theta_t - \\alpha \\nabla L(\\theta_t)`}</BlockMath>
                <CodeBlock language="python" code={`optimizer = torch.optim.SGD(
    params, 
    lr=0.01,
    momentum=0.9  # Optional momentum
)`} />
                <Text size="sm" mt="xs">
                  Simple and reliable, often used with momentum for better convergence
                </Text>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper p="md">
                <Title order={4} mb="sm">Adam (Adaptive Moment Estimation)</Title>
                <BlockMath>{`\\theta_{t+1} = \\theta_t - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}`}</BlockMath>
                <CodeBlock language="python" code={`optimizer = torch.optim.Adam(
    params,
    lr=0.001,
    betas=(0.9, 0.999)
)`} />
                <Text size="sm" mt="xs">
                  Adaptive learning rates, works well for most problems
                </Text>
              </Paper>
            </Grid.Col>
          </Grid>
          
        </div>

        <div data-slide>
          <Title order={2} mt="xl">4. Learning Rate Scheduling</Title>
          
          <Text mb="md">
            Adjusting the learning rate during training can improve convergence:
          </Text>
          
          <CodeBlock language="python" code={`# Create optimizer
optimizer = torch.optim.SGD(params, lr=0.1)

# Step decay scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=30,  # Decay every 30 epochs
    gamma=0.1      # Multiply lr by 0.1
)

# Exponential decay scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95     # Multiply lr by 0.95 each epoch
)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100      # Number of iterations
)

# Usage in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # Update learning rate`} />
        </div>

        <div data-slide>
          <Title order={2} mt="xl">5. Optimizer State and Parameter Groups</Title>
          
          <Text mb="md">
            Optimizers maintain internal state and can handle different parameter groups:
          </Text>
          
          <CodeBlock language="python" code={`# Different learning rates for different parameters
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Group parameters with different learning rates
optimizer = torch.optim.SGD([
    {'params': model[0].parameters(), 'lr': 0.01},  # First layer
    {'params': model[2].parameters(), 'lr': 0.001}  # Output layer
], momentum=0.9)

# Access optimizer state
print(optimizer.state_dict())  # View current state

# Save and load optimizer state
torch.save(optimizer.state_dict(), 'optimizer.pth')
optimizer.load_state_dict(torch.load('optimizer.pth'))`} />
        </div>

      </Stack>
    </Container>
  );
};

export default PyTorchOptimizers;