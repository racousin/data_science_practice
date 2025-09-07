import React from 'react';
import { Container, Title, Text, Stack, Paper, List, Code, Flex, Image } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const TrainingBasics = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        <div data-slide>
          <Title order={1} mb="xl">
            Training Basics
          </Title>
        </div>
        
        <div data-slide>
          <Title order={2} mb="md">1. Building a Neural Network Model</Title>
          <Text mb="md">
            Every PyTorch model inherits from <Code>nn.Module</Code>. This provides the foundation for 
            automatic gradient computation and parameter management.
          </Text>
          
          <CodeBlock language="python" code={`class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        self.layer1 = nn.Linear(20, 128)  # Input: 20, Output: 128
        self.layer2 = nn.Linear(128, 1)   # Input: 128, Output: 1
    
    def forward(self, x):
        # Define forward pass
        x = torch.relu(self.layer1(x))
        return self.layer2(x)`} />
          
          <Text mt="md">
            The <Code>__init__</Code> method defines the layers, while <Code>forward</Code> specifies 
            how data flows through them.
          </Text>
        </div>

        <div data-slide>
          <Title order={2} mb="md">2. Optimizer Setup</Title>
          <Text mb="md">
            The optimizer updates model parameters based on computed gradients. SGD (Stochastic Gradient Descent) 
            is the simplest optimizer:
          </Text>
          
          <CodeBlock language="python" code={`# Create model instance
model = MyModel()

# Setup optimizer with model parameters and learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)`} />
          
          <Text mt="md">
            The learning rate <InlineMath math="lr = 0.01" /> controls the step size for parameter updates.
          </Text>
        </div>

        <div data-slide>
          <Title order={2} mb="md">3. Loss Function</Title>
          <Text mb="md">
            The loss function measures how wrong our predictions are. Mean Squared Error (MSE) is commonly 
            used for regression:
          </Text>
          
          <CodeBlock language="python" code={`# Define loss function
criterion = nn.MSELoss()`} />
        </div>

        <div data-slide>
          <Title order={2} mb="md">4. Creating Simple Training Data</Title>
          <Text mb="md">
            Let's create simple tensors to demonstrate training. We'll generate random data for a 
            regression problem:
          </Text>
          
          <CodeBlock language="python" code={`import torch

# Create simple training data
n_samples = 100
n_features = 20

# Random input features (100 samples, 20 features each)
X_train = torch.randn(n_samples, n_features)

# Random target values
y_train = torch.rand(n_samples, 1)

print(f"Input shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")`} />

          <Text mt="md">
            We're working directly with tensors - no datasets or data loaders needed for basic training.
          </Text>
        </div>

        <div data-slide>
          <Title order={2} mb="md">5. The Training Loop</Title>
          <Text mb="md">
            The training loop is the heart of deep learning. It repeats these steps:
          </Text>
          
          <List mb="md">
            <List.Item>Forward pass: compute predictions</List.Item>
            <List.Item>Calculate loss</List.Item>
            <List.Item>Backward pass: compute gradients</List.Item>
            <List.Item>Update parameters</List.Item>
          </List>
          
          <CodeBlock language="python" code={`# Training parameters
num_epochs = 100
training_history = []

# Training loop
for epoch in range(num_epochs):
    # 1. Zero gradients (important!)
    optimizer.zero_grad()
    
    # 2. Forward pass
    predictions = model(X_train)
    
    # 3. Calculate loss
    loss = criterion(predictions, y_train)
    
    # 4. Backward pass
    loss.backward()
    
    # 5. Update parameters
    optimizer.step()
    
    # Store loss for visualization
    training_history.append(loss.item())
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')`} />
        </div>

        <div data-slide>
          <Title order={2} mb="md">6. Visualizing Training History</Title>
          <Text mb="md">
            Tracking and visualizing the training history helps us understand if the model is learning properly:
          </Text>
          
          <CodeBlock language="python" code={`import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(training_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.show()`} />
          
          <Text mt="md" mb="md">
            A healthy training curve should show decreasing loss over time. If the loss stops decreasing, 
            the model may have converged. If it increases, the learning rate might be too high.
          </Text>
          
          <Flex direction="column" align="center">
            <Image
              src="/assets/python-deep-learning/module3/training_history.png"
              alt="Training History"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>

        <div data-slide>
          <Title order={2} mb="md">7. Model Evaluation (Inference)</Title>
          <Text mb="md">
            After training, we evaluate the model on new data without updating weights:
          </Text>
          
          <CodeBlock language="python" code={`# Create test data
X_test = torch.randn(20, n_features)
y_test = torch.rand(20, 1)

# Set model to evaluation mode (disables dropout, etc.)
model.eval()

# Evaluate without computing gradients
with torch.no_grad():  # Disable gradient computation for efficiency
    # Make predictions
    test_predictions = model(X_test)
    
    # Calculate test loss
    test_loss = criterion(test_predictions, y_test)
    
    print(f"Test Loss: {test_loss.item():.4f}")`} />
        </div>

      </Stack>
    </Container>
  );
};

export default TrainingBasics;