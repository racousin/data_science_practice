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
                  <Text mb="md">
            Let's consider some X and y data. Let's use synthetic data to illustrate:
            </Text>
                      <CodeBlock language="python" code={`import torch

# Set seed for reproducibility
torch.manual_seed(42)

# Total dataset parameters
n_total = 1000
n_features = 20

# Generate complete dataset
X = torch.randn(n_total, n_features)
y = torch.rand(n_total, 1)`} />
          Our journey: 1. Define a <strong>model</strong> size and architecture that respects X and y shapes. 2. Choose a <strong>loss</strong> function to measure the error between predictions and true values. 3. Select an <strong>optimizer</strong>. 4. <strong>Train</strong> the model through iterative optimization. 5. <strong>Evaluate</strong> model performance.
          </div>
        <div data-slide>
          <Title order={2} mb="md">1. Building a Neural Network Model</Title>
          <Text mb="md">
            In the forward pass, you must respect the size of your input (e.g., first layer with input size matching your features) 
            and output (returning a tensor with size matching your target y).
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
          
        </div>

        <div data-slide>
          <Title order={2} mb="md">2. Loss Function</Title>
          <Text mb="md">
            The loss function measures how wrong our predictions are. Mean Squared Error (MSE) is commonly 
            used for regression:
          </Text>
          
          <CodeBlock language="python" code={`# Define loss function
criterion = nn.MSELoss()

# Example: Computing loss between predictions and targets
y_pred = torch.tensor([[1.5, 2.0], [3.0, 4.5]], requires_grad=True)
y_true = torch.tensor([[1.0, 2.5], [3.5, 4.0]])

# Calculate MSE loss
loss = criterion(y_pred, y_true)
print(f"MSE Loss: {loss.item():.4f}")
# Output: MSE Loss: 0.1875`} />
        </div>
        <div data-slide>
          <Title order={2} mb="md">3. Optimizer Setup</Title>
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
          <Title order={2} mb="md">4. Creating Training, Validation, and Test Data</Title>
          <Text mb="md">
            In machine learning, we split our data into three sets, each serving a distinct purpose:
          </Text>
          
          <List mb="md">
            <List.Item><strong>Training Set (60-80%):</strong> Used to train the model and update parameters</List.Item>
            <List.Item><strong>Validation Set (10-20%):</strong> Used to tune hyperparameters and prevent overfitting</List.Item>
            <List.Item><strong>Test Set (10-20%):</strong> Used for final evaluation, never touched during training</List.Item>
          </List>



</div>
 <div data-slide>
          <CodeBlock language="python" code={`# Calculate split sizes
n_train = int(0.7 * n_total)  # 700 samples
n_val = int(0.15 * n_total)    # 150 samples
n_test = n_total - n_train - n_val  # 150 samples

# Perform the splits
X_train = X[:n_train]
y_train = y[:n_train]

X_val = X[n_train:n_train+n_val]
y_val = y[n_train:n_train+n_val]

X_test = X[n_train+n_val:]
y_test = y[n_train+n_val:]`} />

          <Text mb="md">
            <strong>Important:</strong> Never use the test set during training or hyperparameter tuning. 
            It should only be used once at the very end to get an unbiased estimate of model performance.
          </Text>

        </div>

        <div data-slide>
          <Title order={2} mb="md">5. The Training Loop</Title>
          <Text mb="md">
            A proper training loop monitors both training and validation performance to detect overfitting:
          </Text>
          
          <List mb="md">
            <List.Item><strong>Training phase:</strong> Model learns from data (gradients enabled)</List.Item>
            <List.Item><strong>Validation phase:</strong> Model evaluated without learning (no gradients)</List.Item>
          </List>
          
          <Text mb="md">
            Initialize tracking variables:
          </Text>

          <CodeBlock language="python" code={`# Training parameters
num_epochs = 100
train_losses = []
val_losses = []`} />
          <Text mb="md">
            <strong>Key Points:</strong>
            <List size="sm">
              <List.Item><strong>Epoch:</strong> One complete pass through the entire training dataset</List.Item>
              <List.Item><code>torch.no_grad()</code> disables gradient computation for efficiency</List.Item>
              <List.Item>If validation loss increases while training loss decreases, you're overfitting!</List.Item>
            </List>
          </Text>
          </div>
          <div data-slide>
          <Text mb="md">
            The complete training and validation loop:
          </Text>
          
          <CodeBlock language="python" code={`for epoch in range(num_epochs):
    # ===== TRAINING PHASE =====
    model.train()  # Set model to training mode
    
    # 1. Zero gradients
    optimizer.zero_grad()
    # 2. Forward pass on training data
    train_pred = model(X_train)
    # 3. Calculate training loss
    train_loss = criterion(train_pred, y_train)
    # 4. Backward pass (compute gradients)
    train_loss.backward()
    # 5. Update parameters
    optimizer.step()
    
    # ===== VALIDATION PHASE =====
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Disable gradient computation
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
    
    # Store losses for visualization
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss.item():.4f}')
        print(f'  Val Loss:   {val_loss.item():.4f}')`} />


        </div>

        <div data-slide>
          <Title order={2} mb="md">6. Visualizing Training and Validation History</Title>
          <Text mb="md">
            Plotting both training and validation losses helps identify overfitting and underfitting:
          </Text>
          
          <CodeBlock language="python" code={`import matplotlib.pyplot as plt

# Plot training and validation history
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()`} />
</div>
          <div data-slide>
          <Flex direction="column" align="center">
            <Image
              src="/assets/python-deep-learning/module3/training_validation_loss.png"
              alt="Training History"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
          </Flex>
        </div>

        <div data-slide>
          <Title order={2} mb="md">7. Model Evaluation (Inference)</Title>

          
          <CodeBlock language="python" code={`# Set model to evaluation mode (disables dropout, etc.)
model.eval()

# Evaluate without computing gradients
with torch.no_grad():  # Disable gradient computation for efficiency
    # Make predictions
    test_predictions = model(X_test)
    
    # Calculate test loss
    test_loss = criterion(test_predictions, y_test)
    
    print(f"Test Loss: {test_loss.item():.4f}")`} />
        </div>
          <Text mb="md">
            If test loss is significantly higher than training loss, the model may be overfitting.
          </Text>
      </Stack>
    </Container>
  );
};

export default TrainingBasics;