import React from 'react';
import { Title, Text, Stack, Group, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

const Regularization = () => {
  const L2RegExample = `
# Define model with L2 regularization
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop with L2 regularization
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    
    # Calculate primary loss
    loss = criterion(outputs, targets)
    
    # Add L2 regularization term
    l2_lambda = 0.01
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    loss = loss + l2_lambda * l2_norm
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
`;

  const dropoutExample = `
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(p=0.5)  # 50% dropout rate
        self.linear2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = self.linear2(x)
        return x

# Note: Dropout is automatically disabled during model.eval()
`;

  const earlyStoppingExample = `
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
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
            self.counter = 0

# Usage in training loop
early_stopping = EarlyStopping(patience=5)
for epoch in range(num_epochs):
    train_loss = train(model, train_loader)
    val_loss = validate(model, val_loader)
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
`;

  return (
    <Stack spacing="xl" w="100%">
      {/* Understanding Overfitting Section */}
      <div>
        <Title order={2} id="overfitting" mb="md">Understanding Overfitting</Title>
        <Text>
          Overfitting occurs when a model learns the training data too well, including noise and outliers, leading to poor generalization on unseen data. Signs of overfitting include:
        </Text>
        <Table striped highlightOnHover mt="md">
          <thead>
            <tr>
              <th>Indicator</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Training vs Validation Gap</td>
              <td>Large difference between training and validation performance</td>
            </tr>
            <tr>
              <td>Perfect Training Accuracy</td>
              <td>Model achieves near 100% accuracy on training data</td>
            </tr>
            <tr>
              <td>Poor Generalization</td>
              <td>Model performs significantly worse on new, unseen data</td>
            </tr>
          </tbody>
        </Table>
      </div>

      {/* Regularization Techniques Section */}
      <div>
        <Title order={2} id="techniques" mb="md">Regularization Techniques</Title>
        
        {/* L2 Regularization */}
        <Title order={3} mb="sm">L2 Regularization (Weight Decay)</Title>
        <Text mb="md">
          L2 regularization adds a penalty term to the loss function proportional to the square of weights:
        </Text>
        <BlockMath>
          {`L_{total} = L_{original} + \\lambda \\sum_{w} w^2`}
        </BlockMath>
        <Text mb="lg">
          where <InlineMath>{`\\lambda`}</InlineMath> is the regularization strength hyperparameter.
        </Text>

        {/* Dropout */}
        <Title order={3} mb="sm">Dropout</Title>
        <Text mb="md">
          Dropout randomly deactivates neurons during training with probability p, preventing co-adaptation of features. During inference, all neurons are active but their outputs are scaled by (1-p).
        </Text>

        {/* Early Stopping */}
        <Title order={3} mb="sm">Early Stopping</Title>
        <Text mb="md">
          Early stopping monitors validation performance and stops training when the model begins to overfit, saving the best model weights.
        </Text>
      </div>

      {/* PyTorch Implementation Section */}
      <div>
        <Title order={2} id="implementation" mb="md">PyTorch Implementation</Title>
        
        {/* L2 Regularization Implementation */}
        <Title order={3} mb="sm">L2 Regularization Example</Title>
        <CodeBlock language="python" code={L2RegExample} />

        {/* Dropout Implementation */}
        <Title order={3} mt="lg" mb="sm">Dropout Implementation</Title>
        <CodeBlock language="python" code={dropoutExample} />

        {/* Early Stopping Implementation */}
        <Title order={3} mt="lg" mb="sm">Early Stopping Implementation</Title>
        <CodeBlock language="python" code={earlyStoppingExample} />
      </div>

      {/* Summary Table */}
      <div>
        <Title order={2} mb="md">Quick Reference Guide</Title>
        <Table striped highlightOnHover>
          <thead>
            <tr>
              <th>Technique</th>
              <th>Best Used When</th>
              <th>Key Parameters</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>L2 Regularization</td>
              <td>Large weights are causing overfitting</td>
              <td>λ (regularization strength)</td>
            </tr>
            <tr>
              <td>Dropout</td>
              <td>Network is large and prone to co-adaptation</td>
              <td>p (dropout probability)</td>
            </tr>
            <tr>
              <td>Early Stopping</td>
              <td>Validation loss starts increasing</td>
              <td>patience, min_delta</td>
            </tr>
          </tbody>
        </Table>
      </div>
    </Stack>
  );
};

export default Regularization;