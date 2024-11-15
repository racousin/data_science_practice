import React from 'react';
import { Title, Text, Stack, Alert } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

const Regularization = () => {
  return (
    <Stack spacing="xl" w="100%">
      <Title order={2} id="regularization-overview" mb="md">
        Regularization Techniques in Deep Learning
      </Title>
      
      <Text size="lg" mb="xl">
        Regularization techniques help prevent overfitting by adding constraints or modifications to the learning process. Here are the main techniques used in modern deep learning:
      </Text>

      {/* L2 Regularization Section */}
      <Stack spacing="md">
        <Title order={3} id="l2-regularization">L2 Regularization (Weight Decay)</Title>
        
        <Text>
          L2 regularization adds a penalty term to the loss function proportional to the square of weights.
          This encourages the model to use smaller weights and distribute the importance across features.
        </Text>
        
        <Alert variant="light" title="Mathematical Definition">
          <BlockMath>
            {`L_{total} = L_{original} + \\lambda \\sum_{w} w^2`}
          </BlockMath>
          where <InlineMath>{`\\lambda`}</InlineMath> is the regularization strength hyperparameter.
        </Alert>

        <Title order={4} mt="sm">Implementation</Title>
        <CodeBlock 
          language="python" 
          code={`
# Define model with L2 regularization
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # weight_decay is L2 regularization

# Alternative manual implementation
def train_step(model, inputs, targets, optimizer, l2_lambda=0.01):
    # Forward pass
    outputs = model(inputs)
    criterion = nn.MSELoss()
    
    # Calculate primary loss
    loss = criterion(outputs, targets)
    
    # Add L2 regularization term
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    loss = loss + l2_lambda * l2_norm
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()`}
        />
      </Stack>

      {/* Dropout Section */}
      <Stack spacing="md" mt="xl">
        <Title order={3} id="dropout">Dropout</Title>
        
        <Text>
          Dropout is a powerful regularization technique that randomly deactivates neurons during training,
          forcing the network to learn redundant representations and preventing co-adaptation of features.
        </Text>
        
        <Alert variant="light" title="Key Properties">
          • Training: Each neuron has a probability p of being dropped (set to 0)
          • Inference: All neurons are active, but outputs are scaled by (1-p)
          • Common dropout rates: 0.2 to 0.5 (20% to 50% of neurons dropped)
        </Alert>

        <Title order={4} mt="sm">Implementation</Title>
        <CodeBlock 
          language="python"
          code={`
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Add dropout after activation
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Usage
model = NetworkWithDropout()
model.train()  # Enable dropout during training
# ... training loop ...
model.eval()   # Disable dropout during evaluation`}
        />
      </Stack>

      {/* Early Stopping Section */}
      <Stack spacing="md" mt="xl">
        <Title order={3} id="early-stopping">Early Stopping</Title>
        
        <Text>
          Early stopping prevents overfitting by monitoring the validation performance and stopping
          training when the model begins to overfit, saving the best model weights.
        </Text>
        
        <Alert variant="light" title="Parameters">
          • patience: Number of epochs to wait for improvement
          • min_delta: Minimum change in validation loss to qualify as an improvement
          • restore_best_weights: Whether to restore model to best weights after stopping
        </Alert>

        <Title order={4} mt="sm">Implementation</Title>
        <CodeBlock 
          language="python"
          code={`
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            
    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# Usage in training loop
early_stopping = EarlyStopping(patience=5)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train(model, train_loader)
    val_loss = validate(model, val_loader)
    
    early_stopping(model, val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        early_stopping.restore_weights(model)  # Restore best weights
        break`}
        />
      </Stack>
    </Stack>
  );
};

export default Regularization;