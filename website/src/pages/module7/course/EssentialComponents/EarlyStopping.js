import React from 'react';
import { Title, Text, Stack, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const EarlyStopping = () => {
  return (
    <Stack spacing="xl">
      
      <Text>
        Early stopping prevents overfitting by monitoring the validation performance and stopping
        training when the model begins to overfit, saving the best model weights.
      </Text>
      
      <Alert variant="light" title="Parameters">
        • patience: Number of epochs to wait for improvement
        • min_delta: Minimum change in validation loss to qualify as an improvement
        • restore_best_weights: Whether to restore model to best weights after stopping
      </Alert>

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
  );
};

export default EarlyStopping;