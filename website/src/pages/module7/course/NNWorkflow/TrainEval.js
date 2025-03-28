import React from 'react';
import { Stack, Title, Text, Alert, Box, Code } from '@mantine/core';
import { InlineMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const TrainEval = () => {
  return (
    <Stack spacing="md">
      <Text>
        The training process involves iterating over the data in mini-batches, 
        computing gradients, and updating model parameters. Regular evaluation
        on the validation set helps monitor for overfitting.
      </Text>

      <Title order={4}>1. Training Loop Functions</Title>
      <CodeBlock
        language="python"
        code={`
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    
    with torch.no_grad():  # No gradients needed for validation
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)`}
      />

      <Title order={4}>2. Training Loop with Early Stopping</Title>
      <CodeBlock
        language="python"
        code={`
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, epochs=100, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss = validate(model, val_loader, criterion, device)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, device)`}
      />

      <Title order={4}>3. Model Evaluation</Title>
      <CodeBlock
        language="python"
        code={`
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:

            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store predictions and actuals
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    
    return {
        'test_loss': test_loss,
        'mse': mse,
        'predictions': predictions,
        'actuals': actuals
    }

# Evaluate the model
results = evaluate_model(model, test_loader, criterion, device)
print(f"Test Results:")
print(f"MSE: {results['mse']:.4f}")`}
      />
          <Text>
            <Code>model.train()</Code> vs <Code>model.eval()</Code>: Controls dropout and batch normalization behavior
          </Text>
    </Stack>
  );
};

export default TrainEval;