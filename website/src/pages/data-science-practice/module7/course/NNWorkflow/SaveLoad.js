import React from 'react';
import { Stack, Title, Text, Alert, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const SaveLoad = () => {
  return (
    <Stack spacing="md">

      <Text>
        Properly saving and loading models is crucial for model deployment and 
        continuing training from checkpoints. PyTorch provides multiple ways to 
        save models, each with its own use cases.
      </Text>

      <Title order={4}>1. Saving Model Checkpoints</Title>
      <CodeBlock
        language="python"
        code={`
def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Complete training example with checkpointing
def train_with_checkpoints(model, train_loader, val_loader, criterion, 
                          optimizer, device, start_epoch=0, epochs=100, 
                          checkpoint_dir='checkpoints', patience=10):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(
                checkpoint_dir, f'best_model.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f'Saved best model checkpoint to {checkpoint_path}')
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f'model_epoch_{epoch+1}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f'Saved periodic checkpoint to {checkpoint_path}')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return best_val_loss

# Example usage: Initial training
model = RegressionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Start training from scratch
best_loss = train_with_checkpoints(
    model, train_loader, val_loader, criterion, optimizer, device
)

# Example: Resume training from checkpoint
def resume_training(checkpoint_path, epochs=50):
    # Initialize model and optimizer
    model = RegressionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Load checkpoint
    start_epoch, prev_loss = load_checkpoint(model, optimizer, checkpoint_path)
    print(f'Resuming training from epoch {start_epoch+1}')
    print(f'Previous best loss: {prev_loss:.4f}')
    
    # Continue training
    best_loss = train_with_checkpoints(
        model, train_loader, val_loader, criterion, optimizer, device,
        start_epoch=start_epoch+1, epochs=epochs
    )
    
    return best_loss

# Resume training from last checkpoint
best_loss = resume_training('checkpoints/best_model.pth', epochs=150)`}
      />


      <Title order={4}>3. Saving for Inference</Title>
      <CodeBlock
        language="python"
        code={`
# Save just the model state for inference
torch.save(model.state_dict(), 'model_inference.pth')

# Load model for inference
loaded_model = RegressionNet()  # Create uninitialized model
loaded_model.load_state_dict(torch.load('model_inference.pth'))
loaded_model.eval()  # Set to evaluation mode`}
      />

      <Title order={4}>4. Export for Production</Title>
      <CodeBlock
        language="python"
        code={`
# Export to TorchScript for production
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# Load TorchScript model
loaded_scripted_model = torch.jit.load('model_scripted.pt')

# Example inference
with torch.no_grad():
    test_input = torch.randn(1, 1).to(device)
    output = loaded_scripted_model(test_input)
    print(f"Model prediction: {output.item():.4f}")`}
      />

      <Text>
        TorchScript provides a way to serialize and optimize models for production 
        environments, ensuring consistent behavior across different platforms and 
        runtime environments.
      </Text>
    </Stack>
  );
};

export default SaveLoad;