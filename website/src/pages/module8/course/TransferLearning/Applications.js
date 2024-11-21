import React from 'react';
import { Container, Stack, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain } from 'lucide-react';

const Applications = () => {
    return (
      <Stack spacing="md">
        <Text>
          Transfer learning finds applications across various domains:
        </Text>
  
        <CodeBlock
          language="python"
          code={`
  def train_transfer_model(model, train_loader, val_loader, 
                          criterion, optimizer, scheduler, 
                          num_epochs, device):
      """Train a transfer learning model with validation"""
      best_val_acc = 0.0
      
      for epoch in range(num_epochs):
          # Training phase
          model.train()
          for inputs, labels in train_loader:
              inputs, labels = inputs.to(device), labels.to(device)
              
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
              
          # Validation phase
          model.eval()
          val_acc = validate_model(model, val_loader, device)
          
          if val_acc > best_val_acc:
              best_val_acc = val_acc
              torch.save(model.state_dict(), 'best_model.pth')
              
          scheduler.step()`}
        />
      </Stack>
    );
  };

export default Applications