import React from 'react';
import { Text, Stack, List, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const ReduceLROnPlateau = () => {
  return (
    <Stack spacing="md">
      <Text>
        ReduceLROnPlateau is a learning rate scheduler that reduces the learning rate when a metric has stopped improving.
        This is particularly useful when training neural networks, as it allows the model to make larger steps during early training
        and smaller, more refined steps as it approaches an optimum.
      </Text>

      <Text size="sm" className="font-bold">Key Benefits:</Text>
      <List>
        <List.Item>Automatically adapts learning rate based on training progress</List.Item>
        <List.Item>Helps prevent plateaus in training by reducing learning rate when progress stalls</List.Item>
        <List.Item>Reduces the need for manual learning rate tuning</List.Item>
        <List.Item>Can improve final model performance by finding better local minima</List.Item>
      </List>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.optim as optim

# Define model and optimizer
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Create the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # Reduce LR when the quantity monitored has stopped decreasing
    factor=0.1,          # Factor by which the learning rate will be reduced
    patience=10,         # Number of epochs with no improvement after which LR will be reduced
    verbose=True,        # Print message for each update
    min_lr=1e-6         # Lower bound on the learning rate
)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # Step the scheduler based on validation loss
    scheduler.step(val_loss)
    
    # Access the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Current learning rate: {current_lr}')`}
      />
      <Text>
        The learning rate adjustment follows a multiplicative decay:
      </Text>
      <BlockMath>
        {`lr_{new} = lr_{current} \\times factor`}
      </BlockMath>
      <Text>
        For example, with an initial learning rate of 0.1 and factor=0.1, the learning rate sequence would be:
        0.1 → 0.01 → 0.001 → 0.0001
      </Text>
    </Stack>
  );
};

export default ReduceLROnPlateau;