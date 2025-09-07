import React from 'react';
import { Container, Title, Text, Stack, Alert, List, Paper } from '@mantine/core';
import CodeBlock from '../../../../components/CodeBlock';

const FineTuning = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <div data-slide>
        <Title order={1}>Fine-Tuning Pre-trained Models</Title>
        
        <Text>
          Fine-tuning is a powerful technique where we leverage pre-trained models and adapt them to 
          new tasks. Instead of training from scratch, we start with weights learned on large datasets 
          and modify them for our specific use case.
        </Text>

        <Alert color="blue" mb="md">
          <Text>
            <strong>Key Concept:</strong> Pre-trained models have already learned useful features. 
            We can reuse these features and adapt only the necessary parts for our task.
          </Text>
        </Alert>
</div>
        <div data-slide>
        <Title order={2} mt="xl">Why Fine-Tuning?</Title>
        
        <List spacing="sm" mt="md">
          <List.Item><strong>Faster Training:</strong> Start from good initial weights instead of random initialization</List.Item>
          <List.Item><strong>Less Data Required:</strong> Leverage knowledge from large pre-training datasets</List.Item>
          <List.Item><strong>Better Performance:</strong> Often achieves better results than training from scratch</List.Item>
          <List.Item><strong>Resource Efficient:</strong> Requires less computational resources and time</List.Item>
        </List>
</div>
        <div data-slide>
        <Title order={2} mt="xl">Fine-Tuning Process</Title>

        <Title order={3} mt="lg">Step 1: Load a Pre-trained Architecture</Title>
        
        <Text>
          Start by importing a pre-trained model. PyTorch provides many models through torchvision:
        </Text>

        <CodeBlock
          language="python"
          code={`
import torch
import torchvision.models as models

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")`}/>
</div>
<div data-slide>
        <Title order={3} mt="lg">Step 2: Inspect Model Architecture</Title>
        
        <Text>
          Understanding the model structure helps identify which layers to modify:
        </Text>

        <CodeBlock
          language="python"
          code={`
# Check the final classification layer
print(model.fc)  # Linear(in_features=512, out_features=1000)

# The model was trained for 1000 ImageNet classes
# We need to adapt it for our task (e.g., 10 classes)`}/>
</div>
<div data-slide>
        <Title order={3} mt="lg">Step 3: Modify the Output Layer</Title>
        
        <Text>
          Replace the final layer to match your number of classes:
        </Text>

        <CodeBlock
          language="python"
          code={`
# Replace the final layer for 10-class classification
num_classes = 10
model.fc = torch.nn.Linear(512, num_classes)

# The new layer has random weights while others are pre-trained`}/>
</div>
<div data-slide>
        <Title order={3} mt="lg">Step 4: Freeze Early Layers</Title>
        
        <Text>
          Freezing layers prevents their weights from updating during training. This is useful when 
          you want to keep the learned features from pre-training:
        </Text>

        <CodeBlock
          language="python"
          code={`
# Freeze all layers except the final one
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the final layer
for param in model.fc.parameters():
    param.requires_grad = True`}/>
</div>
<div data-slide>
        <Title order={2} mt="xl">Advanced Fine-Tuning Strategies</Title>

        <Title order={3} mt="lg">Gradual Unfreezing</Title>
        
        <Text>
          Start by training only the new layers, then gradually unfreeze earlier layers:
        </Text>

        <CodeBlock
          language="python"
          code={`
# Stage 1: Train only the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# ... train for a few epochs ...

# Stage 2: Unfreeze last few layers
for param in model.layer4.parameters():
    param.requires_grad = True
    
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=0.0001  # Lower learning rate for pre-trained layers
)`}/>
</div>
<div data-slide>
        <Title order={3} mt="lg">Different Learning Rates</Title>
        
        <Text>
          Use different learning rates for pre-trained and new layers:
        </Text>

        <CodeBlock
          language="python"
          code={`
# Create parameter groups with different learning rates
optimizer = torch.optim.SGD([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-2}
], momentum=0.9)`}/>
</div>
<div data-slide>
        <Title order={2} mt="xl">Complete Fine-Tuning Example</Title>

        <Paper p="md" withBorder>
          <Text>
            Here's a complete example fine-tuning ResNet for a custom classification task:
          </Text>
        </Paper>

        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn
import torchvision.models as models

# 1. Load pre-trained model
model = models.resnet18(pretrained=True)

# 2. Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# 3. Replace classifier for 5 classes
num_classes = 5
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

# 4. Setup optimizer (only new layers will update)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 5. Training loop (simplified)
model.train()
for epoch in range(10):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()`}/>
</div>

      </Stack>
    </Container>
  );
};

export default FineTuning;