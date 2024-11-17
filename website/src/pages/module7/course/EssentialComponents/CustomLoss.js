import React from 'react';
import { Title, Text, Stack, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const CustomLoss = () => {
  return (
    <Stack spacing="md">
      <Text>
        Custom loss functions allow you to define specific optimization objectives beyond standard losses like Cross-Entropy or MSE. 
        They're particularly useful when dealing with:
      </Text>

      <ul className="list-disc pl-6 space-y-2">
        <li>Class imbalance problems</li>
        <li>Multi-task learning scenarios</li>
        <li>Domain-specific optimization objectives</li>
        <li>Complex ranking or recommendation systems</li>
      </ul>

      <Title order={3} mt="md">Implementing Custom Loss Functions</Title>
      <Text>
        In PyTorch, custom loss functions can be implemented by subclassing <Code>nn.Module</Code>. 
        The key is to implement the <Code>forward</Code> method that defines how the loss is computed.
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions, targets):
        return self.weight * torch.mean((predictions - targets) ** 2)

# Usage example
criterion = CustomMSELoss(weight=2.0)
loss = criterion(model_outputs, targets)`}
      />

      <Title order={3} mt="md">Example: Focal Loss</Title>
      <Text>
        Focal Loss is a modified cross-entropy loss that addresses class imbalance by down-weighting well-classified examples.
        The mathematical formula for Focal Loss is:
      </Text>

      <BlockMath>
        {`FL(p_t) = -\\alpha_t(1-p_t)^\\gamma \\log(p_t)`}
      </BlockMath>

      <Text>
        where <InlineMath>{`p_t`}</InlineMath> is the model's estimated probability for the target class,
        <InlineMath>{`\\alpha_t`}</InlineMath> is a balancing factor, and <InlineMath>{`\\gamma`}</InlineMath> is the focusing parameter.
      </Text>

      <CodeBlock
        language="python"
        code={`
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Example usage
model = YourModel()
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.Adam(model.parameters())

for inputs, targets in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()`}
      />

      <Title order={3} mt="md">Example: Multi-Task Loss</Title>
      <Text>
        When training models for multiple tasks simultaneously, we often need to combine multiple loss functions:
      </Text>

      <CodeBlock
        language="python"
        code={`
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights={'classification': 1.0, 'regression': 0.5}):
        super().__init__()
        self.task_weights = task_weights
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        class_loss = self.classification_loss(outputs['class'], targets['class'])
        reg_loss = self.regression_loss(outputs['reg'], targets['reg'])
        
        total_loss = (self.task_weights['classification'] * class_loss + 
                     self.task_weights['regression'] * reg_loss)
        return total_loss

# Usage example
criterion = MultiTaskLoss()
loss = criterion({
    'class': class_predictions,
    'reg': reg_predictions
}, {
    'class': class_targets,
    'reg': reg_targets
})`}
      />
    </Stack>
  );
};

export default CustomLoss;