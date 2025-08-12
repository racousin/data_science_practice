import React from 'react';
import { Text, Stack, List, Grid, Table } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const Regularization = () => {
  const dropoutImplementationCode = `
import torch
import torch.nn as nn

class CNNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),  # Spatial dropout
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Regular dropout
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)`;

  const weightRegularizationCode = `
import torch.optim as optim

def train_with_regularization(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate main loss
            loss = criterion(outputs, targets)
            
            # Add L2 regularization explicitly if needed
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            # Add L1 regularization if needed
            l1_lambda = 0.0001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()`;

const customRegularizationCode = `
class SpectralRegularization(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.conv = conv_layer
        self.register_buffer('u', torch.randn(self.conv.weight.shape[0], 1))
        self.register_buffer('v', torch.randn(self.conv.weight.shape[1], 1))
    
    def _power_iteration(self, W, u, v, n_iterations=1):
        for _ in range(n_iterations):
            v = F.normalize(torch.mv(W.t(), u), dim=0)
            u = F.normalize(torch.mv(W, v), dim=0)
        return u, v
    
    def forward(self, x):
        W = self.conv.weight.view(self.conv.weight.shape[0], -1)
        u, v = self._power_iteration(W, self.u, self.v)
        
        # Update buffers
        self.u.copy_(u)
        self.v.copy_(v)
        
        # Calculate spectral norm
        sigma = torch.dot(u, torch.mv(W, v))
        
        # Normalize weights
        weight_normalized = self.conv.weight / sigma
        
        return F.conv2d(x, weight_normalized, self.conv.bias,
                       self.conv.stride, self.conv.padding)`;

  const augmentationCode = `
import torchvision.transforms as T

class AugmentationRegularizer:
    def __init__(self, strength=1.0):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(
                brightness=0.2 * strength,
                contrast=0.2 * strength,
                saturation=0.2 * strength,
                hue=0.1 * strength
            ),
            T.RandomAffine(
                degrees=15 * strength,
                translate=(0.1 * strength, 0.1 * strength),
                scale=(1 - 0.2 * strength, 1 + 0.2 * strength)
            ),
            T.RandomErasing(p=0.2 * strength)
        ])
    
    def __call__(self, img):
        return self.transform(img)

# Mixup augmentation implementation
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# CutMix implementation
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixels
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam`;

  return (
    <Stack spacing="md">
      <Text>
        Regularization techniques are crucial for preventing overfitting in CNNs and
        improving model generalization. This section covers various approaches and their
        implementations.
      </Text>

      <Text weight={700}>1. Dropout Techniques</Text>

      <Text>
        Dropout randomly deactivates neurons during training with probability p:
      </Text>

      <BlockMath>
        {`y = \\begin{cases} 
          \\frac{x}{1-p} & \\text{with probability } 1-p \\\\
          0 & \\text{with probability } p
        \\end{cases}`}
      </BlockMath>

      <Text>Implementation with different dropout types:</Text>

      <CodeBlock
        language="python"
        code={dropoutImplementationCode}
      />

      <Text weight={700}>2. Weight Regularization</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text>L1 Regularization:</Text>
          <BlockMath>
            {`L = L_0 + \\lambda \\sum_{w} |w|`}
          </BlockMath>
        </Grid.Col>

        <Grid.Col span={12} md={6}>
          <Text>L2 Regularization:</Text>
          <BlockMath>
            {`L = L_0 + \\lambda \\sum_{w} w^2`}
          </BlockMath>
        </Grid.Col>
      </Grid>

      <Text>Implementation with both L1 and L2 regularization:</Text>

      <CodeBlock
        language="python"
        code={weightRegularizationCode}
      />

      <Text weight={700}>3. Advanced Regularization Techniques</Text>

      <Text>
        Spectral normalization for weight regularization:
      </Text>

      <CodeBlock
        language="python"
        code={customRegularizationCode}
      />

      <Text weight={700}>4. Data Augmentation as Regularization</Text>

      <Text>
        Implementation of various augmentation techniques:
      </Text>

      <CodeBlock
        language="python"
        code={augmentationCode}
      />

      <Text weight={700}>5. Comparison of Regularization Techniques</Text>

      <Table>
        <thead>
          <tr>
            <th>Technique</th>
            <th>Pros</th>
            <th>Cons</th>
            <th>Best Use Case</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Dropout</td>
            <td>Easy to implement, effective</td>
            <td>Increases training time</td>
            <td>Dense layers, large networks</td>
          </tr>
          <tr>
            <td>L1 Regularization</td>
            <td>Promotes sparsity</td>
            <td>Can be unstable</td>
            <td>Feature selection</td>
          </tr>
          <tr>
            <td>L2 Regularization</td>
            <td>Stable, prevents large weights</td>
            <td>May not prevent overfitting completely</td>
            <td>General purpose</td>
          </tr>
          <tr>
            <td>Data Augmentation</td>
            <td>Increases effective dataset size</td>
            <td>Domain-specific, can be complex</td>
            <td>Limited data scenarios</td>
          </tr>
          <tr>
            <td>Batch Normalization</td>
            <td>Stabilizes training</td>
            <td>Adds complexity</td>
            <td>Deep networks</td>
          </tr>
        </tbody>
      </Table>

      <Text weight={700}>6. Practical Guidelines</Text>

      <List>
        <List.Item>
          <strong>Dropout Rates:</strong>
          <List withPadding>
            <List.Item>Convolutional layers: 0.1-0.2</List.Item>
            <List.Item>Dense layers: 0.4-0.5</List.Item>
            <List.Item>First and last layers: Lower or no dropout</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Weight Decay:</strong>
          <List withPadding>
            <List.Item>Start with 1e-4 for L2</List.Item>
            <List.Item>Start with 1e-5 for L1</List.Item>
            <List.Item>Adjust based on validation performance</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Data Augmentation:</strong>
          <List withPadding>
            <List.Item>Start with simple transformations</List.Item>
            <List.Item>Ensure augmentations are realistic</List.Item>
            <List.Item>Consider domain-specific augmentations</List.Item>
          </List>
        </List.Item>
      </List>

      <Text weight={700}>7. Monitoring Regularization Effects</Text>

      <Text>
        Key metrics to track:
      </Text>

      <List>
        <List.Item>Training vs. validation loss gap</List.Item>
        <List.Item>Weight distribution statistics</List.Item>
        <List.Item>Gradient magnitudes</List.Item>
        <List.Item>Model sparsity (for L1)</List.Item>
        <List.Item>Effective capacity (for dropout)</List.Item>
      </List>

      <BlockMath>
        {`\\text{Regularization Effect} = \\frac{\\text{Validation Error}}{\\text{Training Error}}`}
      </BlockMath>
    </Stack>
  );
};

export default Regularization;