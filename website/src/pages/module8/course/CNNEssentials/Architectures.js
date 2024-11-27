import React from 'react';
import { Text, Stack, List, Grid, Table, Code, Title } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const Architectures = () => {
  const basicCNNCode = `
import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)`;

  const resNetBlockCode = `
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with optional projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out`;

  return (
    <Stack spacing="md">
      <Text>
        CNN architectures have evolved significantly since LeNet-5, with each new design
        introducing innovative concepts to address specific challenges in computer vision.
      </Text>

      <Title order={5} className="mb-2">Basic CNN Architecture:</Title>
      <Text>
        A fundamental CNN architecture typically consists of:
      </Text>

      <CodeBlock
        language="python"
        code={basicCNNCode}
      />

      <Title order={5} className="mb-2">Evolution of CNN Architectures:</Title>
      <Table>
        <thead>
          <tr>
            <th>Architecture</th>
            <th>Key Innovations</th>
            <th>Impact</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>LeNet-5 (1998)</td>
            <td>First successful CNN application</td>
            <td>Proved CNNs viable for digit recognition</td>
          </tr>
          <tr>
            <td>AlexNet (2012)</td>
            <td>ReLU, Dropout, Local Response Normalization</td>
            <td>Started deep learning revolution</td>
          </tr>
          <tr>
            <td>VGGNet (2014)</td>
            <td>Small filters (3×3), deep architectures</td>
            <td>Showed importance of depth</td>
          </tr>
          <tr>
            <td>ResNet (2015)</td>
            <td>Skip connections, batch normalization</td>
            <td>Enabled training of very deep networks</td>
          </tr>
          <tr>
            <td>DenseNet (2017)</td>
            <td>Dense connections between layers</td>
            <td>Improved feature reuse</td>
          </tr>
          <tr>
            <td>EfficientNet (2019)</td>
            <td>Compound scaling method</td>
            <td>Better efficiency-accuracy trade-off</td>
          </tr>
        </tbody>
      </Table>

      <Title order={5} className="mb-2">Residual Blocks:</Title>
      <Text>
        Residual connections are a key innovation that enabled very deep architectures:
      </Text>

      <BlockMath>
        {`F(x) + x`}
      </BlockMath>

      <CodeBlock
        language="python"
        code={resNetBlockCode}
      />

      <Title order={5} className="mb-2">Modern Architecture Principles:</Title>
      <List>
        <List.Item>
          <strong>Network Design:</strong>
          <List withPadding>
            <List.Item>Use small filters (3×3) repeatedly</List.Item>
            <List.Item>Increase channels while decreasing spatial dimensions</List.Item>
            <List.Item>Employ skip connections for gradient flow</List.Item>
            <List.Item>Use batch normalization for stable training</List.Item>
          </List>
        </List.Item>
      </List>
    </Stack>
  );
};

export default Architectures;