import React from 'react';
import { Text, Stack, List, Grid, Table, Code } from '@mantine/core';
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

  const modernArchitectureCode = `
class ModernCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x`;

  return (
    <Stack spacing="md">
      <Text>
        CNN architectures have evolved significantly since LeNet-5, with each new design
        introducing innovative concepts to address specific challenges in computer vision.
      </Text>

      <Text weight={700}>1. Basic CNN Architecture</Text>

      <Text>
        A fundamental CNN architecture typically consists of:
      </Text>

      <CodeBlock
        language="python"
        code={basicCNNCode}
      />

      <Text weight={700}>2. Evolution of CNN Architectures</Text>

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

      <Text weight={700}>3. Residual Blocks</Text>

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

      <Text weight={700}>4. Modern Architecture Principles</Text>

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
        <List.Item>
          <strong>Computational Efficiency:</strong>
          <List withPadding>
            <List.Item>Bottleneck designs to reduce parameters</List.Item>
            <List.Item>Depthwise separable convolutions</List.Item>
            <List.Item>Channel attention mechanisms</List.Item>
          </List>
        </List.Item>
      </List>

      <Text>Here's an example of a modern CNN implementation incorporating these principles:</Text>

      <CodeBlock
        language="python"
        code={modernArchitectureCode}
      />

      <Text weight={700}>5. Architecture Selection Guidelines</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text weight={600}>Choose Simple Architectures When:</Text>
          <List>
            <List.Item>Dataset is small ({`< 50k images`})</List.Item>
            <List.Item>Computing resources are limited</List.Item>
            <List.Item>Real-time inference is required</List.Item>
            <List.Item>Problem is relatively simple</List.Item>
          </List>
        </Grid.Col>

        <Grid.Col span={12} md={6}>
          <Text weight={600}>Choose Complex Architectures When:</Text>
          <List>
            <List.Item>Large dataset available ({"> 1M images"})</List.Item>
            <List.Item>High accuracy is critical</List.Item>
            <List.Item>Complex feature hierarchies needed</List.Item>
            <List.Item>Transfer learning from similar domain</List.Item>
          </List>
        </Grid.Col>
      </Grid>

      <Text weight={700}>6. Performance Considerations</Text>

      <Text>
        When implementing CNN architectures, consider:
      </Text>

      <List>
        <List.Item>
          <strong>Memory Usage:</strong>
          <BlockMath>
            {`Memory \\approx \\sum_{l} (F_l \\times H_l \\times W_l \\times B)`}
          </BlockMath>
          Where <InlineMath>F_l</InlineMath> is features, <InlineMath>H_l</InlineMath> and <InlineMath>W_l</InlineMath> are 
          spatial dimensions, and <InlineMath>B</InlineMath> is batch size.
        </List.Item>
        <List.Item>
          <strong>Computational Cost:</strong>
          <BlockMath>
            {`FLOPs \\approx \\sum_{l} (K_l^2 \\times C_{in} \\times C_{out} \\times H_l \\times W_l)`}
          </BlockMath>
          Where <InlineMath>K_l</InlineMath> is kernel size, <InlineMath>{`C_{in}`}</InlineMath> and <InlineMath>{`C_{out}`}</InlineMath> are input/output channels.
        </List.Item>
      </List>
    </Stack>
  );
};

export default Architectures;