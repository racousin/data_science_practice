import React from 'react';
import { Text, Stack, List, Grid, Table } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const AdvancedConcepts = () => {
  const attentionMechanismCode = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels, key_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels or in_channels // 8

        # Projections for Query, Key and Value
        self.query = nn.Conv2d(in_channels, self.key_channels, 1)
        self.key = nn.Conv2d(in_channels, self.key_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Project and reshape
        q = self.query(x).view(batch_size, self.key_channels, -1)
        k = self.key(x).view(batch_size, self.key_channels, -1)
        v = self.value(x).view(batch_size, self.in_channels, -1)

        # Transpose for attention dot product
        q = q.permute(0, 2, 1)  # B X HW X C
        
        # Attention map
        attn = torch.bmm(q, k)  # B X HW X HW
        attn = F.softmax(attn, dim=2)
        
        # Attention output
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection
        return self.gamma * out + x`;

  const dilatedConvCode = `
class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 
                         padding=rate, dilation=rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in dilation_rates
        ])
        
        # 1x1 conv for residual connection if dimensions don't match
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1)
        ) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = sum(block(x) for block in self.blocks)
        return F.relu(out + residual)`;

  const dynamicConvCode = `
class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 reduction=4, num_kernels=4):
        super().__init__()
        self.num_kernels = num_kernels
        
        # Generate multiple kernels
        self.kernels = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels, 
                       kernel_size, kernel_size)
        )
        
        # Attention mechanism for kernel selection
        mid_channels = max(in_channels // reduction, 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_kernels, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Generate attention weights for kernel selection
        soft_attention = self.attention(x)
        
        # Combine kernels using attention weights
        kernel_weights = torch.sum(
            soft_attention.view(batch_size, self.num_kernels, 1, 1, 1, 1) *
            self.kernels.unsqueeze(0),
            dim=1
        )
        
        # Apply dynamic convolution
        return F.conv2d(x, kernel_weights, padding=self.kernels.size(-1)//2)`;

  const groupNormCode = `
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
        # Adaptive group size based on feature map size
        self.group_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Compute adaptive group size
        group_size = self.group_adapter(x)
        num_groups = max(1, int(self.num_groups * group_size.item()))
        
        # Apply group normalization
        N, C, H, W = x.size()
        x = x.view(N, num_groups, -1)
        
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias`;

  const deformableConvCode = `
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Regular convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Offset prediction
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                                   kernel_size, stride, padding)
        
        # Modulation (mask) prediction
        self.modulator_conv = nn.Conv2d(in_channels, kernel_size * kernel_size,
                                      kernel_size, stride, padding)
        
    def forward(self, x):
        # Get offsets and modulation
        offset = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))
        
        # Apply deformable convolution (simplified version)
        batch_size = x.size(0)
        channels = x.size(1)
        height, width = x.size(2), x.size(3)
        
        # Create sampling grid
        grid_h, grid_w = torch.meshgrid(
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device)
        )
        grid = torch.stack([grid_w, grid_h], dim=0).float()
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Add offsets to grid
        offset = offset.view(batch_size, 2, self.kernel_size**2, height, width)
        grid = grid.unsqueeze(2) + offset
        
        # Sample and apply modulation
        x = F.grid_sample(x, grid, align_corners=True)
        x = x * modulator.unsqueeze(1)
        
        return self.conv(x)`;

  return (
    <Stack spacing="md">
      <Text>
        Advanced CNN concepts introduce sophisticated mechanisms to enhance network
        capability and efficiency. These techniques represent the cutting edge of
        CNN architecture design.
      </Text>

      <Text weight={700}>1. Self-Attention Mechanisms in CNNs</Text>

      <Text>
        Self-attention allows each position to attend to all other positions:
      </Text>

      <BlockMath>
        {`Attention(Q, K, V) = softmax\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V`}
      </BlockMath>

      <CodeBlock
        language="python"
        code={attentionMechanismCode}
      />

      <Text weight={700}>2. Dilated Convolutions</Text>

      <Text>
        Dilated convolutions expand receptive field without increasing parameters:
      </Text>

      <CodeBlock
        language="python"
        code={dilatedConvCode}
      />

      <Text weight={700}>3. Dynamic Convolutions</Text>

      <Text>
        Dynamic convolutions adapt kernel weights based on input content:
      </Text>

      <CodeBlock
        language="python"
        code={dynamicConvCode}
      />

      <Text weight={700}>4. Advanced Normalization Techniques</Text>

      <Text>
        Adaptive normalization methods that adjust to input characteristics:
      </Text>

      <CodeBlock
        language="python"
        code={groupNormCode}
      />

      <Text weight={700}>5. Deformable Convolutions</Text>

      <Text>
        Deformable convolutions allow flexible sampling locations:
      </Text>

      <CodeBlock
        language="python"
        code={deformableConvCode}
      />

      <Text weight={700}>6. Comparison of Advanced Techniques</Text>

      <Table>
        <thead>
          <tr>
            <th>Technique</th>
            <th>Advantages</th>
            <th>Computational Cost</th>
            <th>Best Use Cases</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Self-Attention</td>
            <td>Global context capture</td>
            <td>Quadratic complexity</td>
            <td>High-level feature learning</td>
          </tr>
          <tr>
            <td>Dilated Conv</td>
            <td>Large receptive field</td>
            <td>Linear</td>
            <td>Dense prediction tasks</td>
          </tr>
          <tr>
            <td>Dynamic Conv</td>
            <td>Input-dependent processing</td>
            <td>Moderate</td>
            <td>Adaptive feature extraction</td>
          </tr>
          <tr>
            <td>Adaptive Norm</td>
            <td>Better generalization</td>
            <td>Low</td>
            <td>Various domains/scales</td>
          </tr>
          <tr>
            <td>Deformable Conv</td>
            <td>Geometric adaptation</td>
            <td>High</td>
            <td>Object detection/segmentation</td>
          </tr>
        </tbody>
      </Table>

      <Text weight={700}>7. Implementation Considerations</Text>

      <List>
        <List.Item>
          <strong>Memory Efficiency:</strong>
          <List withPadding>
            <List.Item>Use gradient checkpointing for attention</List.Item>
            <List.Item>Optimize kernel sizes for dilated convolutions</List.Item>
            <List.Item>Balance dynamic conv kernel count</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Training Stability:</strong>
          <List withPadding>
            <List.Item>Initialize attention weights carefully</List.Item>
            <List.Item>Use proper learning rates for different components</List.Item>
            <List.Item>Monitor gradient flow through complex operations</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Architecture Design:</strong>
          <List withPadding>
            <List.Item>Combine techniques strategically</List.Item>
            <List.Item>Consider computational budget</List.Item>
            <List.Item>Profile memory usage and throughput</List.Item>
          </List>
        </List.Item>
      </List>

      <Text weight={700}>8. Recent Developments</Text>

      <List>
        <List.Item>
          <strong>Efficient Attention Mechanisms:</strong>
          <List withPadding>
            <List.Item>Linear attention variants</List.Item>
            <List.Item>Sparse attention patterns</List.Item>
            <List.Item>Local-global attention hybrids</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Advanced Architectures:</strong>
          <List withPadding>
            <List.Item>MobileViT and hybrid designs</List.Item>
            <List.Item>ConvNeXt innovations</List.Item>
            <List.Item>MetaFormer architectures</List.Item>
          </List>
        </List.Item>
      </List>
    </Stack>
  );
};

export default AdvancedConcepts;