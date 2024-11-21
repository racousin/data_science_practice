import React from 'react';
import { Text, Stack, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const ConvolutionBasics = () => {
  const convolutionCode = `
import torch
import torch.nn as nn

class ConvolutionExample(nn.Module):
    def __init__(self):
        super().__init__()
        # Basic convolution with batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,          # RGB input
                out_channels=64,        # Number of filters
                kernel_size=3,          # 3x3 kernel
                stride=1,               # Standard stride
                padding=1               # Same padding
            ),
            nn.BatchNorm2d(64),        # Batch normalization
            nn.ReLU()                  # Activation function
        )
        
    def forward(self, x):
        return self.conv1(x)

# Example usage
model = ConvolutionExample()
x = torch.randn(1, 3, 32, 32)  # Single RGB image 32x32
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")`;

  return (
    <Stack spacing="md">
      <Text>
        The convolution operation is fundamental to CNNs. It performs a sliding window
        operation across the input, computing element-wise multiplications and sums.
      </Text>

      <BlockMath>
        {`(f * g)[n] = \\sum_{m=-\\infty}^{\\infty} f[m]g[n-m]`}
      </BlockMath>

      <Text>
        Key components of a convolution layer:
      </Text>
      
      <ul>
        <li>
          <strong>Kernel Size:</strong> Defines the spatial extent of the convolution
          (e.g., 3×3, 5×5)
        </li>
        <li>
          <strong>Stride:</strong> Controls how the kernel moves across the input
          (<InlineMath>{"s \\in \\mathbb{N}"}</InlineMath>)
        </li>
        <li>
          <strong>Padding:</strong> Added border around input to control output size
        </li>
        <li>
          <strong>Channels:</strong> Number of input/output feature maps
        </li>
      </ul>

      <Text>
        The output size of a convolution layer can be calculated using:
      </Text>

      <BlockMath>
        {`O = \\left\\lfloor\\frac{N - K + 2P}{S}\\right\\rfloor + 1`}
      </BlockMath>

      <Text>
        where:
      </Text>
      <ul>
        <li><InlineMath>O</InlineMath>: Output size</li>
        <li><InlineMath>N</InlineMath>: Input size</li>
        <li><InlineMath>K</InlineMath>: Kernel size</li>
        <li><InlineMath>P</InlineMath>: Padding</li>
        <li><InlineMath>S</InlineMath>: Stride</li>
      </ul>

      <Text>
        Here's a practical implementation using PyTorch 2.5:
      </Text>

      <CodeBlock
        language="python"
        code={convolutionCode}
      />

      <Text>
        Common practices for convolution layers:
      </Text>
      <ul>
        <li>Use small kernels (3×3) with multiple layers instead of large kernels</li>
        <li>Maintain spatial dimensions with appropriate padding</li>
        <li>Use batch normalization after convolution</li>
        <li>Apply non-linear activation (ReLU) after batch normalization</li>
      </ul>
    </Stack>
  );
};

export default ConvolutionBasics;