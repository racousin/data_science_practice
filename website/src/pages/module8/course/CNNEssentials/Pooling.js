import React from 'react';
import { Text, Stack, List, Code, Grid } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const Pooling = () => {
  const poolingCode = `
import torch
import torch.nn as nn

class PoolingExamples(nn.Module):
    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        # Average pooling layer
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Spatial pyramid pooling
        self.spp = nn.Sequential(
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.AdaptiveMaxPool2d((1, 1))
        )
    
    def forward(self, x):
        # Apply different pooling operations
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        global_pooled = self.global_avg_pool(x)
        
        # SPP forward pass
        spp_4x4 = self.spp[0](x)
        spp_2x2 = self.spp[1](x)
        spp_1x1 = self.spp[2](x)
        
        return {
            'max_pooled': max_pooled,
            'avg_pooled': avg_pooled,
            'global_pooled': global_pooled,
            'spp': [spp_4x4, spp_2x2, spp_1x1]
        }

# Example usage
model = PoolingExamples()
x = torch.randn(1, 64, 32, 32)  # Sample feature map
results = model(x)

print(f"Input shape: {x.shape}")
print(f"Max pooled shape: {results['max_pooled'].shape}")
print(f"Avg pooled shape: {results['avg_pooled'].shape}")
print(f"Global pooled shape: {results['global_pooled'].shape}")
print(f"SPP shapes: {[t.shape for t in results['spp']]}")`;

  const customPoolingCode = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        # Unfold input into patches
        patches = F.unfold(
            x,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride)
        )
        
        # Reshape patches for processing
        b, c, n = patches.size()
        patches = patches.view(b, -1, self.kernel_size * self.kernel_size, n)
        
        # Calculate probabilities (softmax over each patch)
        probs = F.softmax(patches, dim=2)
        
        # Stochastic selection based on probabilities
        if self.training:
            indices = torch.multinomial(probs.view(-1, self.kernel_size * self.kernel_size), 1)
            selected = torch.gather(patches, 2, indices.view(b, -1, 1, n))
        else:
            # During inference, use expected value
            selected = (patches * probs).sum(dim=2, keepdim=True)
        
        # Reshape back to spatial dimensions
        output_h = int((x.size(2) - self.kernel_size) / self.stride + 1)
        output_w = int((x.size(3) - self.kernel_size) / self.stride + 1)
        return selected.view(b, -1, output_h, output_w)`;

  return (
    <Stack spacing="md">
      <Text>
        Pooling layers are essential components in CNNs that reduce spatial dimensions
        while retaining important features. They help achieve:
      </Text>
      
      <List>
        <List.Item>Translation invariance</List.Item>
        <List.Item>Spatial hierarchy of features</List.Item>
        <List.Item>Computational efficiency</List.Item>
        <List.Item>Control of overfitting</List.Item>
      </List>

      <Text weight={700}>1. Common Pooling Operations</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text>Max Pooling:</Text>
          <BlockMath>
            {`MaxPool(X)_{i,j} = \\max_{(m,n) \\in R_{i,j}} x_{m,n}`}
          </BlockMath>
        </Grid.Col>
        
        <Grid.Col span={12} md={6}>
          <Text>Average Pooling:</Text>
          <BlockMath>
            {`AvgPool(X)_{i,j} = \\frac{1}{|R_{i,j}|} \\sum_{(m,n) \\in R_{i,j}} x_{m,n}`}
          </BlockMath>
        </Grid.Col>
      </Grid>

      <Text>
        Where <InlineMath>{`R_{i,j}`}</InlineMath> represents the pooling region centered
        at position <InlineMath>(i,j)</InlineMath>.
      </Text>

      <Text weight={700}>2. Implementation Examples</Text>
      
      <CodeBlock
        language="python"
        code={poolingCode}
      />

      <Text weight={700}>3. Advanced Pooling Techniques</Text>

      <Stack spacing="xs">
        <Text>
          a) Spatial Pyramid Pooling (SPP):
        </Text>
        <Text size="sm">
          SPP adapts features of varying sizes into fixed-length representations by pooling
          at multiple scales. This is particularly useful for handling inputs of varying sizes
          or for maintaining multi-scale information.
        </Text>

        <Text>
          b) Stochastic Pooling:
        </Text>
        <Text size="sm">
          A probabilistic alternative to max/average pooling that can help prevent overfitting.
          Here's an implementation:
        </Text>
      </Stack>

      <CodeBlock
        language="python"
        code={customPoolingCode}
      />

      <Text weight={700}>4. Practical Considerations</Text>

      <List>
        <List.Item>
          <strong>Kernel Size & Stride:</strong> Most commonly used is 2×2 with stride 2,
          reducing spatial dimensions by half
        </List.Item>
        <List.Item>
          <strong>Pooling Type Selection:</strong>
          <List withPadding>
            <List.Item>Max Pooling: Better for extracting texture and edge features</List.Item>
            <List.Item>Average Pooling: Better for maintaining spatial information</List.Item>
            <List.Item>Global Pooling: Useful in final layers to reduce parameters</List.Item>
          </List>
        </List.Item>
        <List.Item>
          <strong>Modern Trends:</strong> Some architectures (like ResNet variants) reduce
          pooling layers in favor of strided convolutions
        </List.Item>
      </List>

      <Text weight={700}>5. Impact on Feature Maps</Text>
      
      <Text>
        The output size after pooling can be calculated using:
      </Text>
      
      <BlockMath>
        {`O = \\left\\lfloor\\frac{N - K}{S}\\right\\rfloor + 1`}
      </BlockMath>
      
      <Text>
        Where <InlineMath>N</InlineMath> is input size, <InlineMath>K</InlineMath> is kernel size,
        and <InlineMath>S</InlineMath> is stride. Unlike convolution, pooling typically doesn't
        use padding as we want to reduce spatial dimensions.
      </Text>

      <Text>
        Remember that while pooling helps reduce computational complexity and prevent overfitting,
        excessive pooling can lead to loss of important spatial information. Modern architectures
        often balance this trade-off by:
      </Text>

      <List>
        <List.Item>Using fewer pooling layers</List.Item>
        <List.Item>Combining different pooling types</List.Item>
        <List.Item>Employing learned pooling operations</List.Item>
        <List.Item>Using skip connections to preserve spatial information</List.Item>
      </List>
    </Stack>
  );
};

export default Pooling;