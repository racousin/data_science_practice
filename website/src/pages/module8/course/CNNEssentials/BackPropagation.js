import React from 'react';
import { Text, Stack, List, Grid, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { BlockMath, InlineMath } from 'react-katex';

const BackPropagation = () => {
  const backpropVisualizationCode = `
import torch
import torch.nn as nn

class ConvLayerWithGradients(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        # Register hook to capture gradients
        self.conv.weight.register_hook(self._log_gradients)
        self.gradients = []
    
    def _log_gradients(self, grad):
        self.gradients.append(grad.detach().cpu())
    
    def forward(self, x):
        return self.conv(x)

# Example usage
model = ConvLayerWithGradients()
x = torch.randn(1, 1, 28, 28)  # Sample input
y = model(x)
loss = y.pow(2).mean()  # Simple loss function
loss.backward()

print("Gradient shape:", model.gradients[-1].shape)
print("Gradient statistics:")
print(f"Mean: {model.gradients[-1].mean():.4f}")
print(f"Std: {model.gradients[-1].std():.4f}")`;

  const chainRuleExampleCode = `
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvWithGradientFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.saved_gradients = {}
        
        # Register hooks for all parameters
        def get_hook(name):
            def hook(grad):
                self.saved_gradients[name] = grad.detach().clone()
            return hook
        
        self.conv1.weight.register_hook(get_hook('conv1'))
        self.conv2.weight.register_hook(get_hook('conv2'))
    
    def forward(self, x):
        # Forward pass with intermediate values
        x1 = self.conv1(x)
        a1 = F.relu(x1)  # First activation
        x2 = self.conv2(a1)
        a2 = F.relu(x2)  # Second activation
        
        return a2

# Demonstrate gradient flow
model = ConvWithGradientFlow()
criterion = nn.MSELoss()

# Forward pass
x = torch.randn(1, 1, 28, 28)
target = torch.randn(1, 32, 28, 28)
output = model(x)
loss = criterion(output, target)

# Backward pass
loss.backward()

# Analyze gradients
for name, grad in model.saved_gradients.items():
    print(f"{name} gradient stats:")
    print(f"Mean: {grad.mean():.4f}")
    print(f"Std: {grad.std():.4f}")`;

  const gradientFlowAnalysisCode = `
class GradientFlowAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
        self.gradient_stats = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def get_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradient_stats[name] = {
                    'mean': grad_output[0].mean().item(),
                    'std': grad_output[0].std().item(),
                    'max': grad_output[0].abs().max().item()
                }
            return hook
        
        # Register hooks for each layer
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module.register_backward_hook(get_hook(name))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def analyze_gradients(self):
        return pd.DataFrame(self.gradient_stats).T`;

  return (
    <Stack spacing="md">
      <Text>
        Backpropagation in CNNs follows the chain rule of calculus but requires special
        consideration for convolution operations and pooling layers.
      </Text>

      <Text weight={700}>1. Gradient Flow in Convolution Layers</Text>

      <Text>
        The gradient with respect to a convolution kernel is:
      </Text>

      <BlockMath>
        {`\\frac{\\partial L}{\\partial W} = \\sum_{i,j} \\frac{\\partial L}{\\partial Y_{i,j}} \\frac{\\partial Y_{i,j}}{\\partial W}`}
      </BlockMath>

      <Text>
        Where:
      </Text>
      <List>
        <List.Item><InlineMath>L</InlineMath> is the loss function</List.Item>
        <List.Item><InlineMath>W</InlineMath> is the convolution kernel</List.Item>
        <List.Item><InlineMath>{`Y_{i,j}`}</InlineMath> is the output feature map at position (i,j)</List.Item>
      </List>

      <Text>Here's how to visualize gradients in a convolution layer:</Text>

      <CodeBlock
        language="python"
        code={backpropVisualizationCode}
      />

      <Text weight={700}>2. Chain Rule Application</Text>

      <Text>
        For a typical CNN layer sequence (Conv → ReLU → Pool), the chain rule gives us:
      </Text>

      <BlockMath>
        {`\\frac{\\partial L}{\\partial W_l} = \\frac{\\partial L}{\\partial Y_l} \\cdot \\frac{\\partial Y_l}{\\partial X_l} \\cdot \\frac{\\partial X_l}{\\partial W_l}`}
      </BlockMath>

      <Text>Implementation example showing gradient flow through multiple layers:</Text>

      <CodeBlock
        language="python"
        code={chainRuleExampleCode}
      />

      <Text weight={700}>3. Gradient Flow Analysis</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text weight={600}>Common Issues:</Text>
          <List>
            <List.Item>Vanishing gradients</List.Item>
            <List.Item>Exploding gradients</List.Item>
            <List.Item>Gradient saturation</List.Item>
            <List.Item>Dead ReLUs</List.Item>
          </List>
        </Grid.Col>

        <Grid.Col span={12} md={6}>
          <Text weight={600}>Solutions:</Text>
          <List>
            <List.Item>Batch normalization</List.Item>
            <List.Item>Residual connections</List.Item>
            <List.Item>Careful initialization</List.Item>
            <List.Item>Gradient clipping</List.Item>
          </List>
        </Grid.Col>
      </Grid>

      <Text>Tool for analyzing gradient flow through a network:</Text>

      <CodeBlock
        language="python"
        code={gradientFlowAnalysisCode}
      />

      <Text weight={700}>4. Backpropagation Through Pooling Layers</Text>

      <Grid>
        <Grid.Col span={12} md={6}>
          <Text>Max Pooling Gradient:</Text>
          <BlockMath>
            {`\\frac{\\partial L}{\\partial x_i} = \\begin{cases} 
              \\frac{\\partial L}{\\partial y_j} & \\text{if } x_i \\text{ was max} \\\\
              0 & \\text{otherwise}
            \\end{cases}`}
          </BlockMath>
        </Grid.Col>

        <Grid.Col span={12} md={6}>
          <Text>Average Pooling Gradient:</Text>
          <BlockMath>
            {`\\frac{\\partial L}{\\partial x_i} = \\frac{1}{n} \\frac{\\partial L}{\\partial y_j}`}
          </BlockMath>
        </Grid.Col>
      </Grid>

      <Text weight={700}>5. Practical Considerations</Text>

      <List>
        <List.Item>
          <strong>Gradient Checkpointing:</strong> For memory efficiency in deep networks
          <List withPadding>
            <List.Item>Trade computation for memory</List.Item>
            <List.Item>Useful for very deep networks</List.Item>
          </List>
        </List.Item>
        
        <List.Item>
          <strong>Mixed Precision Training:</strong>
          <List withPadding>
            <List.Item>Use FP16 for forward/backward passes</List.Item>
            <List.Item>Maintain master weights in FP32</List.Item>
          </List>
        </List.Item>

        <List.Item>
          <strong>Gradient Accumulation:</strong>
          <List withPadding>
            <List.Item>Useful for large batch training</List.Item>
            <List.Item>Helps with limited GPU memory</List.Item>
          </List>
        </List.Item>
      </List>

      <Text weight={700}>6. Monitoring and Debugging</Text>

      <Text>
        Key metrics to monitor during training:
      </Text>

      <List>
        <List.Item>Gradient magnitude distribution</List.Item>
        <List.Item>Layer-wise gradient statistics</List.Item>
        <List.Item>Update to weight ratio</List.Item>
        <List.Item>Activation patterns</List.Item>
      </List>

      <Text>
        The gradient flow analysis helps identify:
      </Text>

      <List>
        <List.Item>Layers that are learning effectively</List.Item>
        <List.Item>Potential optimization issues</List.Item>
        <List.Item>Architecture bottlenecks</List.Item>
        <List.Item>Training stability problems</List.Item>
      </List>
    </Stack>
  );
};

export default BackPropagation;