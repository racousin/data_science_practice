import React from 'react';
import { Title, Text, Stack, List, Alert, Divider, Code } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const SkipConnections = () => {
  return (
    <Stack spacing="md">
      <Text>
        Skip connections are a fundamental architectural pattern in neural networks that create shortcuts 
        to bypass one or more layers. They address key challenges in training deep networks by providing 
        alternative pathways for information and gradient flow. While residual connections (introduced by He et al. 
        in 2015) are the most well-known type of skip connection, the general principle has been applied in 
        various forms across different network architectures.
      </Text>

      <Title order={3}>The Problem: Vanishing Gradients in Deep Networks</Title>
      <Text>
        As neural networks become deeper, they face significant challenges:
      </Text>
      <List>
        <List.Item>Gradients tend to vanish as they're backpropagated through many layers</List.Item>
        <List.Item>Learning becomes increasingly difficult in the earlier layers</List.Item>
        <List.Item>Deeper networks can paradoxically perform worse than their shallower counterparts</List.Item>
        <List.Item>Training accuracy saturates and then degrades with increasing network depth</List.Item>
      </List>

      <Title order={3}>Types of Skip Connections</Title>
      <Text>
        Skip connections can be implemented in different ways:
      </Text>
      <List>
        <List.Item>
          <strong>Identity skip connections:</strong> The input is directly added to the output of a block 
          (<InlineMath>{`y = F(x) + x`}</InlineMath>). This is the core of residual connections.
        </List.Item>
        <List.Item>
          <strong>Concatenation-based skip connections:</strong> The input is concatenated with the output along the feature dimension 
          (<InlineMath>{`y = [F(x), x]`}</InlineMath>). Used in architectures like U-Net and DenseNet.
        </List.Item>
        <List.Item>
          <strong>Projection skip connections:</strong> The input is transformed by a linear projection before being added 
          (<InlineMath>{`y = F(x) + Wx`}</InlineMath>). Used when dimensions need to be matched.
        </List.Item>
        <List.Item>
          <strong>Multi-level skip connections:</strong> Connect layers across multiple depths, creating a richer information flow.
        </List.Item>
      </List>

      <Title order={3}>Core Principles of Skip Connections</Title>
      <Text>
        Regardless of specific implementation, all skip connections share these core principles:
      </Text>
      <List>
        <List.Item>They create alternative paths for information flow</List.Item>
        <List.Item>They help mitigate the vanishing gradient problem</List.Item>
        <List.Item>They make it easier for the network to learn identity mappings when needed</List.Item>
        <List.Item>They enable the training of much deeper architectures</List.Item>
      </List>

      <Title order={4}>Basic Skip Connection</Title>
      <BlockMath>
        {`y = G(x) + H(x)`}
      </BlockMath>
      <Text>
        Where <InlineMath>{`G(x)`}</InlineMath> represents the main path (typically multiple layers with non-linear activations), 
        and <InlineMath>{`H(x)`}</InlineMath> represents the skip path (which could be identity, a projection, or another transformation).
      </Text>

      <Title order={3}>Gradient Flow in Networks with Skip Connections</Title>
      <Text>
        The key advantage of skip connections is how they affect gradient flow during backpropagation.
        To understand this, we need to examine how gradients flow through the network to update the parameters.
      </Text>

      <Title order={4}>The Vanishing Gradient Problem</Title>
      <Text>
        In traditional feed-forward networks with L layers, the gradient at the early layers diminishes exponentially 
        with network depth due to repeated multiplication of small weights during backpropagation.
      </Text>

      <Title order={4}>Chain Rule in Plain Networks</Title>
      <Text>
        In a plain network where each layer l computes <InlineMath>{`x_{l+1} = f_l(x_l, W_l)`}</InlineMath>, the gradient of 
        the loss E with respect to weights at a layer i is:
      </Text>
      <BlockMath>
        {`\\frac{\\partial E}{\\partial W_i} = \\frac{\\partial E}{\\partial x_L} 
        \\frac{\\partial x_L}{\\partial x_{L-1}} \\frac{\\partial x_{L-1}}{\\partial x_{L-2}} \\cdots 
        \\frac{\\partial x_{i+1}}{\\partial x_i} \\frac{\\partial x_i}{\\partial W_i}`}
      </BlockMath>
      <Text>
        This can be rewritten as:
      </Text>
      <BlockMath>
        {`\\frac{\\partial E}{\\partial W_i} = \\frac{\\partial E}{\\partial x_L} 
        \\prod_{l=i}^{L-1} \\frac{\\partial x_{l+1}}{\\partial x_l} \\cdot \\frac{\\partial x_i}{\\partial W_i}`}
      </BlockMath>
      <Text>
        When each partial derivative <InlineMath>{`\\frac{\\partial x_{l+1}}{\\partial x_l}`}</InlineMath> has a magnitude less than 1, 
        the product quickly approaches zero as the network depth increases, causing the vanishing gradient problem.
      </Text>

      <Title order={4}>Step 1: Gradient Flow Through Skip Connections</Title>
      <Text>
        Now, let's consider how the gradient flows with respect to activations in a network with skip connections. 
        For a network with L layers and loss function E, if we have a simple skip connection where <InlineMath>{`y_l = G_l(x_l) + x_l`}</InlineMath>, 
        the gradient flowing to the activations at layer l becomes:
      </Text>
      <BlockMath>
        {`\\frac{\\partial E}{\\partial x_l} = \\frac{\\partial E}{\\partial y_L} \\frac{\\partial y_L}{\\partial x_l}`}
      </BlockMath>
      <Text>
        For any layer with a skip connection:
      </Text>
      <BlockMath>
        {`\\frac{\\partial y_l}{\\partial x_l} = \\frac{\\partial G_l(x_l)}{\\partial x_l} + \\frac{\\partial x_l}{\\partial x_l} = \\frac{\\partial G_l(x_l)}{\\partial x_l} + 1`}
      </BlockMath>
      <Text>
        In a deep network with many layers that have skip connections, we get:
      </Text>
      <BlockMath>
        {`\\frac{\\partial y_L}{\\partial x_l} = \\prod_{i=l}^{L-1} \\frac{\\partial y_{i+1}}{\\partial y_i} \\cdot \\frac{\\partial y_l}{\\partial x_l} = \\prod_{i=l}^{L-1} \\frac{\\partial y_{i+1}}{\\partial y_i} \\cdot \\left(\\frac{\\partial G_l(x_l)}{\\partial x_l} + 1\\right)`}
      </BlockMath>

      <Title order={4}>The Critical "+1" Term</Title>
      <Text>
        The addition of the identity term ("+1") in the derivative creates a direct gradient pathway. This is the key insight:
      </Text>
      <BlockMath>
        {`\\frac{\\partial y_L}{\\partial x_l} = \\prod_{i=l}^{L-1} \\left(\\frac{\\partial G_i(y_i)}{\\partial y_i} + 1\\right) \\geq 1`}
      </BlockMath>
      <Text>
        Even if all <InlineMath>{`\\frac{\\partial G_i(y_i)}{\\partial y_i} = 0`}</InlineMath>, the gradient 
        <InlineMath>{`\\frac{\\partial y_L}{\\partial x_l} = 1`}</InlineMath>, making it impossible for gradients to vanish 
        completely due to network depth alone.
      </Text>

      <Alert color="green">
        <strong>Crucial benefit:</strong> The addition of the identity term creates a direct gradient pathway that guarantees 
        at least one path for gradients to flow backward without being multiplied by potentially small weight derivatives.
      </Alert>

      <Title order={4}>Step 2: Gradient Flow to Parameters</Title>
      <Text>
        The improved gradient flow to activations propagates to parameter gradients:
      </Text>
      <BlockMath>
        {`\\frac{\\partial E}{\\partial W_l} = \\frac{\\partial E}{\\partial x_l} \\frac{\\partial x_l}{\\partial W_l}`}
      </BlockMath>
      <Text>
        Since skip connections improve <InlineMath>{`\\frac{\\partial E}{\\partial x_l}`}</InlineMath> through the "+1" term, 
        this benefit extends to parameter gradients, allowing for effective training of deeper networks.
      </Text>

      <Title order={3}>Difference Between Skip Connections and Residual Connections</Title>
      <Text>
        While often used interchangeably, these terms have subtle differences:
      </Text>
      <List>
        <List.Item>
          <strong>Skip Connection:</strong> A general term for any architecture where information from earlier layers 
          "skips" forward to later layers. Can be implemented through addition, concatenation, or other operations.
        </List.Item>
        <List.Item>
          <strong>Residual Connection:</strong> A specific type of skip connection that uses identity mappings (or simple 
          projections) added to the output of a block, following the formula <InlineMath>{`y = F(x) + x`}</InlineMath>. 
          Specifically designed to learn the "residual" between input and output.
        </List.Item>
      </List>
      <Text>
        In essence, residual connections are a subset of skip connections with a particular implementation (addition) 
        and purpose (learning residuals). All residual connections are skip connections, but not all skip connections 
        are residual connections.
      </Text>

      <Title order={3}>Implementation with PyTorch</Title>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class AdditionSkipBlock(nn.Module):
    """Basic skip connection that adds the input to the output (residual-style)"""
    def __init__(self, input_dim, hidden_dim):
        super(AdditionSkipBlock, self).__init__()
        
        # Main path
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)  # Output dim matches input for addition
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        
        # Add skip connection
        out = out + identity
        out = self.relu(out)
        
        return out

class ConcatenationSkipBlock(nn.Module):
    """Skip connection that concatenates the input with the output"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConcatenationSkipBlock, self).__init__()
        
        # Main path
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim - input_dim)  # Reduced output dimension for concatenation
        
    def forward(self, x):
        # Main path
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        
        # Concatenate skip connection
        out = torch.cat([out, x], dim=1)  # Concatenate along feature dimension
        
        return out

class ProjectionSkipBlock(nn.Module):
    """Skip connection with projection to match dimensions"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionSkipBlock, self).__init__()
        
        # Main path
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        # Projection for skip connection
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Main path
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        
        # Projected skip connection
        out = out + self.projection(x)
        out = self.relu(out)
        
        return out

# Example network with different types of skip connections
class SkipConnectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SkipConnectionNetwork, self).__init__()
        
        # First layer
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        current_dim = hidden_dims[0]
        
        # Build the network with various skip connections
        self.blocks = nn.ModuleList()
        
        # First block - addition skip connection (residual style)
        self.blocks.append(AdditionSkipBlock(current_dim, current_dim // 2))
        
        # Second block - projection skip connection
        self.blocks.append(ProjectionSkipBlock(current_dim, hidden_dims[1], hidden_dims[2]))
        current_dim = hidden_dims[2]
        
        # Third block - concatenation skip connection
        self.blocks.append(ConcatenationSkipBlock(current_dim, hidden_dims[3], hidden_dims[4]))
        current_dim = hidden_dims[4]  # Will be original dim + new features
        
        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim)
    
    def forward(self, x):
        x = self.first_layer(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.output_layer(x)
        return x

# Example usage
model = SkipConnectionNetwork(
    input_dim=10,
    hidden_dims=[64, 32, 64, 32, 96],  # The last dimension will be 64+32 due to concatenation
    output_dim=1
)

# Input tensor [batch_size, feature_dim]
x = torch.randn(32, 10)
output = model(x)  # Shape: [32, 1]`}
      />

    </Stack>
  );
};

export default SkipConnections;