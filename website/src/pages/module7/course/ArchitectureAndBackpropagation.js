import React from 'react';
import { Title, Text, Stack, List, Grid, Box } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const ArchitectureAndBackpropagation = () => {
  return (
    <Stack spacing="xl" className="w-full">
      <Title order={1} id="network-structure">Neural Network Architecture</Title>
      
      <svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  {/* <!-- Background --> */}
  <rect width="800" height="400" fill="white"/>
  
  {/* <!-- Layer Labels --> */}
  <text x="50" y="30" font-size="16" fill="#333">Input Layer</text>
  <text x="320" y="30" font-size="16" fill="#333">Hidden Layer</text>
  <text x="600" y="30" font-size="16" fill="#333">Output Layer</text>
  
  {/* <!-- Input Layer Neurons --> */}
  <circle cx="100" cy="100" r="20" fill="#6495ED" />
  <circle cx="100" cy="200" r="20" fill="#6495ED" />
  <circle cx="100" cy="300" r="20" fill="#6495ED" />
  
  {/* <!-- Hidden Layer Neurons --> */}
  <circle cx="350" cy="80" r="20" fill="#4CAF50" />
  <circle cx="350" cy="160" r="20" fill="#4CAF50" />
  <circle cx="350" cy="240" r="20" fill="#4CAF50" />
  <circle cx="350" cy="320" r="20" fill="#4CAF50" />
  
  {/* <!-- Output Layer Neurons --> */}
  <circle cx="600" cy="150" r="20" fill="#FFA500" />
  <circle cx="600" cy="250" r="20" fill="#FFA500" />
  
  {/* <!-- Connections from Input to Hidden Layer --> */}
  <g stroke="#999" stroke-width="1.5" opacity="0.6">
    {/* <!-- From first input neuron --> */}
    <line x1="120" y1="100" x2="330" y2="80" />
    <line x1="120" y1="100" x2="330" y2="160" />
    <line x1="120" y1="100" x2="330" y2="240" />
    <line x1="120" y1="100" x2="330" y2="320" />
    
    {/* <!-- From second input neuron --> */}
    <line x1="120" y1="200" x2="330" y2="80" />
    <line x1="120" y1="200" x2="330" y2="160" />
    <line x1="120" y1="200" x2="330" y2="240" />
    <line x1="120" y1="200" x2="330" y2="320" />
    
    {/* <!-- From third input neuron --> */}
    <line x1="120" y1="300" x2="330" y2="80" />
    <line x1="120" y1="300" x2="330" y2="160" />
    <line x1="120" y1="300" x2="330" y2="240" />
    <line x1="120" y1="300" x2="330" y2="320" />
  </g>
  
  {/* <!-- Connections from Hidden to Output Layer --> */}
  <g stroke="#999" stroke-width="1.5" opacity="0.6">
    {/* <!-- To first output neuron --> */}
    <line x1="370" y1="80" x2="580" y2="150" />
    <line x1="370" y1="160" x2="580" y2="150" />
    <line x1="370" y1="240" x2="580" y2="150" />
    <line x1="370" y1="320" x2="580" y2="150" />
    
    {/* <!-- To second output neuron --> */}
    <line x1="370" y1="80" x2="580" y2="250" />
    <line x1="370" y1="160" x2="580" y2="250" />
    <line x1="370" y1="240" x2="580" y2="250" />
    <line x1="370" y1="320" x2="580" y2="250" />
  </g>
  
  {/* <!-- Layer Annotations --> */}
  <text x="60" y="350" font-size="12" fill="#666">x₁, x₂, x₃</text>
  <text x="320" y="350" font-size="12" fill="#666">h₁, h₂, h₃, h₄</text>
  <text x="580" y="350" font-size="12" fill="#666">y₁, y₂</text>
  
  {/* <!-- Weight Annotation Example --> */}
  <text x="200" y="130" font-size="12" fill="#666">w¹ᵢⱼ</text>
  <text x="450" y="130" font-size="12" fill="#666">w²ᵢⱼ</text>
  
  {/* <!-- Mathematical Expression Examples --> */}
  <text x="350" y="30" font-size="12" fill="#666">h = f(W¹x + b¹)</text>
  <text x="600" y="30" font-size="12" fill="#666">y = f(W²h + b²)</text>
</svg>

      {/* Key Components Section - Kept as is */}
      <Grid gutter="md">
        <Grid.Col span={12}>
          <Title order={3} className="mb-4">Key Components</Title>
          <List spacing="sm">
            <List.Item><strong>Neurons:</strong> Basic computational units that perform weighted sum operations</List.Item>
            <List.Item><strong>Weights:</strong> Learnable parameters that determine the strength of connections</List.Item>
            <List.Item><strong>Biases:</strong> Learnable offsets added to each neuron's output</List.Item>
            <List.Item><strong>Activation Functions:</strong> Non-linear transformations applied to neuron outputs</List.Item>
          </List>
        </Grid.Col>
      </Grid>

      {/* Layers Section - Kept as is with enhancement */}
      <Title order={2} id="layers" className="mt-6">Network Layers</Title>
      <Text>
        Neural networks are organized into distinct layers, each serving a specific purpose in transforming input data into desired outputs.
      </Text>

      <Grid gutter="md">
        <Grid.Col span={12}>
          <List spacing="sm">
            <List.Item>
              <strong>Input Layer:</strong> Receives the raw input features (dimension determined by data)
            </List.Item>
            <List.Item>
              <strong>Hidden Layers:</strong> Intermediate layers where most computation occurs
            </List.Item>
            <List.Item>
              <strong>Output Layer:</strong> Produces the final prediction (dimension determined by task)
            </List.Item>
          </List>
        </Grid.Col>
      </Grid>

      {/* Simple Numerical Example */}
      <Title order={2} id="numerical-example" className="mt-6">Simple Neural Network Example</Title>
      
      <Box className="p-4 border rounded">
        <Title order={4}>Single Neuron Computation</Title>
        <Text>Consider a single neuron with 2 inputs:</Text>
        <BlockMath>{`
          \\begin{aligned}
          & \\text{Inputs: } x = [x_1, x_2] = [0.5, 1.0] \\\\
          & \\text{Weights: } w = [w_1, w_2] = [0.3, -0.2] \\\\
          & \\text{Bias: } b = 0.1 \\\\
          & \\text{Activation: ReLU}(x) = max(0, x)
          \\end{aligned}
        `}</BlockMath>
        <Text className="mt-4">Forward computation:</Text>
        <BlockMath>{`
          \\begin{aligned}
          z &= w_1x_1 + w_2x_2 + b \\\\
          &= (0.3 \\times 0.5) + (-0.2 \\times 1.0) + 0.1 \\\\
          &= 0.15 - 0.2 + 0.1 = 0.05 \\\\
          a &= ReLU(0.05) = 0.05
          \\end{aligned}
        `}</BlockMath>
      </Box>

      {/* Error and Loss Introduction */}
      <Title order={2} id="loss-function" className="mt-6">Loss Function and Error Measurement</Title>
      
      <Text>
        The network's performance is measured by comparing its output with desired targets using a loss function. For regression tasks, we often use Mean Squared Error (MSE):
      </Text>

      <Box className="p-4 border rounded">
        <BlockMath>{`L = \\frac{1}{N}\\sum_{i=1}^N (y_i - \\hat{y}_i)^2`}</BlockMath>
        <Text>Where:</Text>
        <List>
          <List.Item><InlineMath>{`y_i`}</InlineMath>: true target value</List.Item>
          <List.Item><InlineMath>{`\\hat{y}_i`}</InlineMath>: network prediction</List.Item>
          <List.Item><InlineMath>{`N`}</InlineMath>: number of samples</List.Item>
        </List>
      </Box>

      {/* Backpropagation Theory */}
      <Title order={2} id="backpropagation" className="mt-6">Backpropagation: The Learning Algorithm</Title>
      
      <Text>
        Backpropagation computes gradients of the loss with respect to network parameters through repeated application of the chain rule.
      </Text>

      <Grid gutter="md">
        <Grid.Col span={12}>
          <Box className="p-4 border rounded">
            <Title order={4}>Forward Pass</Title>
            <Text>For each layer l, compute:</Text>
            <BlockMath>{`
              \\begin{aligned}
              z^{(l)} &= W^{(l)}a^{(l-1)} + b^{(l)} \\\\
              a^{(l)} &= f(z^{(l)})
              \\end{aligned}
            `}</BlockMath>

            <Title order={4} className="mt-4">Backward Pass</Title>
            <Text>Starting from the output layer L, compute:</Text>
            <BlockMath>{`
              \\begin{aligned}
              \\delta^{(L)} &= \\nabla_a L \\odot f'(z^{(L)}) \\\\
              \\delta^{(l)} &= ((W^{(l+1)})^T\\delta^{(l+1)}) \\odot f'(z^{(l)})
              \\end{aligned}
            `}</BlockMath>

            <Text className="mt-4">Parameter gradients:</Text>
            <BlockMath>{`
              \\begin{aligned}
              \\frac{\\partial L}{\\partial W^{(l)}} &= \\delta^{(l)}(a^{(l-1)})^T \\\\
              \\frac{\\partial L}{\\partial b^{(l)}} &= \\delta^{(l)}
              \\end{aligned}
            `}</BlockMath>
          </Box>
        </Grid.Col>
      </Grid>

      {/* Simple Backpropagation Example */}
      <Box className="p-4 border rounded mt-6">
        <Title order={3}>Numerical Backpropagation Example</Title>
        <Text>Using our previous single neuron example, assume target y = 0.2:</Text>
        <BlockMath>{`
          \\begin{aligned}
          \\text{Loss } L &= (a - y)^2 = (0.05 - 0.2)^2 = 0.0225 \\\\
          \\frac{\\partial L}{\\partial a} &= 2(a - y) = 2(0.05 - 0.2) = -0.3 \\\\
          \\frac{\\partial a}{\\partial z} &= 1 \\text{ (ReLU derivative for z > 0)} \\\\
          \\frac{\\partial z}{\\partial w_1} &= x_1 = 0.5
          \\end{aligned}
        `}</BlockMath>
        <Text className="mt-4">Therefore, the gradient for w₁:</Text>
        <BlockMath>{`
          \\frac{\\partial L}{\\partial w_1} = \\frac{\\partial L}{\\partial a} \\frac{\\partial a}{\\partial z} \\frac{\\partial z}{\\partial w_1} = (-0.3)(1)(0.5) = -0.15
        `}</BlockMath>
      </Box>

      {/* Simple Code Illustration */}
      <CodeBlock
        language="python"
        code={`
# Simple numerical example in PyTorch
import torch

# Single neuron setup
x = torch.tensor([0.5, 1.0])
w = torch.tensor([0.3, -0.2], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# Forward pass
z = torch.dot(w, x) + b
a = torch.relu(z)

# Compute loss
y = torch.tensor([0.2])
loss = (a - y)**2

# Backward pass
loss.backward()

# Print gradients
print(f'Weight gradients: {w.grad}')
print(f'Bias gradient: {b.grad}')
`}
      />

<Title order={2} id="backpropagation" className="mt-6">
Backpropagation and Automatic Differentiation
</Title>
<Text>
Backpropagation is the cornerstone algorithm for training neural networks, utilizing the chain rule to compute gradients efficiently through the network.
</Text>

<Box className="p-4 bg-gray-50 rounded mt-4">
<Title order={4}>Mathematical Formulation</Title>
<BlockMath>
  {`\\frac{\\partial L}{\\partial w_{i,j}^{(l)}} = \\frac{\\partial L}{\\partial a_j^{(l)}} \\frac{\\partial a_j^{(l)}}{\\partial z_j^{(l)}} \\frac{\\partial z_j^{(l)}}{\\partial w_{i,j}^{(l)}}`}
</BlockMath>
<Text className="mt-2">
  Where:
</Text>
<List>
  <List.Item><InlineMath>{`L`}</InlineMath> is the loss function</List.Item>
  <List.Item><InlineMath>{`w_{i,j}^{(l)}`}</InlineMath> is the weight connecting neuron i in layer l-1 to neuron j in layer l</List.Item>
  <List.Item><InlineMath>{`a_j^{(l)}`}</InlineMath> is the activation of neuron j in layer l</List.Item>
  <List.Item><InlineMath>{`z_j^{(l)}`}</InlineMath> is the weighted input to neuron j in layer l</List.Item>
</List>
</Box>


      <Title order={2} id="model-capacity" className="mt-6">Model Capacity and Depth</Title>
      <Text>
        The capacity of a neural network is determined by its architecture - specifically the number of layers (depth) 
        and neurons per layer (width). These choices affect the model's ability to learn complex patterns.
      </Text>

      <Grid gutter="md">
        <Grid.Col span={12}>
          <Title order={4}>Impact of Network Size</Title>
          <List spacing="sm">
            <List.Item>
              <strong>Width:</strong> More neurons per layer increase the model's ability to represent complex patterns 
              within each layer
            </List.Item>
            <List.Item>
              <strong>Depth:</strong> More layers allow the model to learn hierarchical representations and more 
              complex functions
            </List.Item>
            <List.Item>
              <strong>Trade-offs:</strong> Larger networks have more capacity but require more data, compute resources, 
              and are prone to overfitting
            </List.Item>
          </List>
        </Grid.Col>
      </Grid>

      <Title order={2} id="computations" className="mt-6">Forward and Backward Computations</Title>
      
      <Stack spacing="md">
        <Text>
          The forward pass computes the output of each layer sequentially. For a single neuron:
        </Text>
        <BlockMath math={`z = \\sum_{i=1}^n w_i x_i + b`} />
        <Text>where:</Text>
        <List>
          <List.Item>z is the pre-activation output</List.Item>
          <List.Item>w_i are the weights</List.Item>
          <List.Item>x_i are the inputs</List.Item>
          <List.Item>b is the bias term</List.Item>
        </List>
        
        <Text className="mt-4">
          The backward pass computes gradients using the chain rule:
        </Text>
        <BlockMath math={`\\frac{\\partial L}{\\partial w_i} = \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w_i}`} />
      </Stack>

    </Stack>
  );
};

export default ArchitectureAndBackpropagation;