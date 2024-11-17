import React from 'react';
import { Title, Text, Stack, Grid, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const ActivationPlot = ({ data, title, equation }) => {
  // SVG dimensions
  const width = 300;
  const height = 200;
  const margin = 30;
  
  // Scale points to SVG coordinates
  const xScale = (width - 2 * margin) / 10;
  const yScale = (height - 2 * margin) / 2;
  
  const scaledPoints = data.map(({ x, y }) => [
    margin + (x + 5) * xScale,
    margin + height - 2 * margin - (y + 1) * yScale
  ]);
  
  const pathD = scaledPoints.map((point, i) => 
    (i === 0 ? 'M' : 'L') + point.join(',')
  ).join(' ');

  return (
    <div className="w-full">
      <Title order={4} className="mb-2">{title}</Title>
      <BlockMath>{equation}</BlockMath>
      <svg 
        viewBox={`0 0 ${width} ${height}`} 
        className="w-full h-full"
        style={{ maxWidth: '400px' }}
      >
        {/* Axes */}
        <line 
          x1={margin} 
          y1={height - margin} 
          x2={width - margin} 
          y2={height - margin} 
          stroke="#ced4da" 
          strokeWidth="1"
        />
        <line 
          x1={margin} 
          y1={margin} 
          x2={margin} 
          y2={height - margin} 
          stroke="#ced4da" 
          strokeWidth="1"
        />
        
        {/* Function curve */}
        <path
          d={pathD}
          fill="none"
          stroke="#228be6"
          strokeWidth="2"
        />

        {/* Origin point */}
        {title !== 'Sigmoid' && (
  <circle 
    cx={margin + 5 * xScale} 
    cy={height - margin - yScale} 
    r="2" 
    fill="#228be6" 
  />
)}

      </svg>
    </div>
  );
};

const generatePlotData = (func, start = -5, end = 5, steps = 100) => {
  const data = [];
  const step = (end - start) / steps;
  for (let x = start; x <= end; x += step) {
    data.push({ x, y: func(x) });
  }
  return data;
};

const Activation = () => {
  // Generate data for each activation function
  const reluData = generatePlotData(x => Math.max(0, x));
  const sigmoidData = generatePlotData(x => 1 / (1 + Math.exp(-x)));
  const tanhData = generatePlotData(x => Math.tanh(x));
  const leakyReluData = generatePlotData(x => x > 0 ? x : 0.01 * x);

  return (
    <Stack spacing="xl" className="w-full">
      

      {/* Common Functions Section */}
      <section>
        
        <Grid mb="lg">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={reluData} 
              title="ReLU (Rectified Linear Unit)"
              equation={"f(x) = \\max(0, x)"}
            />
          </Grid.Col>
          
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={sigmoidData} 
              title="Sigmoid"
              equation={"f(x) = \\frac{1}{1 + e^{-x}}"}
            />
          </Grid.Col>
          
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={tanhData} 
              title="Tanh"
              equation={"f(x) = \\tanh(x)"}
            />
          </Grid.Col>
          
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={leakyReluData} 
              title="Leaky ReLU"
              equation={"f(x) = \\max(0.01x, x)"}
            />
          </Grid.Col>
        </Grid>

        <CodeBlock
          language="python"
          code={`
import torch.nn as nn

# Using PyTorch's built-in activation functions
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),           # Most common for hidden layers
    nn.Linear(20, 20),
    nn.LeakyReLU(0.01),  # Alternative to ReLU
    nn.Linear(20, 10),
    nn.Tanh()         # Alternative to ReLU
)`}
        />
      </section>

      {/* Mathematical Properties Section */}
      <section>
        
        <Table>
          <thead>
            <tr>
              <th>Function</th>
              <th>Range</th>
              <th>Derivative</th>
              <th>Key Properties</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>ReLU</td>
              <td>[0, ∞)</td>
              <td><InlineMath>{"f'(x) = \\begin{cases} 1 & x > 0 \\\\ 0 & x \\leq 0 \\end{cases}"}</InlineMath></td>
              <td>Non-saturating, sparse activation</td>
            </tr>
            <tr>
              <td>Sigmoid</td>
              <td>(0, 1)</td>
              <td><InlineMath>{"f'(x) = f(x)(1-f(x))"}</InlineMath></td>
              <td>Smooth, bounded output</td>
            </tr>
            <tr>
              <td>Tanh</td>
              <td>(-1, 1)</td>
              <td><InlineMath>{"f'(x) = 1 - f(x)^2"}</InlineMath></td>
              <td>Zero-centered, bounded output</td>
            </tr>
            <tr>
              <td>Leaky ReLU</td>
              <td>(-∞, ∞)</td>
              <td><InlineMath>{"f'(x) = \\begin{cases} 1 & x > 0 \\\\ 0.01 & x \\leq 0 \\end{cases}"}</InlineMath></td>
              <td>Prevents dying ReLU problem</td>
            </tr>
          </tbody>
        </Table>
      </section>

      {/* Usage Guidelines Section */}
      {/* <section>
        <Title order={2} id="usage-guidelines" mb="md">Usage Guidelines</Title>
        
        <Table mb="lg">
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Recommended Activation</th>
              <th>Rationale</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Hidden Layers</td>
              <td>ReLU</td>
              <td>Fast training, no vanishing gradient for positive values</td>
            </tr>
            <tr>
              <td>Binary Classification</td>
              <td>Sigmoid</td>
              <td>Output interpretable as probability</td>
            </tr>
            <tr>
              <td>Multi-class Classification</td>
              <td>Softmax</td>
              <td>Normalized probability distribution across classes</td>
            </tr>
            <tr>
              <td>Deep Networks</td>
              <td>Leaky ReLU</td>
              <td>Prevents dying ReLU problem in deep architectures</td>
            </tr>
          </tbody>
        </Table>

        <CodeBlock
          language="python"
          code={`
# Example of activation function usage in a practical network
class DeepNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        
        # Different activations for different purposes
        self.relu = nn.ReLU()           # Hidden layers
        self.leaky_relu = nn.LeakyReLU(0.01)  # Alternative for deep networks
        self.softmax = nn.Softmax(dim=1) # Multi-class classification
    
    def forward(self, x):
        x = self.relu(self.layer1(x))      # First hidden layer
        x = self.leaky_relu(self.layer2(x)) # Second hidden layer
        x = self.layer3(x)                 # No activation (use with CrossEntropyLoss)
        return x

# Initialize and test the model
model = DeepNetwork(input_size=784, hidden_size=256, num_classes=10)
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)
print(f"Output shape: {output.shape}")  # Should be (32, 10)`}
        />
      </section> */}
    </Stack>
  );
};

export default Activation;