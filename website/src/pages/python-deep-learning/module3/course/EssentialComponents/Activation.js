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
      <div data-slide>
      <Title order={2}>Activation Functions</Title>


        
        <Grid mb="lg">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={reluData} 
              title="ReLU (Rectified Linear Unit)"
              equation={"f(x) = \\max(0, x)"}
            />
                    <CodeBlock
          language="python"
          code={`nn.ReLU()`}/>
          </Grid.Col>
          
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={sigmoidData} 
              title="Sigmoid"
              equation={"f(x) = \\frac{1}{1 + e^{-x}}"}
            />

                                <CodeBlock
          language="python"
          code={`nn.Sigmoid()`}/>
          </Grid.Col>
          </Grid>
          </div>
          <div data-slide>
          <Grid mb="lg">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={tanhData} 
              title="Tanh"
              equation={"f(x) = \\tanh(x)"}
            />
                                <CodeBlock
          language="python"
          code={`nn.Tanh()`}/>
          </Grid.Col>
          
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ActivationPlot 
              data={leakyReluData} 
              title="Leaky ReLU"
              equation={"f(x) = \\max(0.01x, x)"}
            />
                                <CodeBlock
          language="python"
          code={`nn.LeakyReLU(0.1)`}/>
          </Grid.Col>
        </Grid>

      </div>

      {/* Mathematical Properties Section */}
      
        <div data-slide>
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
      </div>

    </Stack>
  );
};

export default Activation;