import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const MLPFundamentals = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Part 3: Multi Layer Perceptron in a nutshell */}
        <div data-slide>
          <Title order={1} mb="xl">
            Part 3: Multi Layer Perceptron in a nutshell
          </Title>
          
          {/* Neuron as Computational Unit */}
          
            <Title order={2} mb="xl" id="neuron">
              Neuron as Computational Unit
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 mb-6">
              <Title order={3} mb="md">From Biology to Mathematics</Title>
              <Text size="lg" mb="md">
                The artificial neuron is the fundamental building block of neural networks. It's a mathematical 
                abstraction inspired by biological neurons, but operates purely through linear algebra.
              </Text>
              
              <Paper className="p-4 bg-white mb-4">
                <Title order={4} mb="sm">Mathematical Definition of a Neuron</Title>
                <div className="space-y-4">
                  <div>
                    <Text fw="bold" mb="xs">Basic Neuron Operation:</Text>
                    <BlockMath>{`y = \\sigma(w^T x + b)`}</BlockMath>
                    <List>
                      <List.Item><InlineMath>{`x \\in \\mathbb{R}^d`}</InlineMath> - Input vector</List.Item>
                      <List.Item><InlineMath>{`w \\in \\mathbb{R}^d`}</InlineMath> - Weight vector</List.Item>
                      <List.Item><InlineMath>{`b \\in \\mathbb{R}`}</InlineMath> - Bias scalar</List.Item>
                      <List.Item><InlineMath>{`\\sigma`}</InlineMath> - Activation function</List.Item>
                    </List>
                  </div>
                  
                  <div>
                    <Text fw="bold" mb="xs">Two-Step Process:</Text>
                    <List>
                      <List.Item><strong>Linear Transformation:</strong> <InlineMath>{`z = w^T x + b`}</InlineMath></List.Item>
                      <List.Item><strong>Activation:</strong> <InlineMath>{`y = \\sigma(z)`}</InlineMath></List.Item>
                    </List>
                  </div>
                </div>
              </Paper>

              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-yellow-50">
                    <Title order={4} mb="sm">Neuron with Linear Activation</Title>
                    <Text size="sm" className="mb-3">
                      When <InlineMath>{`\\sigma(z) = z`}</InlineMath> (identity function):
                    </Text>
                    <BlockMath>{`y = w^T x + b`}</BlockMath>
                    <Text size="sm" className="mt-2">
                      <strong>This is exactly linear regression!</strong> The neuron learns the same linear 
                      relationship between inputs and outputs.
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-green-50">
                    <Title order={4} mb="sm">Neuron with Sigmoid Activation</Title>
                    <Text size="sm" className="mb-3">
                      When <InlineMath>{`\\sigma(z) = \\frac{1}{1 + e^{-z}}`}</InlineMath>:
                    </Text>
                    <BlockMath>{`y = \\frac{1}{1 + e^{-(w^T x + b)}}`}</BlockMath>
                    <Text size="sm" className="mt-2">
                      <strong>This is exactly logistic regression!</strong> The neuron outputs a probability 
                      between 0 and 1 for binary classification.
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

          
        </div>

        {/* Network Architecture */}
        <div data-slide>
          
            <Title order={2} mb="xl" id="network-architecture">
              Network Architecture
            </Title>
                                <Flex direction="column" align="center" mb="md">
                                  <Image
                                    src="/assets/python-deep-learning/module1/mlp.png"
                                    alt="Computer Vision Applications"
                                    style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                                    fluid
                                  />
                                </Flex>
                                                  <Text component="p" ta="center" mt="xs">
                                    Source: https://aimresearch.co/
                                  </Text>
            <Paper className="p-6 bg-blue-50 mb-6">
              <Title order={3} mb="md">From Single Neuron to Multi-Layer Networks</Title>
              <Text size="lg" mb="md">
                A Multi-Layer Perceptron (MLP) stacks multiple layers of neurons, where each layer transforms 
                the input through the same mathematical operations we've seen in traditional ML models.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">Single Layer (Linear Model)</Title>
                    <BlockMath>{`y = W x + b`}</BlockMath>
                    <Text size="sm" className="mb-2">
                      Where <InlineMath>{`W \\in \\mathbb{R}^{m \\times n}`}</InlineMath>, <InlineMath>{`x \\in \\mathbb{R}^n`}</InlineMath>
                    </Text>
                    <Text size="sm" color="dimmed">
                      Limited to linear decision boundaries - same as traditional linear regression/classification
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">Multi-Layer Perceptron</Title>
                    <BlockMath>{`h_1 = \\sigma_1(W_1 x + b_1)`}</BlockMath>
                    <BlockMath>{`h_2 = \\sigma_2(W_2 h_1 + b_2)`}</BlockMath>
                    <BlockMath>{`y = W_3 h_2 + b_3`}</BlockMath>
                    <Text size="sm" color="dimmed">
                      Composition of transformations enables complex non-linear relationships
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Why Multi-Layer? */}
            <Paper className="p-6 bg-green-50 mb-6">
              <Title order={3} mb="md">Why Multiple Layers?</Title>
              Research has demonstrated that a Multi-Layer Perceptron (MLP) with just one hidden layer can model even the most complex functions, provided it has enough neurons. This finding might suggest that deeper networks are unnecessary.

However, the real power of neural networks lies in their depth. While a single hidden layer can theoretically model any function, deeper networks (those with more than one hidden layer) can do so much more efficiently. They can model complex functions with exponentially fewer neurons than shallow networks, leading to better performance with the same amount of training data
              </Paper>
        </div>

            {/* Activation Functions */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} mb="md">Common Activation Functions</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-blue-50 h-full">
                    <Title order={4} mb="sm">Linear (Identity)</Title>
                    <BlockMath>{`\\sigma(z) = z`}</BlockMath>
                    <Text size="sm" className="mb-2">Range: <InlineMath>{`(-\\infty, +\\infty)`}</InlineMath></Text>
                    <Text size="xs">Used in: Regression output layers</Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-green-50 h-full">
                    <Title order={4} mb="sm">Sigmoid</Title>
                    <BlockMath>{`\\sigma(z) = \\frac{1}{1 + e^{-z}}`}</BlockMath>
                    <Text size="sm" className="mb-2">Range: <InlineMath>{`(0, 1)`}</InlineMath></Text>
                    <Text size="xs">Used in: Binary classification, gates in LSTM</Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-purple-50 h-full">
                    <Title order={4} mb="sm">ReLU</Title>
                    <BlockMath>{`\\sigma(z) = \\max(0, z)`}</BlockMath>
                    <Text size="sm" className="mb-2">Range: <InlineMath>{`[0, +\\infty)`}</InlineMath></Text>
                    <Text size="xs">Used in: Hidden layers (most popular)</Text>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper p="md" bg="white" mt="md">
                <Title order={4} mb="sm">Why Activation Functions?</Title>
                <Text size="sm" className="mb-3">
                  Without activation functions, multiple layers would collapse to a single linear transformation:
                </Text>
                <BlockMath>{`f(x) = W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W' x + b'`}</BlockMath>
                <Text size="sm" className="mt-2">
                  Activation functions introduce non-linearity, enabling complex pattern recognition.
                </Text>
              </Paper>
            </Paper>
        {/* Parameters to Optimize */}
        <div data-slide>
          
            <Title order={2} mb="xl" id="parameters">
              Parameters to Optimize
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-orange-50 to-red-50 mb-6">
              <Title order={3} mb="md">Understanding the Parameter Space</Title>
              <Text size="lg" mb="md">
                Neural networks learn by optimizing millions of parameters. Understanding what these parameters 
                represent and how they scale is crucial for designing and training networks effectively.
              </Text>
              
              <Paper className="p-4 bg-white">
                <Title order={4} mb="sm">Complete Parameter Set</Title>
                <Text size="sm" className="mb-3">
                  For an L-layer MLP with layer sizes <InlineMath>{`[n_0, n_1, n_2, ..., n_L]`}</InlineMath>:
                </Text>
                <BlockMath>{`\\theta = \\{W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, ..., W^{(L)}, b^{(L)}\\}`}</BlockMath>
                <Grid gutter="lg" className="mt-4">
                  <Grid.Col span={6}>
                    <Text size="sm"><InlineMath>{`W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}`}</InlineMath> - Weight matrix for layer l</Text>
                  </Grid.Col>
                  <Grid.Col span={6}>
                    <Text size="sm"><InlineMath>{`b^{(l)} \\in \\mathbb{R}^{n_l}`}</InlineMath> - Bias vector for layer l</Text>
                  </Grid.Col>
                </Grid>
              </Paper>
            </Paper>

            {/* Parameter Counting */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} mb="md">Parameter Count Analysis</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-blue-50">
                    <Title order={4} mb="sm">Per-Layer Parameters</Title>
                    <Text size="sm" className="mb-3">
                      For a layer with <InlineMath>{`n_{in}`}</InlineMath> inputs and <InlineMath>{`n_{out}`}</InlineMath> outputs:
                    </Text>
                    <BlockMath>{`\text{Parameters} = n_{in} \\times n_{out} + n_{out}`}</BlockMath>
                    <List size="sm">
                      <List.Item><InlineMath>{`n_{in} \\times n_{out}`}</InlineMath> weights in matrix <InlineMath>{`W`}</InlineMath></List.Item>
                      <List.Item><InlineMath>{`n_{out}`}</InlineMath> biases in vector <InlineMath>{`b`}</InlineMath></List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-green-50">
                    <Title order={4} mb="sm">Total Network Parameters</Title>
                    <BlockMath>{`\text{Total} = \\sum_{l=1}^L (n_{l-1} \\times n_l + n_l)`}</BlockMath>
                    <Text size="sm" className="mb-2">
                      Example: 784 → 128 → 64 → 10
                    </Text>
                    <List size="sm">
                      <List.Item>Layer 1: (784 × 128) + 128 = 100,480</List.Item>
                      <List.Item>Layer 2: (128 × 64) + 64 = 8,256</List.Item>
                      <List.Item>Layer 3: (64 × 10) + 10 = 650</List.Item>
                      <List.Item><strong>Total: 109,386 parameters</strong></List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
              
            </Paper>

          
        </div>


      </Stack>
    </Container>
  );
};

export default MLPFundamentals;