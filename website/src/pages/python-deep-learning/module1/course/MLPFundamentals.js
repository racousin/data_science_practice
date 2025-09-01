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
          
            <Title order={2} className="mb-6" id="neuron">
              Neuron as Computational Unit
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 mb-6">
              <Title order={3} className="mb-4">From Biology to Mathematics</Title>
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
                      <List.Item><InlineMath>{`x \\in \\mathbb{R}^n`}</InlineMath> - Input vector</List.Item>
                      <List.Item><InlineMath>{`w \\in \\mathbb{R}^n`}</InlineMath> - Weight vector</List.Item>
                      <List.Item><InlineMath>{`b \\in \\mathbb{R}`}</InlineMath> - Bias scalar</List.Item>
                      <List.Item><InlineMath>{`\\sigma`}</InlineMath> - Activation function</List.Item>
                    </List>
                  </div>
                  
                  <div>
                    <Text fw="bold" mb="xs">Two-Step Process:</Text>
                    <List>
                      <List.Item><strong>Linear Transformation:</strong> <InlineMath>{`z = w^T x + b`}</InlineMath></List.Item>
                      <List.Item><strong>Non-linear Activation:</strong> <InlineMath>{`y = \\sigma(z)`}</InlineMath></List.Item>
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

            {/* Activation Functions */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Common Activation Functions</Title>
              
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
          
        </div>

        {/* Network Architecture */}
        <div data-slide>
          
            <Title order={2} className="mb-6" id="network-architecture">
              Network Architecture
            </Title>
            
            <Paper className="p-6 bg-blue-50 mb-6">
              <Title order={3} className="mb-4">From Single Neuron to Multi-Layer Networks</Title>
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
              <Title order={3} className="mb-4">Why Multiple Layers?</Title>
              
              <Paper className="p-4 bg-white mb-4">
                <Title order={4} mb="sm">The Limitation of Linear Models</Title>
                <Text size="sm" className="mb-3">
                  Traditional machine learning models are fundamentally limited by their linearity:
                </Text>
                <List spacing="sm">
                  <List.Item><strong>Linear Regression:</strong> Can only fit straight lines/planes</List.Item>
                  <List.Item><strong>Logistic Regression:</strong> Can only create linear decision boundaries</List.Item>
                  <List.Item><strong>SVM (linear):</strong> Finds optimal linear separating hyperplane</List.Item>
                </List>
              </Paper>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-yellow-50">
                    <Title order={4} mb="sm">Linear Decision Boundary</Title>
                    <Text size="sm" className="mb-3">
                      Single layer can only separate data with a straight line:
                    </Text>
                    <BlockMath>{`w_1 x_1 + w_2 x_2 + b = 0`}</BlockMath>
                    <Text size="sm" className="mt-2">
                      ❌ Cannot solve XOR problem<br/>
                      ❌ Cannot handle complex patterns<br/>
                      ❌ Limited expressivity
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-blue-50">
                    <Title order={4} mb="sm">Non-Linear Decision Boundary</Title>
                    <Text size="sm" className="mb-3">
                      Multiple layers can create complex decision boundaries:
                    </Text>
                    <Text size="sm" className="mt-2">
                      ✅ Can solve XOR problem<br/>
                      ✅ Can approximate any continuous function<br/>
                      ✅ Can learn hierarchical features
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper p="md" bg="white" mt="md">
                <Title order={4} mb="sm">Mathematical Intuition</Title>
                <Text size="sm" className="mb-3">
                  Each layer performs a geometric transformation of the input space:
                </Text>
                <List size="sm">
                  <List.Item><strong>Layer 1:</strong> <InlineMath>{`\\mathbb{R}^n \\rightarrow \\mathbb{R}^{h_1}`}</InlineMath> - Projects input to hidden space</List.Item>
                  <List.Item><strong>Layer 2:</strong> <InlineMath>{`\\mathbb{R}^{h_1} \\rightarrow \\mathbb{R}^{h_2}`}</InlineMath> - Further transforms representation</List.Item>
                  <List.Item><strong>Output:</strong> <InlineMath>{`\\mathbb{R}^{h_2} \\rightarrow \\mathbb{R}^m`}</InlineMath> - Maps to desired output dimension</List.Item>
                </List>
              </Paper>
            </Paper>

            {/* Non-Linear Problem Examples */}
            <Paper className="p-6 bg-purple-50 mb-6">
              <Title order={3} className="mb-4">Examples of Non-Linear Problems</Title>
              
              <Text className="mb-4">
                Multi-layer networks excel at problems that single-layer models cannot solve:
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">XOR Problem</Title>
                    <Text size="sm" className="mb-3">
                      Classic example of non-linear separability:
                    </Text>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ backgroundColor: '#f8f9fa' }}>
                            <th style={{ border: '1px solid #dee2e6', padding: '8px' }}><InlineMath>{`x_1`}</InlineMath></th>
                            <th style={{ border: '1px solid #dee2e6', padding: '8px' }}><InlineMath>{`x_2`}</InlineMath></th>
                            <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>XOR</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0</td>
                          </tr>
                          <tr>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1</td>
                          </tr>
                          <tr>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1</td>
                          </tr>
                          <tr>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1</td>
                            <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <Text size="sm" className="mt-2">
                      No single line can separate 0s from 1s!
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">Concentric Circles</Title>
                    <Text size="sm" className="mb-3">
                      Points inside inner circle vs. points in outer ring:
                    </Text>
                    <BlockMath>{`\text{Class} = \begin{cases} 
                      1 & \text{if } x_1^2 + x_2^2 < r_1^2 \\
                      0 & \text{if } r_1^2 < x_1^2 + x_2^2 < r_2^2
                    \end{cases}`}</BlockMath>
                    <Text size="sm" className="mt-2">
                      Requires circular decision boundary, impossible with linear models.
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper p="md" bg="white" mt="md">
                <Title order={4} mb="sm">Real-World Complex Patterns</Title>
                
                <Flex direction="column" align="center" mb="md">
                  <Image
                    src="/assets/python-deep-learning/module1/signal_processing.png"
                    alt="Signal Processing Applications"
                    w={{ base: 400, sm: 600, md: 800 }}
                    h="auto"
                    fluid
                  />
                  <Text component="p" ta="center" mt="xs">
                    Source: https://www.edgeimpulse.com/blog/dsp-key-embedded-ml/
                  </Text>
                </Flex>
                
                <Text size="sm" className="mb-3">
                  These complex patterns require multiple layers to:
                </Text>
                <List size="sm">
                  <List.Item><strong>Image Recognition:</strong> Detect edges → shapes → objects</List.Item>
                  <List.Item><strong>Speech Processing:</strong> Phonemes → words → meaning</List.Item>
                  <List.Item><strong>Time Series:</strong> Local patterns → seasonal trends → long-term dependencies</List.Item>
                  <List.Item><strong>Natural Language:</strong> Characters → words → syntax → semantics</List.Item>
                </List>
              </Paper>
            </Paper>
          
        </div>

        {/* Parameters to Optimize */}
        <div data-slide>
          
            <Title order={2} className="mb-6" id="parameters">
              Parameters to Optimize
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-orange-50 to-red-50 mb-6">
              <Title order={3} className="mb-4">Understanding the Parameter Space</Title>
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
              <Title order={3} className="mb-4">Parameter Count Analysis</Title>
              
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
              
              <Paper className="p-4 bg-yellow-50 mt-4">
                <Title order={4} mb="sm">Parameter Scaling</Title>
                <Text size="sm" className="mb-3">
                  Parameter count grows quadratically with layer width:
                </Text>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#fff' }}>
                        <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>Hidden Size</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>Parameters (MNIST)</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>Memory (FP32)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>128</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>109,386</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0.44 MB</td>
                      </tr>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>512</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>669,706</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>2.68 MB</td>
                      </tr>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1024</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1,590,282</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>6.36 MB</td>
                      </tr>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>2048</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>3,670,026</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>14.68 MB</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Paper>
            </Paper>

            {/* Learning Process */}
            <Paper className="p-6 bg-indigo-50 mb-6">
              <Title order={3} className="mb-4">The Learning Process</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">Forward Pass</Title>
                    <Text size="sm" className="mb-3">
                      Compute predictions by propagating inputs through layers:
                    </Text>
                    <BlockMath>{`a^{(0)} = x`}</BlockMath>
                    <BlockMath>{`z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}`}</BlockMath>
                    <BlockMath>{`a^{(l)} = \\sigma(z^{(l)})`}</BlockMath>
                    <BlockMath>{`\\hat{y} = a^{(L)}`}</BlockMath>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">Parameter Updates</Title>
                    <Text size="sm" className="mb-3">
                      Use gradient descent to minimize loss:
                    </Text>
                    <BlockMath>{`\\mathcal{L} = \\frac{1}{n} \\sum_{i=1}^n \\ell(\\hat{y}_i, y_i)`}</BlockMath>
                    <BlockMath>{`W^{(l)} \\leftarrow W^{(l)} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}}`}</BlockMath>
                    <BlockMath>{`b^{(l)} \\leftarrow b^{(l)} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial b^{(l)}}`}</BlockMath>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper p="md" bg="white" mt="md">
                <Title order={4} mb="sm">Key Insight: Gradient-Based Optimization</Title>
                <Text size="sm" className="mb-3">
                  MLPs use the same gradient descent principles from traditional ML:
                </Text>
                <List size="sm">
                  <List.Item>Same optimization objective: minimize loss function</List.Item>
                  <List.Item>Same update rule: parameters ← parameters - learning_rate × gradient</List.Item>
                  <List.Item>Difference: gradients computed via backpropagation (chain rule)</List.Item>
                  <List.Item>Backpropagation will be covered in detail in Module 2</List.Item>
                </List>
              </Paper>
            </Paper>
          
        </div>

        {/* Summary */}
        <div data-slide>
          
            <Title order={2} className="mb-6">Part 3 Summary: Multi Layer Perceptron in a nutshell</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                  <Title order={3} className="mb-4">Core Architectural Concepts</Title>
                  <List spacing="md">
                    <List.Item>Neurons are mathematical units: <InlineMath>{`y = \\sigma(w^T x + b)`}</InlineMath></List.Item>
                    <List.Item>Linear activation = linear regression</List.Item>
                    <List.Item>Sigmoid activation = logistic regression</List.Item>
                    <List.Item>Multiple layers enable non-linear pattern recognition</List.Item>
                    <List.Item>Each layer performs geometric transformation of input space</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                  <Title order={3} className="mb-4">Key Learning Insights</Title>
                  <List spacing="md">
                    <List.Item>MLPs solve problems linear models cannot</List.Item>
                    <List.Item>Parameter count grows quadratically with layer width</List.Item>
                    <List.Item>Same gradient-based optimization as traditional ML</List.Item>
                    <List.Item>Activation functions provide essential non-linearity</List.Item>
                    <List.Item>Forward pass + gradient descent = learning</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
            
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6">
              <Title order={3} className="mb-4 text-center">Neural Networks are Powerful Function Approximators</Title>
              <Text size="lg" className="text-center">
                MLPs extend traditional machine learning by stacking simple mathematical operations to create 
                complex, non-linear functions. The same optimization principles apply—we just have many more 
                parameters to learn through gradient descent.
              </Text>
            </Paper>
          
        </div>

      </Stack>
    </Container>
  );
};

export default MLPFundamentals;