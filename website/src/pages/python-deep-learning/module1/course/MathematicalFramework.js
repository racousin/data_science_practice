import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const MathematicalFramework = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
        {/* Part 2: Mathematical Framework */}
        <div data-slide>
          <Title order={1} mb="xl">
            Part 2: Mathematical Framework
          </Title>
          
          {/* Machine Learning Objective */}
          
            <Title order={2} className="mb-6" id="ml-objective">
              The Machine Learning Objective
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 mb-6">
              <Title order={3} className="mb-4">The Fundamental Learning Problem</Title>
              <Text size="lg" className="mb-4">
                At its core, machine learning seeks to find a function <InlineMath>f</InlineMath> that maps inputs to outputs 
                by learning from data. Deep learning extends this by using highly expressive neural network functions.
              </Text>
              
              <Paper className="p-4 bg-white mb-4">
                <Title order={4} className="mb-3">The Learning Framework</Title>
                <div className="space-y-4">
                  <div>
                    <Text className="font-semibold mb-2">Given:</Text>
                    <List>
                      <List.Item>Training dataset: <InlineMath>{`\\mathcal{D} = \\{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\\}`}</InlineMath></List.Item>
                      <List.Item>Input space: <InlineMath>{`x_i \\in \\mathcal{X} \\subseteq \\mathbb{R}^d`}</InlineMath></List.Item>
                      <List.Item>Output space: <InlineMath>{`y_i \\in \\mathcal{Y}`}</InlineMath> (continuous for regression, discrete for classification)</List.Item>
                    </List>
                  </div>
                  
                  <div>
                    <Text className="font-semibold mb-2">Objective:</Text>
                    <Text>Find a function <InlineMath>{`f: \\mathcal{X} \\rightarrow \\mathcal{Y}`}</InlineMath> that:</Text>
                    <List>
                      <List.Item>Minimizes error on training data (empirical risk)</List.Item>
                      <List.Item>Generalizes well to unseen data (expected risk)</List.Item>
                    </List>
                  </div>
                  
                  <div>
                    <Text className="font-semibold mb-2">Mathematical Formulation:</Text>
                    <BlockMath>{`\\min_{f \\in \\mathcal{F}} \\mathbb{E}_{(x,y) \\sim P_{data}}[\\ell(f(x), y)]`}</BlockMath>
                    <Text size="sm" color="dimmed">
                      where <InlineMath>{`\\mathcal{F}`}</InlineMath> is the hypothesis class (e.g., neural networks), 
                      and <InlineMath>{`\\ell`}</InlineMath> is the loss function
                    </Text>
                  </div>
                </div>
              </Paper>

              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-yellow-50">
                    <Title order={4} className="mb-3">Empirical Risk Minimization</Title>
                    <Text size="sm" className="mb-3">
                      Since we don't know the true data distribution <InlineMath>{`P_{data}`}</InlineMath>, 
                      we approximate with the empirical risk:
                    </Text>
                    <BlockMath>{`\\hat{R}(f) = \\frac{1}{n} \\sum_{i=1}^n \\ell(f(x_i), y_i)`}</BlockMath>
                    <Text size="sm" className="mt-2">
                      This is what we actually optimize in practice using gradient descent.
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-green-50">
                    <Title order={4} className="mb-3">The Bias-Variance Tradeoff</Title>
                    <Text size="sm" className="mb-3">
                      Expected prediction error decomposes into:
                    </Text>
                    <BlockMath>{`\\mathbb{E}[(y - f(x))^2] = \\text{Bias}^2 + \\text{Variance} + \\text{Noise}`}</BlockMath>
                    <List size="sm">
                      <List.Item><strong>Bias:</strong> Error from wrong assumptions</List.Item>
                      <List.Item><strong>Variance:</strong> Error from sensitivity to data</List.Item>
                      <List.Item><strong>Noise:</strong> Irreducible error</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Types of Learning Problems */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Types of Learning Problems</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-blue-50 h-full">
                    <Title order={4} className="mb-3">Supervised Learning</Title>
                    <Text size="sm" className="mb-3">Learning from labeled pairs <InlineMath>{`(x, y)`}</InlineMath></Text>
                    
                    <div className="mb-3">
                      <Text className="font-semibold text-sm">Classification:</Text>
                      <BlockMath>{`y \\in \\{1, 2, ..., K\\}`}</BlockMath>
                      <Text size="xs">Examples: Image recognition, spam detection</Text>
                    </div>
                    
                    <div>
                      <Text className="font-semibold text-sm">Regression:</Text>
                      <BlockMath>{`y \\in \\mathbb{R}^m`}</BlockMath>
                      <Text size="xs">Examples: Price prediction, weather forecasting</Text>
                    </div>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-green-50 h-full">
                    <Title order={4} className="mb-3">Unsupervised Learning</Title>
                    <Text size="sm" className="mb-3">Learning structure from unlabeled data <InlineMath>{`x`}</InlineMath></Text>
                    
                    <List size="sm">
                      <List.Item><strong>Clustering:</strong> Group similar data points</List.Item>
                      <List.Item><strong>Dimensionality Reduction:</strong> Find low-dimensional representations</List.Item>
                      <List.Item><strong>Density Estimation:</strong> Model data distribution</List.Item>
                      <List.Item><strong>Generation:</strong> Sample new data points</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-purple-50 h-full">
                    <Title order={4} className="mb-3">Reinforcement Learning</Title>
                    <Text size="sm" className="mb-3">Learning through interaction and rewards</Text>
                    
                    <Text size="sm" className="mb-2">Maximize expected return:</Text>
                    <BlockMath>{`G_t = \\sum_{k=0}^\\infty \\gamma^k R_{t+k+1}`}</BlockMath>
                    <Text size="xs">
                      Agent learns policy <InlineMath>{`\\pi(a|s)`}</InlineMath> through 
                      trial and error
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>
          
        </div>

        {/* Models and Parameters */}
        <div data-slide>
          
            <Title order={2} className="mb-6" id="models-parameters">
              Models and Parameters
            </Title>
            
            <Paper className="p-6 bg-blue-50 mb-6">
              <Title order={3} className="mb-4">Parametric Models</Title>
              <Text size="lg" className="mb-4">
                Deep learning models are parametric: they learn a fixed set of parameters <InlineMath>{`\\theta`}</InlineMath> 
                from training data. The model function becomes <InlineMath>{`f_\\theta(x)`}</InlineMath>.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Linear Models (Baseline)</Title>
                    <Text size="sm" className="mb-3">The simplest parametric model:</Text>
                    <BlockMath>{`f_\\theta(x) = w^T x + b`}</BlockMath>
                    <Text size="sm" className="mb-2">Parameters: <InlineMath>{`\\theta = \\{w \\in \\mathbb{R}^d, b \\in \\mathbb{R}\\}`}</InlineMath></Text>
                    <Text size="sm" color="dimmed">
                      Limited to linear decision boundaries, cannot capture complex patterns
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Neural Networks</Title>
                    <Text size="sm" className="mb-3">Composition of linear and non-linear functions:</Text>
                    <BlockMath>{`f_\\theta(x) = W_L \\cdot \\sigma(W_{L-1} \\cdot ... \\cdot \\sigma(W_1 x + b_1) ... + b_{L-1}) + b_L`}</BlockMath>
                    <Text size="sm" className="mb-2">Parameters: <InlineMath>{`\\theta = \\{W_1, b_1, ..., W_L, b_L\\}`}</InlineMath></Text>
                    <Text size="sm" color="dimmed">
                      Can approximate any continuous function (universal approximation)
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Parameter Counting */}
            <Paper className="p-6 bg-green-50 mb-6">
              <Title order={3} className="mb-4">Understanding Parameter Counts</Title>
              
              <Text className="mb-4">
                The number of parameters determines model capacity and memory requirements:
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Fully Connected Layer</Title>
                    <Text size="sm" className="mb-3">
                      Connecting <InlineMath>{`n_{in}`}</InlineMath> inputs to <InlineMath>{`n_{out}`}</InlineMath> outputs:
                    </Text>
                    <BlockMath>{`\\text{Parameters} = n_{in} \\times n_{out} + n_{out}`}</BlockMath>
                    <Text size="xs" color="dimmed">Weights matrix + bias vector</Text>
                    
                    <CodeBlock language="python" code={`# PyTorch Example
import torch.nn as nn

layer = nn.Linear(in_features=784, out_features=128)
# Parameters: 784 × 128 + 128 = 100,480`} />
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Convolutional Layer</Title>
                    <Text size="sm" className="mb-3">
                      Filters with weight sharing:
                    </Text>
                    <BlockMath>{`\\text{Parameters} = (k_h \\times k_w \\times c_{in} + 1) \\times c_{out}`}</BlockMath>
                    <Text size="xs" color="dimmed">kernel size × input channels × output channels + biases</Text>
                    
                    <CodeBlock language="python" code={`# PyTorch Example
conv = nn.Conv2d(in_channels=3, out_channels=64, 
                  kernel_size=3)
# Parameters: (3 × 3 × 3 + 1) × 64 = 1,792`} />
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper className="p-4 bg-yellow-50 mt-4">
                <Title order={4} className="mb-3">Modern Model Scales</Title>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#fff' }}>
                        <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>Model</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>Parameters</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>Memory (FP32)</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '8px' }}>Year</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>LeNet-5</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>60K</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>0.24 MB</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>1998</td>
                      </tr>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>AlexNet</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>61M</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>244 MB</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>2012</td>
                      </tr>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>ResNet-50</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>25.6M</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>102 MB</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>2015</td>
                      </tr>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>GPT-3</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>175B</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>700 GB</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>2020</td>
                      </tr>
                      <tr>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>GPT-4</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>~1.76T</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>~7 TB</td>
                        <td style={{ border: '1px solid #dee2e6', padding: '8px' }}>2023</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Paper>
            </Paper>
          
        </div>

        {/* Loss Functions and Optimization */}
        <div data-slide>
          
            <Title order={2} className="mb-6" id="loss-functions">
              Loss Functions and Optimization Problems
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mb-6">
              <Title order={3} className="mb-4">The Role of Loss Functions</Title>
              <Text size="lg" className="mb-4">
                Loss functions quantify how wrong our predictions are. They transform the learning problem into an 
                optimization problem that can be solved with gradient descent.
              </Text>
              
              <Paper className="p-4 bg-white">
                <Text className="font-semibold mb-3">General Optimization Problem:</Text>
                <BlockMath>{`\\theta^* = \\arg\\min_\\theta \\frac{1}{n} \\sum_{i=1}^n \\ell(f_\\theta(x_i), y_i) + \\lambda \\Omega(\\theta)`}</BlockMath>
                <Grid gutter="lg" className="mt-4">
                  <Grid.Col span={6}>
                    <Text size="sm"><InlineMath>{`\\ell(f_\\theta(x_i), y_i)`}</InlineMath> - Loss on sample i</Text>
                  </Grid.Col>
                  <Grid.Col span={6}>
                    <Text size="sm"><InlineMath>{`\\lambda \\Omega(\\theta)`}</InlineMath> - Regularization term</Text>
                  </Grid.Col>
                </Grid>
              </Paper>
            </Paper>

            {/* Common Loss Functions */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Common Loss Functions</Title>
              
              <Grid gutter="lg">
                {/* Regression Losses */}
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-blue-50">
                    <Title order={4} className="mb-3">Regression Losses</Title>
                    
                    <div className="space-y-4">
                      <div>
                        <Text className="font-semibold text-sm">Mean Squared Error (MSE):</Text>
                        <BlockMath>{`\\ell_{MSE}(y, \\hat{y}) = \\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2`}</BlockMath>
                        <Text size="xs" color="dimmed">Penalizes large errors heavily, sensitive to outliers</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Mean Absolute Error (MAE):</Text>
                        <BlockMath>{`\\ell_{MAE}(y, \\hat{y}) = \\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|`}</BlockMath>
                        <Text size="xs" color="dimmed">Robust to outliers, non-differentiable at zero</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Huber Loss:</Text>
                        <BlockMath>{`\\ell_{Huber}(y, \\hat{y}) = \\begin{cases} 
                          \\frac{1}{2}(y - \\hat{y})^2 & \\text{if } |y - \\hat{y}| \\leq \\delta \\\\
                          \\delta|y - \\hat{y}| - \\frac{1}{2}\\delta^2 & \\text{otherwise}
                        \\end{cases}`}</BlockMath>
                        <Text size="xs" color="dimmed">Combines MSE and MAE benefits</Text>
                      </div>
                    </div>
                  </Paper>
                </Grid.Col>
                
                {/* Classification Losses */}
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-green-50">
                    <Title order={4} className="mb-3">Classification Losses</Title>
                    
                    <div className="space-y-4">
                      <div>
                        <Text className="font-semibold text-sm">Cross-Entropy Loss:</Text>
                        <BlockMath>{`\\ell_{CE}(y, \\hat{p}) = -\\sum_{i=1}^n \\sum_{c=1}^C y_{ic} \\log(\\hat{p}_{ic})`}</BlockMath>
                        <Text size="xs" color="dimmed">Standard for multi-class classification</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Binary Cross-Entropy:</Text>
                        <BlockMath>{`\\ell_{BCE}(y, \\hat{p}) = -[y\\log(\\hat{p}) + (1-y)\\log(1-\\hat{p})]`}</BlockMath>
                        <Text size="xs" color="dimmed">For binary classification problems</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Focal Loss:</Text>
                        <BlockMath>{`\\ell_{FL}(p_t) = -\\alpha_t(1-p_t)^\\gamma \\log(p_t)`}</BlockMath>
                        <Text size="xs" color="dimmed">Addresses class imbalance by focusing on hard examples</Text>
                      </div>
                    </div>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <CodeBlock language="python" code={`import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Loss Functions
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
cross_entropy = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()

# Custom Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()`} />
            </Paper>

            {/* Regularization */}
            <Paper className="p-6 bg-yellow-50 mb-6">
              <Title order={3} className="mb-4">Regularization Techniques</Title>
              <Text className="mb-4">
                Regularization prevents overfitting by adding constraints or penalties to the optimization problem:
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Weight Decay</Title>
                    <BlockMath>{`\\Omega(\\theta) = \\frac{1}{2}||\\theta||_2^2`}</BlockMath>
                    <Text size="sm">L2 penalty on weights, encourages small parameters</Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Dropout</Title>
                    <BlockMath>{`h_i = \\begin{cases} 0 & \\text{with prob } p \\\\ \\frac{h_i}{1-p} & \\text{otherwise} \\end{cases}`}</BlockMath>
                    <Text size="sm">Randomly deactivates neurons during training</Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Early Stopping</Title>
                    <Text size="sm" className="mb-2">Monitor validation loss and stop when it increases</Text>
                    <Text size="xs" color="dimmed">Implicit regularization through optimization</Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>
          
        </div>

        {/* Gradient Descent */}
        <div data-slide>
          
            <Title order={2} className="mb-6" id="gradient-descent">
              Gradient Descent and Optimization
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-blue-50 to-cyan-50 mb-6">
              <Title order={3} className="mb-4">The Gradient Descent Algorithm</Title>
              <Text size="lg" className="mb-4">
                Gradient descent is the workhorse of deep learning optimization. It iteratively updates parameters 
                in the direction that reduces the loss function.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Basic Update Rule</Title>
                    <BlockMath>{`\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}(\\theta_t)`}</BlockMath>
                    <List size="sm" className="mt-3">
                      <List.Item><InlineMath>{`\\eta`}</InlineMath> - Learning rate (step size)</List.Item>
                      <List.Item><InlineMath>{`\\nabla_\\theta \\mathcal{L}`}</InlineMath> - Gradient of loss w.r.t. parameters</List.Item>
                      <List.Item>Direction of steepest decrease in loss</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Intuition</Title>
                    <Text size="sm" className="mb-3">
                      Imagine minimizing loss as finding the lowest point in a landscape:
                    </Text>
                    <List size="sm">
                      <List.Item>Gradient points uphill (direction of increase)</List.Item>
                      <List.Item>Negative gradient points downhill</List.Item>
                      <List.Item>Learning rate controls step size</List.Item>
                      <List.Item>Stop when gradient ≈ 0 (local minimum)</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Variants of Gradient Descent */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Gradient Descent Variants</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-blue-50 h-full">
                    <Title order={4} className="mb-3">Batch Gradient Descent</Title>
                    <BlockMath>{`\\nabla_\\theta \\mathcal{L} = \\frac{1}{n}\\sum_{i=1}^n \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)`}</BlockMath>
                    <List size="sm" className="mt-3">
                      <List.Item>Uses entire dataset per update</List.Item>
                      <List.Item>Stable convergence</List.Item>
                      <List.Item>Slow for large datasets</List.Item>
                      <List.Item>Can converge to sharp minima</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-green-50 h-full">
                    <Title order={4} className="mb-3">Stochastic GD (SGD)</Title>
                    <BlockMath>{`\\nabla_\\theta \\mathcal{L} \\approx \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)`}</BlockMath>
                    <List size="sm" className="mt-3">
                      <List.Item>One sample per update</List.Item>
                      <List.Item>Very noisy updates</List.Item>
                      <List.Item>Fast iterations</List.Item>
                      <List.Item>Can escape local minima</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-yellow-50 h-full">
                    <Title order={4} className="mb-3">Mini-batch SGD</Title>
                    <BlockMath>{`\\nabla_\\theta \\mathcal{L} \\approx \\frac{1}{m}\\sum_{i \\in \\mathcal{B}} \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)`}</BlockMath>
                    <List size="sm" className="mt-3">
                      <List.Item>Batch size m (typically 32-512)</List.Item>
                      <List.Item>Balance of speed and stability</List.Item>
                      <List.Item>GPU parallelization efficient</List.Item>
                      <List.Item>Standard in practice</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Advanced Optimizers */}
            <Paper className="p-6 bg-purple-50 mb-6">
              <Title order={3} className="mb-4">Modern Optimization Algorithms</Title>
              
              <Text className="mb-4">
                Simple SGD often struggles with complex loss landscapes. Modern optimizers add momentum and adaptive learning rates:
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Momentum SGD</Title>
                    <BlockMath>{`v_{t+1} = \\beta v_t + (1-\\beta) \\nabla_\\theta \\mathcal{L}`}</BlockMath>
                    <BlockMath>{`\\theta_{t+1} = \\theta_t - \\eta v_{t+1}`}</BlockMath>
                    <Text size="sm" className="mt-2">
                      Accumulates gradient history, accelerates convergence in consistent directions
                    </Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Adam (Adaptive Moment Estimation)</Title>
                    <BlockMath>{`m_t = \\beta_1 m_{t-1} + (1-\\beta_1) \\nabla_\\theta \\mathcal{L}`}</BlockMath>
                    <BlockMath>{`v_t = \\beta_2 v_{t-1} + (1-\\beta_2) (\\nabla_\\theta \\mathcal{L})^2`}</BlockMath>
                    <BlockMath>{`\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{v_t} + \\epsilon} m_t`}</BlockMath>
                    <Text size="sm" className="mt-2">
                      Combines momentum with per-parameter adaptive learning rates
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <CodeBlock language="python" code={`import torch.optim as optim

# PyTorch Optimizers
model = MyNeuralNetwork()

# Basic SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# SGD with Momentum
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (most popular)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters`} />
            </Paper>

            {/* Learning Rate Schedules */}
            <Paper className="p-6 bg-orange-50 mb-6">
              <Title order={3} className="mb-4">Learning Rate Scheduling</Title>
              
              <Text className="mb-4">
                The learning rate is crucial for convergence. Too high causes divergence, too low causes slow convergence:
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Step Decay</Title>
                    <BlockMath>{`\\eta_t = \\eta_0 \\cdot \\gamma^{\\lfloor t/s \\rfloor}`}</BlockMath>
                    <Text size="sm">Decrease by factor γ every s epochs</Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Cosine Annealing</Title>
                    <BlockMath>{`\\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})(1 + \\cos(\\frac{t\\pi}{T}))`}</BlockMath>
                    <Text size="sm">Smooth cosine decay over T epochs</Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Warmup</Title>
                    <BlockMath>{`\\eta_t = \\begin{cases} \\frac{t}{T_{warmup}} \\eta_0 & t < T_{warmup} \\\\ \\eta_0 & \\text{otherwise} \\end{cases}`}</BlockMath>
                    <Text size="sm">Gradual increase to prevent instability</Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>
          
        </div>

        {/* Linear Algebra Concepts */}
        <div data-slide>
          
            <Title order={2} className="mb-6" id="linear-algebra">
              Essential Linear Algebra Concepts
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-green-50 to-teal-50 mb-6">
              <Title order={3} className="mb-4">Vectors, Matrices, and Tensors</Title>
              <Text size="lg" className="mb-4">
                Deep learning operates on multi-dimensional arrays. Understanding their properties and operations 
                is crucial for implementing and debugging neural networks.
              </Text>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Vectors</Title>
                    <BlockMath>{`x \\in \\mathbb{R}^n`}</BlockMath>
                    <Text size="sm" className="mb-2">1D array of numbers</Text>
                    <CodeBlock language="python" code={`import torch
# Vector of size 5
x = torch.tensor([1, 2, 3, 4, 5])
print(x.shape)  # torch.Size([5])`} />
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Matrices</Title>
                    <BlockMath>{`W \\in \\mathbb{R}^{m \\times n}`}</BlockMath>
                    <Text size="sm" className="mb-2">2D array (rows × columns)</Text>
                    <CodeBlock language="python" code={`# Matrix of size 3×4
W = torch.randn(3, 4)
print(W.shape)  # torch.Size([3, 4])`} />
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Tensors</Title>
                    <BlockMath>{`T \\in \\mathbb{R}^{d_1 \\times d_2 \\times ... \\times d_k}`}</BlockMath>
                    <Text size="sm" className="mb-2">k-dimensional array</Text>
                    <CodeBlock language="python" code={`# 4D tensor (batch, channels, height, width)
T = torch.randn(32, 3, 224, 224)
print(T.shape)  # torch.Size([32, 3, 224, 224])`} />
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Matrix Operations */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} className="mb-4">Fundamental Operations</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-blue-50">
                    <Title order={4} className="mb-3">Matrix Multiplication</Title>
                    <BlockMath>{`C = AB \\text{ where } C_{ij} = \\sum_k A_{ik}B_{kj}`}</BlockMath>
                    <Text size="sm" className="mb-2">Dimensions: <InlineMath>{`(m \\times n) \\cdot (n \\times p) = (m \\times p)`}</InlineMath></Text>
                    
                    <CodeBlock language="python" code={`# Matrix multiplication
A = torch.randn(10, 5)
B = torch.randn(5, 3)
C = torch.matmul(A, B)  # or A @ B
print(C.shape)  # torch.Size([10, 3])

# Batched matrix multiplication
A_batch = torch.randn(32, 10, 5)
B_batch = torch.randn(32, 5, 3)
C_batch = torch.bmm(A_batch, B_batch)
print(C_batch.shape)  # torch.Size([32, 10, 3])`} />
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-green-50">
                    <Title order={4} className="mb-3">Element-wise Operations</Title>
                    <BlockMath>{`C = A \\odot B \\text{ where } C_{ij} = A_{ij} \\cdot B_{ij}`}</BlockMath>
                    <Text size="sm" className="mb-2">Hadamard product (element-wise multiplication)</Text>
                    
                    <CodeBlock language="python" code={`# Element-wise operations
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Element-wise multiplication
C = A * B  # [[5, 12], [21, 32]]

# Element-wise addition
D = A + B  # [[6, 8], [10, 12]]

# Broadcasting (automatic dimension expansion)
v = torch.tensor([1, 2])
E = A + v  # [[2, 4], [4, 6]]`} />
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Grid gutter="lg" className="mt-4">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-yellow-50">
                    <Title order={4} className="mb-3">Dot Product</Title>
                    <BlockMath>{`x \\cdot y = x^T y = \\sum_{i=1}^n x_i y_i`}</BlockMath>
                    <Text size="sm" className="mb-2">Measures similarity between vectors</Text>
                    
                    <CodeBlock language="python" code={`# Dot product
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
dot_product = torch.dot(x, y)  # 32.0

# Cosine similarity
cos_sim = torch.nn.functional.cosine_similarity(
    x.unsqueeze(0), y.unsqueeze(0)
)  # Normalized dot product`} />
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-purple-50">
                    <Title order={4} className="mb-3">Transpose and Reshape</Title>
                    <BlockMath>{`A^T \\text{ where } (A^T)_{ij} = A_{ji}`}</BlockMath>
                    <Text size="sm" className="mb-2">Swapping dimensions</Text>
                    
                    <CodeBlock language="python" code={`# Transpose
A = torch.randn(3, 5)
A_T = A.T  # or A.transpose(0, 1)
print(A_T.shape)  # torch.Size([5, 3])

# Reshape/View
x = torch.randn(2, 3, 4)
y = x.view(6, 4)  # Reshape to 6×4
z = x.reshape(-1)  # Flatten to 1D`} />
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Important Properties */}
            <Paper className="p-6 bg-indigo-50 mb-6">
              <Title order={3} className="mb-4">Key Linear Algebra Properties for Deep Learning</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Norms and Distances</Title>
                    
                    <div className="space-y-3">
                      <div>
                        <Text className="font-semibold text-sm">L2 Norm (Euclidean):</Text>
                        <BlockMath>{`||x||_2 = \\sqrt{\\sum_{i=1}^n x_i^2}`}</BlockMath>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">L1 Norm (Manhattan):</Text>
                        <BlockMath>{`||x||_1 = \\sum_{i=1}^n |x_i|`}</BlockMath>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Frobenius Norm (matrices):</Text>
                        <BlockMath>{`||A||_F = \\sqrt{\\sum_{i,j} A_{ij}^2}`}</BlockMath>
                      </div>
                    </div>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Broadcasting Rules</Title>
                    
                    <Text size="sm" className="mb-3">
                      PyTorch automatically broadcasts tensors for element-wise operations:
                    </Text>
                    
                    <List size="sm">
                      <List.Item>Compare shapes element-wise from right to left</List.Item>
                      <List.Item>Dimensions are compatible if equal or one is 1</List.Item>
                      <List.Item>Missing dimensions treated as 1</List.Item>
                    </List>
                    
                    <CodeBlock language="python" code={`# Broadcasting examples
A = torch.randn(5, 3)     # Shape: [5, 3]
b = torch.randn(3)         # Shape: [3]
C = A + b                  # Shape: [5, 3]

X = torch.randn(2, 1, 3)   # Shape: [2, 1, 3]
Y = torch.randn(1, 4, 3)   # Shape: [1, 4, 3]
Z = X + Y                  # Shape: [2, 4, 3]`} />
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Gradients and Derivatives */}
            <Paper className="p-6 bg-pink-50">
              <Title order={3} className="mb-4">Gradients and Jacobians</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Gradient Vector</Title>
                    <Text size="sm" className="mb-3">
                      For scalar function <InlineMath>{`f: \\mathbb{R}^n \\rightarrow \\mathbb{R}`}</InlineMath>:
                    </Text>
                    <BlockMath>{`\\nabla f = \\begin{bmatrix} \\frac{\\partial f}{\\partial x_1} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x_n} \\end{bmatrix}`}</BlockMath>
                    <Text size="sm" className="mt-2">Points in direction of steepest increase</Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">Jacobian Matrix</Title>
                    <Text size="sm" className="mb-3">
                      For vector function <InlineMath>{`f: \\mathbb{R}^n \\rightarrow \\mathbb{R}^m`}</InlineMath>:
                    </Text>
                    <BlockMath>{`J = \\begin{bmatrix} \\frac{\\partial f_1}{\\partial x_1} & \\cdots & \\frac{\\partial f_1}{\\partial x_n} \\\\ \\vdots & \\ddots & \\vdots \\\\ \\frac{\\partial f_m}{\\partial x_1} & \\cdots & \\frac{\\partial f_m}{\\partial x_n} \\end{bmatrix}`}</BlockMath>
                    <Text size="sm" className="mt-2">Used in backpropagation chain rule</Text>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper className="p-4 bg-white mt-4">
                <Title order={4} className="mb-3">Chain Rule for Backpropagation</Title>
                <Text size="sm" className="mb-3">
                  For composite function <InlineMath>{`h(x) = f(g(x))`}</InlineMath>:
                </Text>
                <BlockMath>{`\\frac{\\partial h}{\\partial x} = \\frac{\\partial f}{\\partial g} \\cdot \\frac{\\partial g}{\\partial x}`}</BlockMath>
                
                <CodeBlock language="python" code={`# Automatic differentiation in PyTorch
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
z = y.sum()

# Compute gradients
z.backward()
print(x.grad)  # tensor([2., 4., 6.])

# Gradient computation: dz/dx = d(sum(x^2))/dx = 2x`} />
              </Paper>
            </Paper>
          
        </div>

        {/* Summary */}
        <div data-slide>
          
            <Title order={2} className="mb-6">Part 2 Summary: Mathematical Framework</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                  <Title order={3} className="mb-4">Core Mathematical Concepts</Title>
                  <List spacing="md">
                    <List.Item>Machine learning as function approximation from data</List.Item>
                    <List.Item>Parametric models with learnable weights <InlineMath>{`\\theta`}</InlineMath></List.Item>
                    <List.Item>Loss functions quantify prediction errors</List.Item>
                    <List.Item>Gradient descent iteratively minimizes loss</List.Item>
                    <List.Item>Linear algebra operations form computational backbone</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                  <Title order={3} className="mb-4">Key Optimization Insights</Title>
                  <List spacing="md">
                    <List.Item>Mini-batch SGD balances efficiency and stability</List.Item>
                    <List.Item>Modern optimizers (Adam) adapt learning rates per parameter</List.Item>
                    <List.Item>Regularization prevents overfitting</List.Item>
                    <List.Item>Learning rate scheduling improves convergence</List.Item>
                    <List.Item>Automatic differentiation enables efficient gradient computation</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
            
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6">
              <Title order={3} className="mb-4 text-center">Mathematical Foundation Complete</Title>
              <Text size="lg" className="text-center">
                These mathematical principles—optimization, linear algebra, and gradient-based learning—form the 
                foundation upon which all deep learning architectures are built. Next, we'll see how these concepts 
                combine to create neural networks.
              </Text>
            </Paper>
          
        </div>

      </Stack>
    </Container>
  );
};

export default MathematicalFramework;