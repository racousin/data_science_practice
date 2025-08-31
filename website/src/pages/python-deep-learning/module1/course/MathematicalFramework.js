import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const MathematicalFramework = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
        {/* Part 2: Mathematical Framework */}
          <div data-slide>
          <Title order={1} mb="xl">
            Machine Learning in a nutshell
          </Title>
          
          {/* Machine Learning Objective */}
          </div>
          <div data-slide>
            <Title order={2} className="mb-6" id="ml-objective">
              The Machine Learning Objective
            </Title>
                    <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj1.png"
                        alt="Computer Vision Applications"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Example : predict the future
                      </Text>
          </div>
          <div data-slide data-tag="ml-variables">
            <Text size="lg" className="mb-4">
              <strong>X:</strong> Explanatory variables (features/characteristics)<br/>
              <strong>Y:</strong> Variable to explain (target)<br/>
              Concretely, we want to find a relationship Y = f(X)
            </Text>
            
            <Text className="mb-4">
              <strong>Example:</strong> f(year, latitude, altitude) = predicted temperature
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} className="mb-2">Training Phase</Title>
                  <List size="sm">
                    <List.Item>We have observed/measured (X, Y) pairs</List.Item>
                    <List.Item>The model learns the relationship between X and Y</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
              <Grid.Col span={6}>
                <Paper className="p-4 bg-green-50">
                  <Title order={4} className="mb-2">Prediction Phase</Title>
                  <List size="sm">
                    <List.Item>We receive new X values</List.Item>
                    <List.Item>Predict the corresponding Y values</List.Item>
                  </List>
                </Paper>
              </Grid.Col>
            </Grid>
          </div>
          <div data-slide>
            <Title order={3} className="mb-4">Regression, Classification & Mixed Problems</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={4}>
                <Paper className="p-4 bg-blue-50 h-full">
                  <Title order={4} className="mb-3">Regression Problems</Title>
                  <Text size="sm" className="mb-3">Predict continuous values</Text>
                  <BlockMath>{`y \\in \\mathbb{R}^m`}</BlockMath>
                  
                  <div className="mt-3">
                    <Text className="font-semibold text-sm">Examples:</Text>
                    <List size="sm">
                      <List.Item><strong>1D:</strong> House price (y ∈ ℝ)</List.Item>
                      <List.Item><strong>2D:</strong> 2D coordinates (x,y)</List.Item>
                      <List.Item><strong>Multi-dim:</strong> Stock portfolio values</List.Item>
                      <List.Item><strong>High-dim:</strong> Image generation (pixel values)</List.Item>
                    </List>
                  </div>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-4 bg-green-50 h-full">
                  <Title order={4} className="mb-3">Classification Problems</Title>
                  <Text size="sm" className="mb-3">Predict discrete categories</Text>
                  <BlockMath>{`y \\in \\{1, 2, ..., K\\}`}</BlockMath>
                  
                  <div className="mt-3">
                    <Text className="font-semibold text-sm">Examples:</Text>
                    <List size="sm">
                      <List.Item><strong>Binary:</strong> Spam detection (y ∈ {`{0,1}`})</List.Item>
                      <List.Item><strong>Multi-class:</strong> Image recognition (10 classes)</List.Item>
                      <List.Item><strong>Multi-label:</strong> Document tagging</List.Item>
                      <List.Item><strong>Hierarchical:</strong> Species classification</List.Item>
                    </List>
                  </div>
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Paper className="p-4 bg-purple-50 h-full">
                  <Title order={4} className="mb-3">Mixed Problems</Title>
                  <Text size="sm" className="mb-3">Combine regression & classification</Text>
                  <BlockMath>{`y = [y_{reg}, y_{cls}]`}</BlockMath>
                  
                  <div className="mt-3">
                    <Text className="font-semibold text-sm">Examples:</Text>
                    <List size="sm">
                      <List.Item><strong>Object detection:</strong> Bounding box + class</List.Item>
                      <List.Item><strong>Facial analysis:</strong> Age (reg) + gender (cls)</List.Item>
                      <List.Item><strong>Medical diagnosis:</strong> Severity + disease type</List.Item>
                      <List.Item><strong>Autonomous driving:</strong> Distance + object type</List.Item>
                    </List>
                  </div>
                </Paper>
              </Grid.Col>
            </Grid>
          </div>
                      <div data-slide >
              <Title order={3} className="mb-4">Data Characteristics</Title>
                                  <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj2.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        44 observations/samples, 2 dimensions
                      </Text>
                                                        <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj3.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        9 observations, 3 dimensions
                      </Text>
                      
            <Paper className="p-4 bg-white mt-4">
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Title order={4} className="mb-2">Quantity</Title>
                  <Text size="sm">Sample size: number of observations</Text>
                </Grid.Col>
                <Grid.Col span={6}>
                  <Title order={4} className="mb-2">Dimension</Title>
                  <Text size="sm">Dimensionality: number of variables per observation (features)</Text>
                </Grid.Col>
              </Grid>
            </Paper>
              </div>
            <div data-slide>
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
                    <Text size="sm" c="dimmed">
                      where <InlineMath>{`\\mathcal{F}`}</InlineMath> is the hypothesis class, 
                      and <InlineMath>{`\\ell`}</InlineMath> is the loss function
                    </Text>
                  </div>
                </div>
              </Paper>
</div>
<div data-slide>
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

</div>
<div data-slide>
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
        <div data-slide>
            <Title order={2} className="mb-6" id="models-parameters">
              Models and Parameters
            </Title>
                                  <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj8.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Example: Linear function R to R 2 params
                      </Text>
                                                        <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj6.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Polynomial function R to R 3 params
                      </Text>
                                                          <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj15.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Linear function R2 to R 3 params
                      </Text>
                                                                                <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj16.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Quadratic function R² to R, 5 params
                      </Text>
            
            <Paper className="p-4 bg-white mt-6">
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Title order={4} className="mb-2">Parameters</Title>
                  <Text size="sm">Number of values the model must learn</Text>
                </Grid.Col>
                <Grid.Col span={4}>
                  <Title order={4} className="mb-2">Input/Output Dimension</Title>
                  <List size="sm">
                    <List.Item><strong>Input dimension:</strong> dimension of explanatory variables (features)</List.Item>
                    <List.Item><strong>Output dimension:</strong> number of variables to predict (target)</List.Item>
                  </List>
                </Grid.Col>
                <Grid.Col span={4}>
                  <Title order={4} className="mb-2">Operations</Title>
                  <Text size="sm">Between parameters and input data:<br/>Addition, multiplication, other operations</Text>
                </Grid.Col>
              </Grid>
            </Paper>
        </div>
        {/* Models and Parameters */}
        <div data-slide data-tag="parametric-models">
          

            
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
                    <Text size="sm" c="dimmed">
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
                    <Text size="sm" c="dimmed">
                      Can approximate any continuous function (universal approximation)
                    </Text>
                  </Paper>
                </Grid.Col>
              </Grid>
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
                        <Text size="xs" c="dimmed">Penalizes large errors heavily, sensitive to outliers</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Mean Absolute Error (MAE):</Text>
                        <BlockMath>{`\\ell_{MAE}(y, \\hat{y}) = \\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|`}</BlockMath>
                        <Text size="xs" c="dimmed">Robust to outliers, non-differentiable at zero</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Huber Loss:</Text>
                        <BlockMath>{`\\ell_{Huber}(y, \\hat{y}) = \\begin{cases} 
                          \\frac{1}{2}(y - \\hat{y})^2 & \\text{if } |y - \\hat{y}| \\leq \\delta \\\\
                          \\delta|y - \\hat{y}| - \\frac{1}{2}\\delta^2 & \\text{otherwise}
                        \\end{cases}`}</BlockMath>
                        <Text size="xs" c="dimmed">Combines MSE and MAE benefits</Text>
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
                        <Text size="xs" c="dimmed">Standard for multi-class classification</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Binary Cross-Entropy:</Text>
                        <BlockMath>{`\\ell_{BCE}(y, \\hat{p}) = -[y\\log(\\hat{p}) + (1-y)\\log(1-\\hat{p})]`}</BlockMath>
                        <Text size="xs" c="dimmed">For binary classification problems</Text>
                      </div>
                      
                      <div>
                        <Text className="font-semibold text-sm">Focal Loss:</Text>
                        <BlockMath>{`\\ell_{FL}(p_t) = -\\alpha_t(1-p_t)^\\gamma \\log(p_t)`}</BlockMath>
                        <Text size="xs" c="dimmed">Addresses class imbalance by focusing on hard examples</Text>
                      </div>
                    </div>
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
            </div>

<div data-slide>
                                  <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj10.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Example: Initial linear regression params
                      </Text>
                                                        <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj5.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Example: Compute error
                      </Text>
            <Paper className="p-6 bg-gray-50 mt-6">
              <Title order={3} className="mb-4">Model Training Process</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">0. Initialization</Title>
                    <List size="sm">
                      <List.Item>Choose a model type (e.g., linear regression)</List.Item>
                      <List.Item>Initialize parameters randomly</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">1. Prediction</Title>
                    <List size="sm">
                      <List.Item>Predict Y from X with current model</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">2. Error Calculation</Title>
                    <List size="sm">
                      <List.Item>Compare Y_predicted vs Y_actual</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} className="mb-3">3. Adjustment</Title>
                    <List size="sm">
                      <List.Item>Modify parameters to reduce error (gradient descent)</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
              </Grid>
              
              <Paper className="p-4 bg-blue-50 mt-4">
                <Title order={4} className="mb-2">4. Iteration</Title>
                <Text size="sm" className="mb-2">Repeat from step 1 until error no longer decreases</Text>
                <Text size="sm" className="font-semibold">Objective: Minimize prediction error by optimizing parameters</Text>
              </Paper>
            </Paper>

</div>

<div data-slide>
                                                          <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/error_loss_evolution.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Example: error_loss_evolution
                      </Text>
                                                                                <Flex direction="column" align="center" className="mb-4">
                      <Image
                        src="/assets/python-deep-learning/module1/gradent_descent.png"
                        w={{ base: 400, sm: 600, md: 800 }}
                        h="auto"
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Gradient descent visualization
                      </Text>
</div>

<div data-slide>
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

            </Grid>
            
        </div>

      </Stack>
    </Container>
  );
};

export default MathematicalFramework;