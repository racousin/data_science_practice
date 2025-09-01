import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const MathematicalFramework = () => {
  return (
    <Container size="xl" py="xl">
      <Stack spacing="xl">
        
        {/* Part 2: Mathematical Framework */}
          <div data-slide>
          <Title order={1} mb="xl">
            Machine Learning in a nutshell
          </Title>
          
          {/* Machine Learning Objective */}
          </div>
          <div data-slide>
            <Title order={2} mb="xl" className="slide-title"  id="ml-objective">
              The Machine Learning Objective
            </Title>
                        <Text size="lg" mb="xl">
Machine learning is a field of artificial intelligence that enables systems to automatically learn patterns from data.
            </Text>
            
                    <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj1.png"
                        alt="Computer Vision Applications"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Example : predict the future
                      </Text>
          </div>
          <div data-slide>
            <Title order={4} mb="xs" className="slide-title">Features and Target</Title>
            <Text size="lg" mb="xl">
              <strong>X:</strong> Explanatory variables (features/characteristics)<br/>
              <strong>Y:</strong> Variable to explain (target)<br/>
              Concretely, we want to find a relationship Y = f(X)
            </Text>
            
            <Text mb="xl">
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
          
<Title order={3} mb="md" className="slide-title">Data Characteristics</Title>

<Paper p="md" bg="white" mt="md">
  <Grid gutter="lg">
    <Grid.Col span={6}>
      <Title order={4} className="mb-2">Quantity</Title>
      <Text size="sm" mb="xs">Sample size: number of observations</Text>
      <Text size="sm" className="text-gray-600" mb="md">
        Number of samples: <InlineMath>{`n \\in \\mathbb{N}`}</InlineMath>
      </Text>
      
      <Flex direction="column" align="center" mb="md">
        <Image
          src="/assets/python-deep-learning/module1/ai_obj3.png"
          style={{ maxWidth: 'min(400px, 90vw)', height: 'auto' }}
          fluid
        />
      </Flex>
      <Text component="p" ta="center" mt="xs">
        <InlineMath>{`n = 9`}</InlineMath> observations, <InlineMath>{`d = 3`}</InlineMath> dimensions
      </Text>
                                            
    </Grid.Col>
    <Grid.Col span={6}>
      <Title order={4} className="mb-2">Dimension</Title>
      <Text size="sm" mb="xs">Dimensionality: number of variables per observation (features)</Text>
      <Text size="sm" className="text-gray-600" mb="md">
        Number of features: <InlineMath>{`d \\in \\mathbb{N}`}</InlineMath>
      </Text>
      
      <Flex direction="column" align="center" mb="md">
        <Image
          src="/assets/python-deep-learning/module1/ai_obj2.png"
          style={{ maxWidth: 'min(400px, 90vw)', height: 'auto' }}
          fluid
        />
      </Flex>
      <Text component="p" ta="center" mt="xs">
        <InlineMath>{`n = 44`}</InlineMath> observations/samples, <InlineMath>{`d = 2`}</InlineMath> dimensions
      </Text>
    </Grid.Col>
  </Grid>
</Paper>
<div data-slide>
            <Title order={3} mb="md" className="slide-title">Regression, Classification</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={4}>
                <Paper className="p-4 bg-blue-50 h-full">
                  <Title order={4} mb="sm">Regression Problems</Title>
                  <Text size="sm" className="mb-3">Predict continuous values</Text>
                  <BlockMath>{`y \\in \\mathbb{R}^k`}</BlockMath>
                  
                  <div className="mt-3">
                    <Text fw="bold" size="sm">Examples:</Text>
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
                  <Title order={4} mb="sm">Classification Problems</Title>
                  <Text size="sm" className="mb-3">Predict discrete categories</Text>
                  <BlockMath>{`y \\in \\{1, 2, ..., K\\}`}</BlockMath>
                  
                  <div className="mt-3">
                    <Text fw="bold" size="sm">Examples:</Text>
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
                  <Title order={4} mb="sm">Mixed Problems</Title>
                  <Text size="sm" className="mb-3">Combine regression & classification</Text>
                  <BlockMath>{`y = [y_{reg}, y_{cls}]`}</BlockMath>
                  
                  <div className="mt-3">
                    <Text fw="bold" size="sm">Examples:</Text>
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
<div data-slide>
  <Title order={2} mb="xl" className="slide-title" id="models-parameters">
    Parametric Models
  </Title>
  <Text size="lg" mb="md">
    Parametric Models are defined by 3 characteristics:
  </Text>
  
  <Paper className="p-4 bg-white mt-6">
    <Grid gutter="lg">
      <Grid.Col span={4}>
        <Title order={4} className="mb-2">Parameters</Title>
        <Text size="sm" mb="xs">Number of values the model must learn</Text>
        <Text size="sm" className="text-gray-600">
          Parameter vector: <InlineMath>{`\\theta \\in \\mathbb{R}^p`}</InlineMath>
          <br />
          where <InlineMath>{`p`}</InlineMath> is the number of parameters
        </Text>
      </Grid.Col>
      <Grid.Col span={4}>
        <Title order={4} className="mb-2">Input/Output Dimension</Title>
        <List size="sm">
          <List.Item>
            <strong>Input dimension:</strong> <InlineMath>{`x \\in \\mathbb{R}^d`}</InlineMath> 
            <br />
            <Text size="xs" className="text-gray-600">dimension of explanatory variables (features)</Text>
          </List.Item>
          <List.Item>
            <strong>Output dimension:</strong> <InlineMath>{`y \\in \\mathbb{R}^k`}</InlineMath>
            <br />
            <Text size="xs" className="text-gray-600">number of variables to predict (target)</Text>
          </List.Item>
        </List>
      </Grid.Col>
      <Grid.Col span={4}>
        <Title order={4} className="mb-2">Operations</Title>
        <Text size="sm" mb="xs">Between parameters and input data</Text>
        <Text size="sm" className="text-gray-600">
          Model function: <InlineMath>{`f_{\\theta}: \\mathbb{R}^d \\to \\mathbb{R}^k`}</InlineMath>
          <br />
          <InlineMath>{`\\hat{y} = f_{\\theta}(x)`}</InlineMath>
          <br />
          Operations: addition, multiplication, composition
        </Text>
      </Grid.Col>
    </Grid>
  </Paper>
                                    <Paper className="p-4 bg-white mt-6">
              <Grid gutter="lg">
                <Grid.Col span={6}>
                                  <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj8.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Example: Linear function R to R 2 params
                      </Text>

                </Grid.Col>
                <Grid.Col span={6}>
                                                           <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj6.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Polynomial function R to R 3 params
                      </Text>
                </Grid.Col>
              </Grid>
            </Paper>



            

        </div>
        {/* Models and Parameters */}
        <div data-slide>
          
                                    <Paper className="p-4 bg-white mt-6">
              <Grid gutter="lg">
                <Grid.Col span={6}>
                                                       <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj15.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Linear function R2 to R 3 params
                      </Text>
                </Grid.Col>
                <Grid.Col span={6}>
                                                                                <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj16.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Quadratic function R² to R, 5 params
                      </Text>
                </Grid.Col>
              </Grid>
            </Paper>
            
            <Paper className="p-6 bg-blue-50 mb-6">
              
              <Grid gutter="lg">
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">Linear Models</Title>
                    <Text size="sm" className="mb-3">The simplest parametric model:</Text>
                    <BlockMath>{`f_\\theta(x) = w^T x + b`}</BlockMath>
                    <Text size="sm" className="mb-2">Parameters: <InlineMath>{`\\theta = \\{w \\in \\mathbb{R}^d, b \\in \\mathbb{R}\\}`}</InlineMath></Text>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">Not linear Models (eg : Neural Networks)</Title>
                    <Text size="sm" className="mb-3">Composition of linear and non-linear functions:</Text>
                    <BlockMath>{`f_\\theta(x) = W_L \\cdot \\sigma(W_{L-1} \\cdot ... \\cdot \\sigma(W_1 x + b_1) ... + b_{L-1}) + b_L`}</BlockMath>
                    <Text size="sm" className="mb-2">Parameters: <InlineMath>{`\\theta = \\{W_1, b_1, ..., W_L, b_L\\}`}</InlineMath></Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Paper>
          
        </div>

        {/* Loss Functions and Optimization */}
        <div data-slide>
          
            <Title order={2} mb="xl" className="slide-title" id="loss-functions">
              Loss Functions and Optimization Problems
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mb-6">
              <Title order={3} mb="md">The Role of Loss Functions</Title>
<Text size="lg" mb="md">
  Loss functions quantify the error/distance between real targets <InlineMath>{`y`}</InlineMath> and our model outputs <InlineMath>{`\\hat{y} = f_{\\theta}(x)`}</InlineMath>.
</Text>   
<BlockMath>
{`
\\begin{cases}
l: \\mathbb{R}^{k \\times n} \\times \\mathbb{R}^{k \\times n} & \\to \\mathbb{R} \\\\
(Y, \\hat{Y}) & \\mapsto l(Y, \\hat{Y})
\\end{cases}
`}
</BlockMath>
            </Paper>

            {/* Common Loss Functions */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} mb="md">Common Loss Functions</Title>
              
              <Grid gutter="lg">
                {/* Regression Losses */}
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-blue-50">
                    <Title order={4} mb="sm">Regression Losses</Title>
                    
                    <div className="space-y-4">
                      <div>
                        <Text fw="bold" size="sm">Mean Squared Error (MSE):</Text>
                        <BlockMath>{`\\ell_{MSE}(y, \\hat{y}) = \\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2`}</BlockMath>
                        <Text size="xs" c="dimmed">Penalizes large errors heavily, sensitive to outliers</Text>
                      </div>
                      
                      <div>
                        <Text fw="bold" size="sm">Mean Absolute Error (MAE):</Text>
                        <BlockMath>{`\\ell_{MAE}(y, \\hat{y}) = \\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|`}</BlockMath>
                        <Text size="xs" c="dimmed">Robust to outliers, non-differentiable at zero</Text>
                      </div>
                      
                    </div>
                  </Paper>
                </Grid.Col>
                
                {/* Classification Losses */}
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-green-50">
                    <Title order={4} mb="sm">Classification Losses</Title>
                    
                    <div className="space-y-4">
                                            <div>
                        <Text fw="bold" size="sm">Binary Cross-Entropy:</Text>
                        <BlockMath>{`\\ell_{BCE}(y, \\hat{p}) = -[y\\log(\\hat{p}) + (1-y)\\log(1-\\hat{p})]`}</BlockMath>
                        <Text size="xs" c="dimmed">For binary classification problems</Text>
                      </div>
                      <div>
                        <Text fw="bold" size="sm">Cross-Entropy Loss:</Text>
                        <BlockMath>{`\\ell_{CE}(y, \\hat{p}) = -\\sum_{i=1}^n \\sum_{c=1}^C y_{ic} \\log(\\hat{p}_{ic})`}</BlockMath>
                        <Text size="xs" c="dimmed">Standard for multi-class classification</Text>
                      </div>
                      

                      
                    </div>
                  </Paper>
                </Grid.Col>
              </Grid>
              
            </Paper>

          
        </div>
<div data-slide>
  <Title order={3} mb="md" className="slide-title">The Fundamental Learning Problem</Title>
  
  <Text size="lg" mb="md">
    For a given parametric model <InlineMath>{`f_{\\theta}`}</InlineMath>, machine learning seeks to find the best function <InlineMath>{`f_{\\theta}`}</InlineMath> that maps inputs to outputs.
  </Text>
  
  <Text size="lg" mb="md">
    Concretely, we will try to find the <InlineMath>{`\\theta`}</InlineMath> that minimizes the loss.
  </Text>
  
  <Text size="lg" mb="md">
    Fixing the observations <InlineMath>{`X`}</InlineMath> and <InlineMath>{`Y`}</InlineMath>, the loss becomes a function from <InlineMath>{`\\theta \\in \\mathbb{R}^p`}</InlineMath> to <InlineMath>{`\\mathbb{R}`}</InlineMath>:
  </Text>
  
  <BlockMath>
  {`
  \\begin{cases}
  \\ell: \\mathbb{R}^p & \\to \\mathbb{R} \\\\
  \\theta & \\mapsto \\ell(\\theta) = l(Y, f_{\\theta}(X))
  \\end{cases}
  `}
  </BlockMath>
  
  <Text size="lg" mb="md">
    And the machine learning problem becomes:
  </Text>
  
  <BlockMath>
  {`
  \\theta^* = \\arg\\min_{\\theta \\in \\mathbb{R}^p} \\ell(\\theta) = \\arg\\min_{\\theta \\in \\mathbb{R}^p} l(Y, f_{\\theta}(X))
  `}
  </BlockMath>
</div>

        {/* Gradient Descent */}
        <div data-slide>
          
            <Title order={2} className="slide-title" mb="xl" id="gradient-descent">
              Gradient Descent and Optimization
            </Title>
            
            <Paper className="p-6 bg-gradient-to-r from-blue-50 to-cyan-50 mb-6">
              <Title order={3} mb="md">The Gradient Descent Algorithm</Title>
              <Text size="lg" mb="md">
                Gradient descent is the workhorse of machine learning optimization. It iteratively updates parameters 
                in the direction that reduces the loss function <InlineMath>{`l(\\theta) = l(Y, f_{\\theta}(X)`}</InlineMath>. 
              </Text>
              
              <Grid gutter="lg">
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">0. Initialization</Title>
                    <List size="sm" mt="sm">
                      <List.Item>Initialize <InlineMath>{`\\theta_0`}</InlineMath> parameters randomly</List.Item>
                      <List.Item>Choose a learning rate <InlineMath>{`\\eta`}</InlineMath></List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">1. Update Rule</Title>
                    <BlockMath>{`\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}(\\theta_t)`}</BlockMath>
                  </Paper>

                </Grid.Col>
                
              </Grid>
                            <Grid gutter="lg">
                
                <Grid.Col span={6}>
                  <Paper className="p-4 bg-white">
                    <Title order={4} mb="sm">2. Iterate</Title>
                                        <List size="sm" mt="sm">
                      <List.Item>Repeat 1. while <InlineMath>{`l(\\theta_{t+1}) < l(\\theta_t)`}</InlineMath></List.Item></List>
                   
                  </Paper>

                </Grid.Col>
                <Grid.Col span={6}>
                                                                                                                      <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/gradient_descent_optimal.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Gradient descent visualization
                      </Text>
                      </Grid.Col>
              </Grid>
            </Paper>
            </div>

<div data-slide>

            <Paper className="p-6 bg-gray-50 mt-6">
              <Title order={3} mb="md">Model Training Process</Title>
              
                    <List size="sm">
                      <List.Item>Choose a model type (e.g., linear regression)</List.Item>
                      <List.Item>Initialize parameters randomly</List.Item>
                      <List.Item>Train, mimizing the loss using gradient descent</List.Item>
                    </List>
                  </Paper>
                  <Grid>
                
                <Grid.Col span={6}>
<Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/step_000_regression.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Model Initial step
                      </Text>
                </Grid.Col>
                
                <Grid.Col span={6}>
                                                                                                                              <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/step_010_regression.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Model After some steps
                      </Text>
                </Grid.Col>
                
                <Grid.Col span={6}>
                                                                                                        <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/gradient_descent_path.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Loss though steps
                      </Text>
                </Grid.Col>
                                <Grid.Col span={6}>
                                                                                                        <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/cost_convergence.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Loss though steps
                      </Text>
                </Grid.Col>
              </Grid>


</div>

<div data-slide>

                          <Title order={3} mb="md">Learning Rate Importance</Title>                                                                               
                  <Grid>
                
                <Grid.Col span={6}>
                                                                                                      <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/gradient_descent_optimal.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Gradient descent visualization
                      </Text>
                </Grid.Col>
                
                <Grid.Col span={6}>
                                                                                                      <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/gradient_descent_small_lr.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Gradient descent visualization
                      </Text>
                </Grid.Col>
                
                <Grid.Col span={6}>
                                                                                <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/gradient_descent_large_lr.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Gradient descent visualization
                      </Text>
                </Grid.Col>
                                <Grid.Col span={6}>
                                                          <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/gradient_descent_non_convex.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                      <Text component="p" ta="center" mt="xs">
                        Non-Convex Functions
                      </Text>
                </Grid.Col>
              </Grid>

</div>

<div data-slide>




                      
</div>
<div data-slide>


                    
            {/* Variants of Gradient Descent */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} mb="md" className="slide-title">Gradient Descent Variants</Title>
                                                                          <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/batch_stochastic.png"
                        style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                        fluid
                      />
                            <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
                              Source: https://medium.com/data-science/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a
                            </Text>
                    </Flex>
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-blue-50 h-full">
                    <Title order={4} mb="sm">Batch Gradient Descent</Title>
                    <BlockMath>{`\\nabla_\\theta \\mathcal{L} = \\frac{1}{n}\\sum_{i=1}^n \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)`}</BlockMath>
                    <List size="sm" mt="sm">
                      <List.Item>Uses entire dataset per update</List.Item>
                      <List.Item>Stable convergence</List.Item>
                      <List.Item>Slow for large datasets</List.Item>
                      <List.Item>Can converge to sharp minima</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-green-50 h-full">
                    <Title order={4} mb="sm">Stochastic GD (SGD)</Title>
                    <BlockMath>{`\\nabla_\\theta \\mathcal{L} \\approx \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)`}</BlockMath>
                    <List size="sm" mt="sm">
                      <List.Item>One sample per update</List.Item>
                      <List.Item>Very noisy updates</List.Item>
                      <List.Item>Fast iterations</List.Item>
                      <List.Item>Can escape local minima</List.Item>
                    </List>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-yellow-50 h-full">
                    <Title order={4} mb="sm">Mini-batch SGD</Title>
                    <BlockMath>{`\\nabla_\\theta \\mathcal{L} \\approx \\frac{1}{m}\\sum_{i \\in \\mathcal{B}} \\nabla_\\theta \\ell(f_\\theta(x_i), y_i)`}</BlockMath>
                    <List size="sm" mt="sm">
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

<div data-slide>
<Title order={3} mb="md" className="slide-title">More parameters is not necessarly a better model</Title>
              <Grid gutter="lg">
                                <Grid.Col span={6}>
                                                          <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj14.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                                                           <Text component="p" ta="center" mt="xs">
                        Small bias, small variance
                      </Text>
                    </Flex>

                </Grid.Col>
                <Grid.Col span={6}>
                                                          <Flex direction="column" align="center" mb="md">
                      <Image
                        src="/assets/python-deep-learning/module1/ai_obj13.png"
                        style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                                                          <Text component="p" ta="center" mt="xs">
                        No bias, high variance
                      </Text>
 
                </Grid.Col>
                

                
              </Grid>
</div>

<div data-slide>
            {/* Types of Learning Problems */}
            <Paper className="p-6 bg-gray-50 mb-6">
              <Title order={3} mb="md" className="slide-title">Types of Learning Problems</Title>
              
              <Grid gutter="lg">
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-blue-50 h-full">
                    <Title order={4} mb="sm">Supervised Learning</Title>
                    <Text size="sm" className="mb-3">Learning from labeled pairs <InlineMath>{`(x, y)`}</InlineMath></Text>
                    
                    <div className="mb-3">
                      <Text fw="bold" size="sm">Classification:</Text>
                      <BlockMath>{`y \\in \\{1, 2, ..., K\\}`}</BlockMath>
                      <Text size="xs">Examples: Image recognition, spam detection</Text>
                    </div>
                    
                    <div>
                      <Text fw="bold" size="sm">Regression:</Text>
                      <BlockMath>{`y \\in \\mathbb{R}^m`}</BlockMath>
                      <Text size="xs">Examples: Price prediction, weather forecasting</Text>
                    </div>
                  </Paper>
                </Grid.Col>
                
                <Grid.Col span={4}>
                  <Paper className="p-4 bg-green-50 h-full">
                    <Title order={4} mb="sm">Unsupervised Learning</Title>
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
                    <Title order={4} mb="sm">Reinforcement Learning</Title>
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
          For parmaters models, On the other cases (unsupervised, Reinforcment), we will find a way to formulate the problem in a same supervised learning framweork
        </div>

      
      </Stack>
    </Container>
  );
};

export default MathematicalFramework;