import React from 'react';
import { Title, Text, Stack, Grid, Table, Box } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const Optimization = () => {
  return (
    <Stack spacing="xl" w="100%">
      <Title order={1} id="optimization-techniques">Optimization Techniques in Deep Learning</Title>
      <Text>
        Optimization is crucial in training neural networks effectively. This section covers the fundamental concepts
        of loss functions, optimization algorithms, and their mathematical foundations.
      </Text>

      {/* Loss Functions Section */}
      <Title order={2} id="loss-functions">Loss Functions</Title>
      <Text>
        Loss functions measure the difference between predicted and actual values, guiding the network's learning process.
        Here are the most common loss functions:
      </Text>
      
      <Grid>
        <Grid.Col span={12} md={6}>
          <Table withBorder withColumnBorders>
            <thead>
              <tr>
                <th>Loss Function</th>
                <th>Formula</th>
                <th>Use Case</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Mean Squared Error (MSE)</td>
                <td><InlineMath>{`\\frac{1}{n}\\sum_{i=1}^n(y_i - \\hat{y}_i)^2`}</InlineMath></td>
                <td>Regression tasks</td>
              </tr>
              <tr>
                <td>Cross-Entropy Loss</td>
                <td><InlineMath>{`-\\sum_{i=1}^n y_i \\log(\\hat{y}_i)`}</InlineMath></td>
                <td>Classification tasks</td>
              </tr>
              <tr>
                <td>Mean Absolute Error (MAE)</td>
                <td><InlineMath>{`\\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|`}</InlineMath></td>
                <td>Regression with outliers</td>
              </tr>
            </tbody>
          </Table>
        </Grid.Col>
        
        <Grid.Col span={12} md={6}>
          <Box className="w-full h-64">
            <svg viewBox="0 0 400 300" className="w-full h-full">
              {/* Grid lines */}
              <g stroke="#eee" strokeWidth="1">
                {[0, 1, 2, 3, 4].map((i) => (
                  <line 
                    key={`v-${i}`} 
                    x1={80 + i * 80} 
                    y1="30" 
                    x2={80 + i * 80} 
                    y2="250" 
                  />
                ))}
                {[0, 1, 2, 3, 4].map((i) => (
                  <line 
                    key={`h-${i}`} 
                    x1="80" 
                    y1={30 + i * 55} 
                    x2="400" 
                    y2={30 + i * 55} 
                  />
                ))}
              </g>

              {/* Axes */}
              <line x1="80" y1="250" x2="400" y2="250" stroke="black" strokeWidth="2" />
              <line x1="80" y1="30" x2="80" y2="250" stroke="black" strokeWidth="2" />

              {/* MSE Line */}
              <path 
                d={`M80,230 L160,170 L240,120 L320,90 L400,70`}
                stroke="#8884d8"
                strokeWidth="2"
                fill="none"
              />

              {/* MAE Line */}
              <path 
                d={`M80,220 L160,160 L240,110 L320,80 L400,60`}
                stroke="#82ca9d"
                strokeWidth="2"
                fill="none"
              />

              {/* Cross Entropy Line */}
              <path 
                d={`M80,240 L160,180 L240,130 L320,100 L400,80`}
                stroke="#ffc658"
                strokeWidth="2"
                fill="none"
              />

              {/* Legend */}
              <g transform="translate(100, 20)">
                <rect x="0" y="0" width="15" height="15" fill="#8884d8" />
                <text x="20" y="12" fontSize="12">MSE</text>
                <rect x="80" y="0" width="15" height="15" fill="#82ca9d" />
                <text x="100" y="12" fontSize="12">MAE</text>
                <rect x="160" y="0" width="15" height="15" fill="#ffc658" />
                <text x="180" y="12" fontSize="12">Cross Entropy</text>
              </g>

              {/* Axis Labels */}
              <text x="240" y="280" textAnchor="middle">Epochs</text>
              <text x="40" y="140" textAnchor="middle" transform="rotate(-90, 40, 140)">Loss</text>
            </svg>
          </Box>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

# Common PyTorch loss functions
mse_loss = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()
mae_loss = nn.L1Loss()

# Example usage
predictions = torch.tensor([0.7, 0.2, 0.1])
targets = torch.tensor([1, 0, 0])

# Calculate loss
mse = mse_loss(predictions, targets)
print(f"MSE Loss: {mse.item():.4f}")
`}
      />

      {/* Rest of the component remains the same... */}
      {/* Common Optimizers Section */}
      <Title order={2} id="optimizers">Common Optimizers</Title>
      <Text>
        Optimizers update network weights based on the computed gradients. Each optimizer has unique characteristics
        suited for different scenarios.
      </Text>

      <Grid>
        <Grid.Col span={12}>
          <Table withBorder withColumnBorders>
            <thead>
              <tr>
                <th>Optimizer</th>
                <th>Key Features</th>
                <th>Mathematical Update Rule</th>
                <th>Best Use Case</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>SGD</td>
                <td>Simple, memory-efficient</td>
                <td><InlineMath>{`w_{t+1} = w_t - \\eta \\nabla L(w_t)`}</InlineMath></td>
                <td>Simple problems, good convergence understanding</td>
              </tr>
              <tr>
                <td>Adam</td>
                <td>Adaptive learning rates, momentum</td>
                <td><InlineMath>{`w_{t+1} = w_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t`}</InlineMath></td>
                <td>Deep networks, sparse gradients</td>
              </tr>
              <tr>
                <td>RMSprop</td>
                <td>Adaptive learning rates</td>
                <td><InlineMath>{`w_{t+1} = w_t - \\frac{\\eta}{\\sqrt{E[g^2]_t + \\epsilon}} g_t`}</InlineMath></td>
                <td>Non-stationary objectives</td>
              </tr>
            </tbody>
          </Table>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.optim as optim

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Initialize different optimizers
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Training loop example
def train_step(optimizer):
    optimizer.zero_grad()  # Clear gradients
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
`}
      />

      {/* Mathematical Foundations Section */}
      <Title order={2} id="math-formulations">Mathematical Foundations</Title>
      <Text>
        Understanding the mathematical foundations of optimization helps in choosing appropriate algorithms and
        hyperparameters for specific problems.
      </Text>

      <BlockMath>{`
        \\text{Gradient Descent Update Rule:} \\quad w_{t+1} = w_t - \\eta \\nabla L(w_t)
      `}</BlockMath>

      <BlockMath>{`
        \\text{Adam Momentum Update:} \\quad m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t
      `}</BlockMath>

      <BlockMath>{`
        \\text{RMSprop Update:} \\quad v_t = \\beta v_{t-1} + (1-\\beta)g_t^2
      `}</BlockMath>

      {/* Hyperparameter Impact Section */}
      <Title order={2} id="hyperparameters">Hyperparameter Impact</Title>
      <Text>
        Hyperparameters significantly affect the training process. Here's a guide to the most important ones:
      </Text>

      <Grid>
        <Grid.Col span={12}>
          <Table withBorder withColumnBorders>
            <thead>
              <tr>
                <th>Hyperparameter</th>
                <th>Typical Range</th>
                <th>Impact</th>
                <th>Guidelines</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Learning Rate (η)</td>
                <td>1e-4 to 1e-1</td>
                <td>Controls step size in weight updates</td>
                <td>Start with 0.01, adjust based on loss curve</td>
              </tr>
              <tr>
                <td>Batch Size</td>
                <td>16 to 512</td>
                <td>Affects gradient estimation quality</td>
                <td>Larger for more stable gradients</td>
              </tr>
              <tr>
                <td>Momentum (β)</td>
                <td>0.9 to 0.999</td>
                <td>Helps escape local minima</td>
                <td>0.9 is a good default</td>
              </tr>
            </tbody>
          </Table>
        </Grid.Col>
      </Grid>

      <CodeBlock
        language="python"
        code={`
# Example of hyperparameter tuning
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize optimizer with different learning rates
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                             verbose=True)

# Training loop with scheduler
def train_epoch(model, optimizer, scheduler, train_loader, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Update learning rate based on validation loss
    val_loss = validate(model, val_loader, criterion)
    scheduler.step(val_loss)
`}
      />
    </Stack>
  );
};

export default Optimization;