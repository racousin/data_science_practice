import React from 'react';
import { Title, Text, Stack, Group, Table } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const Advanced = () => {
  return (
    <Stack spacing="xl" className="w-full">
      <section id="hyperparameter-opt">
        <Title order={2} mb="md">Hyperparameter Optimization</Title>
        <Text mb="md">
          Hyperparameter optimization is crucial for achieving optimal model performance. We'll explore several systematic approaches:
        </Text>
        
        <Title order={3} mb="sm">Grid Search vs Random Search</Title>
        <Text mb="md">
          While grid search exhaustively searches through a predefined parameter space, random search often finds better parameters in less time by sampling from parameter distributions.
        </Text>

        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import RandomizedSearchCV
import torch.nn as nn

# Define parameter space
param_distributions = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'n_layers': [2, 3, 4],
    'hidden_size': [64, 128, 256]
}

# Example of random search implementation
def train_model_with_params(params):
    model = nn.Sequential(
        nn.Linear(input_size, params['hidden_size']),
        nn.ReLU(),
        nn.Linear(params['hidden_size'], output_size)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    # Training loop implementation
    return validation_score

# Perform random search
best_params = None
best_score = float('-inf')
for _ in range(20):  # Number of trials
    params = {k: random.choice(v) for k, v in param_distributions.items()}
    score = train_model_with_params(params)
    if score > best_score:
        best_score = score
        best_params = params`}
        />

        <Title order={3} mt="lg" mb="sm">Bayesian Optimization</Title>
        <Text mb="md">
          Bayesian optimization uses probabilistic models to guide the search for optimal hyperparameters, making it more efficient than random or grid search.
        </Text>

        <CodeBlock
          language="python"
          code={`
from bayes_opt import BayesianOptimization

# Define the objective function
def objective(learning_rate, hidden_size):
    # Convert to appropriate types
    hidden_size = int(hidden_size)
    
    # Create and train model
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    return validation_score

# Define parameter bounds
pbounds = {
    'learning_rate': (1e-4, 1e-2),
    'hidden_size': (32, 256)
}

# Initialize optimizer
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1
)

# Optimize
optimizer.maximize(
    init_points=5,    # Number of initial random explorations
    n_iter=20         # Number of optimization iterations
)`}
        />
      </section>

      <section id="custom-loss">
        <Title order={2} mt="xl" mb="md">Custom Loss Functions</Title>
        <Text mb="md">
          Custom loss functions allow you to define specific optimization objectives for your model. Here's how to implement custom losses in PyTorch:
        </Text>

        <CodeBlock
          language="python"
          code={`
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Example usage
model = YourModel()
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for inputs, targets in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
        />
      </section>

      <section id="advanced-reg">
        <Title order={2} mt="xl" mb="md">Advanced Regularization</Title>
        
        <Title order={3} mb="sm">Label Smoothing</Title>
        <Text mb="md">
          Label smoothing helps prevent the model from becoming overconfident by softening the target distributions:
        </Text>

        <BlockMath>
          y'_i = (1 - \alpha)y_i + \alpha/K
        </BlockMath>

        <CodeBlock
          language="python"
          code={`
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))`}
        />

        <Title order={3} mt="lg" mb="sm">Mixup Training</Title>
        <Text mb="md">
          Mixup creates virtual training examples by linearly interpolating between pairs of examples and their labels:
        </Text>

        <CodeBlock
          language="python"
          code={`
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Training loop with mixup
for inputs, targets in train_loader:
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
    outputs = model(inputs)
    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
        />
      </section>

      <section id="batch-norm">
        <Title order={2} mt="xl" mb="md">Batch Normalization</Title>
        <Text mb="md">
          Batch normalization normalizes the input of each layer, reducing internal covariate shift and enabling faster training:
        </Text>

        <BlockMath>
  {"\\hat{x}^{(k)} = \\frac{x^{(k)} - E[x^{(k)}]}{\\sqrt{Var[x^{(k)}] + \\epsilon}}"}
</BlockMath>



        <Text mb="md">
          The normalized values are then scaled and shifted using learned parameters γ and β:
        </Text>

        <BlockMath>
  {String.raw`\hat{x}_i = \frac{x_i - E[x_i]}{\sqrt{Var[x_i] + \epsilon}}`}
</BlockMath>
        <CodeBlock
          language="python"
          code={`
class BatchNormNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# Example usage with different batch norm moments
model = BatchNormNet(input_size=784, hidden_size=256, output_size=10)
model.train()  # Use batch statistics during training
predictions = model(train_data)

model.eval()   # Use running statistics during inference
predictions = model(test_data)`}
        />
      </section>
    </Stack>
  );
};

export default Advanced;