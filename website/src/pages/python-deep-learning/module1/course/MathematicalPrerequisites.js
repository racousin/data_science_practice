import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import CodeBlock from 'components/CodeBlock';

const MathematicalPrerequisites = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        {/* Model Parameters */}
        <div id="model-parameters">
          <Title order={1} className="mb-6">
            Mathematical Prerequisites
          </Title>
          <Text size="xl" className="mb-6">
            Model Parameters & Parameter Spaces
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Parameter Spaces in Deep Learning</Title>
            <Text className="mb-4">
              A neural network with parameters <InlineMath>{`\theta`}</InlineMath> defines a function <InlineMath>{`f(x; \theta)`}</InlineMath> that maps inputs <InlineMath>x</InlineMath> to outputs. 
              The parameter space <InlineMath>{`\Theta`}</InlineMath> is the set of all possible parameter values.
            </Text>
            
            <BlockMath>{`f: \\mathbb{R}^d \\times \\Theta \\rightarrow \\mathbb{R}^k`}</BlockMath>
            <Text size="sm" className="mt-2">
              where <InlineMath>d</InlineMath> is input dimension, <InlineMath>k</InlineMath> is output dimension, and <InlineMath>{`\\theta \\in \\Theta \\subseteq \\mathbb{R}^p`}</InlineMath> with <InlineMath>p</InlineMath> parameters.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Title order={4} className="mb-3">Mathematical Notation</Title>
                <List>
                  <List.Item><InlineMath>{`\\theta \\in \\mathbb{R}^d`}</InlineMath>: Parameter vector in d-dimensional space</List.Item>
                  <List.Item><InlineMath>{`W \\in \\mathbb{R}^{m \\times n}`}</InlineMath>: Weight matrix (m outputs, n inputs)</List.Item>
                  <List.Item><InlineMath>{`b \\in \\mathbb{R}^m`}</InlineMath>: Bias vector</List.Item>
                  <List.Item><InlineMath>{`f(x; \\theta)`}</InlineMath>: Function parameterized by <InlineMath>{`\\theta`}</InlineMath></List.Item>
                </List>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Title order={4} className="mb-3">Parameter Count Example</Title>
                <Text size="sm">
                  For a linear layer: input_dim × output_dim + output_dim
                </Text>
                <CodeBlock language="python" code={`# Linear layer: 784 -> 256
input_dim = 784
output_dim = 256

# Weight matrix parameters
weight_params = input_dim * output_dim  # 200,704
# Bias parameters  
bias_params = output_dim                # 256
total_params = weight_params + bias_params  # 200,960

print(f"Total parameters: {total_params:,}")`} />
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-green-50">
            <Title order={4} className="mb-3">Parameter Space Geometry</Title>
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <CodeBlock language="python" code={`import torch
import torch.nn as nn

# Simple 2-layer network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)  # 2*3 + 3 = 9 params
        self.layer2 = nn.Linear(3, 1)  # 3*1 + 1 = 4 params
        # Total: 13 parameters
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

net = SimpleNet()
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")

# Parameter vector representation
theta = torch.cat([p.flatten() for p in net.parameters()])
print(f"Parameter space dimension: {theta.shape[0]}")`} />
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Text className="mb-3"><strong>Key Concepts:</strong></Text>
                <List size="sm">
                  <List.Item><strong>High-dimensional spaces:</strong> Modern networks have millions of parameters</List.Item>
                  <List.Item><strong>Non-convex landscape:</strong> Multiple local minima exist</List.Item>
                  <List.Item><strong>Over-parameterization:</strong> More parameters than training samples</List.Item>
                  <List.Item><strong>Parameter sharing:</strong> CNNs reduce effective parameter count</List.Item>
                </List>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Loss Functions */}
        <div id="loss-functions">
          <Title order={2} className="mb-6">Loss Functions from Mathematical Perspective</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Loss Function Theory</Title>
            <Text className="mb-4">
              A loss function <InlineMath>{`\\mathcal{L}(y, \\hat{y})`}</InlineMath> measures the discrepancy between true labels <InlineMath>y</InlineMath> and predictions <InlineMath>{`\\hat{y}`}</InlineMath>. 
              The empirical risk is:
            </Text>
            <BlockMath>{`\\mathcal{R}_{\\text{emp}}(\\theta) = \\frac{1}{n} \\sum_{i=1}^n \\mathcal{L}(y_i, f(x_i; \\theta))`}</BlockMath>
            <Text className="mb-4">
              The goal of learning is to find <InlineMath>{`\\theta^* = \\arg\\min_{\\theta \\in \\Theta} \\mathcal{R}_{\\text{emp}}(\\theta)`}</InlineMath>
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} className="mb-3">Regression Losses</Title>
                  <CodeBlock language="python" code={`import torch
import torch.nn.functional as F

# Mean Squared Error (L2 Loss)
# L(y, ŷ) = (1/2)(y - ŷ)²
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([0.9, 2.1, 2.8])
mse = F.mse_loss(y_pred, y_true)
print(f"MSE: {mse:.4f}")

# Mean Absolute Error (L1 Loss)  
# L(y, ŷ) = |y - ŷ|
mae = F.l1_loss(y_pred, y_true)
print(f"MAE: {mae:.4f}")

# Huber Loss (robust to outliers)
# L(y, ŷ) = { (1/2)(y-ŷ)² if |y-ŷ| ≤ δ
#           { δ|y-ŷ| - (1/2)δ² otherwise
huber = F.huber_loss(y_pred, y_true, delta=1.0)
print(f"Huber: {huber:.4f}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-yellow-50">
                  <Title order={4} className="mb-3">Classification Losses</Title>
                  <CodeBlock language="python" code={`# Cross-Entropy Loss
# L(y, ŷ) = -Σₖ yₖ log(ŷₖ)
logits = torch.randn(3, 5)  # 3 samples, 5 classes
targets = torch.tensor([1, 3, 2])
ce_loss = F.cross_entropy(logits, targets)
print(f"Cross-Entropy: {ce_loss:.4f}")

# Binary Cross-Entropy
# L(y, ŷ) = -[y log(ŷ) + (1-y) log(1-ŷ)]
y_binary = torch.tensor([1.0, 0.0, 1.0])
y_prob = torch.tensor([0.8, 0.2, 0.9])
bce = F.binary_cross_entropy(y_prob, y_binary)
print(f"BCE: {bce:.4f}")

# Focal Loss (for imbalanced data)
# FL(p) = -α(1-p)ᵞ log(p)
def focal_loss(pred, target, alpha=1.0, gamma=2.0):
    ce = F.cross_entropy(pred, target, reduction='none')
    p = torch.exp(-ce)
    loss = alpha * (1 - p) ** gamma * ce
    return loss.mean()`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-purple-50">
            <Title order={4} className="mb-3">Loss Properties and Selection</Title>
            <Grid gutter="lg">
              <Grid.Col span={4}>
                <Title order={5} className="mb-2">Convexity</Title>
                <List size="sm">
                  <List.Item><strong>MSE:</strong> Convex in linear models</List.Item>
                  <List.Item><strong>Cross-Entropy:</strong> Convex in logistic regression</List.Item>
                  <List.Item><strong>Neural Networks:</strong> Non-convex due to composition</List.Item>
                </List>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Title order={5} className="mb-2">Robustness</Title>
                <List size="sm">
                  <List.Item><strong>L1 (MAE):</strong> Robust to outliers</List.Item>
                  <List.Item><strong>L2 (MSE):</strong> Sensitive to outliers</List.Item>
                  <List.Item><strong>Huber:</strong> Combines L1 and L2 benefits</List.Item>
                </List>
              </Grid.Col>
              
              <Grid.Col span={4}>
                <Title order={5} className="mb-2">Differentiability</Title>
                <List size="sm">
                  <List.Item><strong>MSE:</strong> Smooth everywhere</List.Item>
                  <List.Item><strong>MAE:</strong> Not differentiable at zero</List.Item>
                  <List.Item><strong>Cross-Entropy:</strong> Smooth for probabilities &gt; 0</List.Item>
                </List>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Gradient Descent */}
        <div id="gradient-descent">
          <Title order={2} className="mb-6">Gradient Descent Variants</Title>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Gradient Descent Mathematics</Title>
            <Text className="mb-4">
              Gradient descent updates parameters by moving in the direction of steepest descent:
            </Text>
            <BlockMath>{`\\theta_{t+1} = \\theta_t - \\eta \\nabla_{\\theta} \\mathcal{L}(\\theta_t)`}</BlockMath>
            <Text className="mb-4">
              where <InlineMath>{`\\eta > 0`}</InlineMath> is the learning rate and <InlineMath>{`\\nabla_{\\theta} \\mathcal{L}`}</InlineMath> is the gradient of the loss with respect to parameters.
            </Text>
            
            <CodeBlock language="python" code={`import torch
import matplotlib.pyplot as plt

# 1D quadratic function: f(x) = x²
def quadratic(x):
    return x**2

def quadratic_grad(x):
    return 2*x

# Gradient descent implementation
def gradient_descent(start_x, learning_rate, num_steps):
    x = start_x
    history = [x.item()]
    
    for step in range(num_steps):
        grad = quadratic_grad(x)
        x = x - learning_rate * grad
        history.append(x.item())
        
        print(f"Step {step+1}: x = {x:.4f}, f(x) = {quadratic(x):.4f}, grad = {grad:.4f}")
    
    return x, history

# Example: Start at x=2, lr=0.1, 10 steps
start = torch.tensor(2.0, requires_grad=True)
final_x, history = gradient_descent(start.clone(), 0.1, 5)`} />
          </Paper>

          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Batch Gradient Descent</Title>
                <Text className="mb-3"><strong>Update Rule:</strong> θₜ₊₁ = θₜ - η∇L(θₜ)</Text>
                <CodeBlock language="python" code={`# Full batch gradient descent
def batch_gradient_descent(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    
    # Accumulate gradients over entire dataset
    optimizer.zero_grad()
    
    for batch_x, batch_y in data_loader:
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()  # Accumulate gradients
        total_loss += loss.item()
    
    # Single update step
    optimizer.step()
    
    return total_loss / len(data_loader)`} />
                
                <Text size="sm" className="mt-3">
                  <strong>Properties:</strong> Stable convergence, expensive per iteration
                </Text>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">Stochastic Gradient Descent</Title>
                <Text className="mb-3"><strong>Update Rule:</strong> θₜ₊₁ = θₜ - η∇L(θₜ; xᵢ, yᵢ)</Text>
                <CodeBlock language="python" code={`# Stochastic gradient descent (SGD)
def sgd_step(model, batch_x, batch_y, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(batch_x)
    loss = loss_fn(output, batch_y)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    return loss.item()

# Usage with mini-batches
for batch_x, batch_y in data_loader:
    loss = sgd_step(model, batch_x, batch_y, optimizer, loss_fn)`} />
                
                <Text size="sm" className="mt-3">
                  <strong>Properties:</strong> Fast updates, noisy convergence, enables escape from local minima
                </Text>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Convergence Theory */}
        <div id="convergence-theory">
          <Title order={2} className="mb-6">Convergence Theory Basics</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Theoretical Foundations</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} className="mb-3">Convex Case</Title>
                  <Text className="mb-3">For convex functions with L-Lipschitz gradients:</Text>
                  <List size="sm">
                    <List.Item><strong>Learning Rate:</strong> η ≤ 1/L guarantees convergence</List.Item>
                    <List.Item><strong>Convergence Rate:</strong> O(1/t) for SGD</List.Item>
                    <List.Item><strong>Global Minimum:</strong> Always reachable</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Lipschitz constant estimation
def estimate_lipschitz(model, data_loader):
    """Estimate Lipschitz constant of gradients"""
    model.eval()
    max_grad_norm = 0
    
    for batch_x, batch_y in data_loader:
        batch_x.requires_grad_(True)
        output = model(batch_x)
        grad = torch.autograd.grad(
            outputs=output.sum(), 
            inputs=batch_x,
            create_graph=True
        )[0]
        grad_norm = grad.norm().item()
        max_grad_norm = max(max_grad_norm, grad_norm)
    
    return max_grad_norm`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-yellow-50">
                  <Title order={4} className="mb-3">Non-Convex Case (Neural Networks)</Title>
                  <Text className="mb-3">For neural networks (non-convex):</Text>
                  <List size="sm">
                    <List.Item><strong>Local Minima:</strong> Multiple solutions exist</List.Item>
                    <List.Item><strong>Saddle Points:</strong> More common in high dimensions</List.Item>
                    <List.Item><strong>Escape Mechanisms:</strong> SGD noise helps escape</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Detect critical points
def analyze_gradients(model, data_loader, threshold=1e-6):
    """Analyze gradient norms to detect critical points"""
    model.eval()
    grad_norms = []
    
    for batch_x, batch_y in data_loader:
        # Compute gradients
        loss = F.mse_loss(model(batch_x), batch_y)
        grads = torch.autograd.grad(loss, model.parameters())
        
        # Concatenate all gradients
        grad_norm = torch.cat([g.flatten() for g in grads]).norm()
        grad_norms.append(grad_norm.item())
    
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    
    if avg_grad_norm < threshold:
        print("Approaching critical point (local minimum or saddle)")
    
    return avg_grad_norm`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-purple-50">
            <Title order={4} className="mb-3">Practical Convergence Analysis</Title>
            <CodeBlock language="python" code={`import matplotlib.pyplot as plt
import torch.optim as optim

def analyze_convergence(model, train_loader, val_loader, num_epochs=100):
    """Analyze convergence properties during training"""
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []
    grad_norms = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        grad_norm_sum = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            grad_norm_sum += total_norm
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                val_loss += loss_fn(output, batch_y).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        grad_norms.append(grad_norm_sum / len(train_loader))
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.6f}, "
                  f"Val Loss={val_losses[-1]:.6f}, Grad Norm={grad_norms[-1]:.6f}")
    
    return train_losses, val_losses, grad_norms`} />
          </Paper>
        </div>

        {/* Summary */}
        <div>
          <Title order={2} className="mb-8">Summary: Mathematical Prerequisites</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} className="mb-4">Key Mathematical Concepts</Title>
                <List spacing="md">
                  <List.Item>Parameter spaces define the optimization landscape</List.Item>
                  <List.Item>Loss functions measure model performance mathematically</List.Item>
                  <List.Item>Gradient descent variants balance speed vs stability</List.Item>
                  <List.Item>Convergence theory guides learning rate selection</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} className="mb-4">Practical Implications</Title>
                <List spacing="md">
                  <List.Item>Over-parameterization can aid optimization despite theory</List.Item>
                  <List.Item>SGD noise helps escape local minima in neural networks</List.Item>
                  <List.Item>Loss function choice affects convergence behavior</List.Item>
                  <List.Item>Gradient analysis reveals training dynamics</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6 text-center">
            <Text size="lg" className="font-semibold">
              Mathematical rigor provides the foundation for understanding deep learning optimization
            </Text>
            <Text className="mt-2">
              These concepts bridge theory and practice in neural network training
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default MathematicalPrerequisites;