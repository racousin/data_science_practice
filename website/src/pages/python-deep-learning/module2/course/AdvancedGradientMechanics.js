import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const AdvancedGradientMechanics = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        <div id="gradient-flow">
          <Title order={1} mb="xl">
            Advanced Gradient Mechanics
          </Title>
          <Text size="xl" className="mb-6">
            Gradient Flow & Vanishing/Exploding Gradients
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} mb="md">Understanding Gradient Flow</Title>
            <Text className="mb-4">
              Gradient flow refers to how gradients propagate through network layers during backpropagation.
              Poor gradient flow can lead to vanishing or exploding gradients.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Vanishing Gradients</Title>
                  <CodeBlock language="python" code={`import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DeepNetwork(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.Sigmoid())  # Problematic activation
        layers.append(nn.Linear(100, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Analyze gradient magnitudes
model = DeepNetwork(num_layers=10)
x = torch.randn(32, 100)
y = torch.randn(32, 1)

output = model(x)
loss = ((output - y)**2).mean()
loss.backward()

# Check gradient norms by layer
grad_norms = []
for i, param in enumerate(model.parameters()):
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        print(f"Layer {i}: grad norm = {grad_norm:.6f}")

print(f"Gradient ratio (first/last): {grad_norms[0]/grad_norms[-1]:.2e}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} mb="sm">Exploding Gradients</Title>
                  <CodeBlock language="python" code={`# Demonstrate exploding gradients
class UnstableNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(20)
        ])
        # Initialize with large weights
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0, std=2.0)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x.sum()

model = UnstableNetwork()
x = torch.randn(1, 10)

# Monitor gradient explosion
try:
    for step in range(10):
        model.zero_grad()
        output = model(x)
        output.backward()
        
        # Check for gradient explosion
        total_norm = 0
        for param in model.parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        print(f"Step {step}: Gradient norm = {total_norm:.2e}")
        
        if total_norm > 1e6:
            print("Gradient explosion detected!")
            break
            
except RuntimeError as e:
    print(f"Runtime error: {e}")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        <div id="gradient-clipping">
          <Title order={2} mb="xl">Gradient Clipping & Normalization</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} mb="md">Gradient Clipping Techniques</Title>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} mb="sm">Norm-based Clipping</Title>
                  <CodeBlock language="python" code={`import torch.nn.utils as utils

# Gradient clipping by norm
def train_step_with_clipping(model, data, target, optimizer, max_norm=1.0):
    optimizer.zero_grad()
    
    output = model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Clip gradients
    utils.clip_grad_norm_(model.parameters(), max_norm)
    
    optimizer.step()
    return loss.item()

# Example usage
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Before clipping
model.zero_grad()
output = model(x)
loss = F.mse_loss(output, y)
loss.backward()
original_norm = utils.clip_grad_norm_(model.parameters(), float('inf'))

print(f"Original gradient norm: {original_norm:.4f}")

# With clipping
model.zero_grad()
output = model(x)
loss = F.mse_loss(output, y)
loss.backward()
clipped_norm = utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

print(f"Clipped gradient norm: {clipped_norm:.4f}")
print(f"Actual gradient norm after clipping: {utils.clip_grad_norm_(model.parameters(), float('inf')):.4f}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-yellow-50">
                  <Title order={4} mb="sm">Value-based Clipping</Title>
                  <CodeBlock language="python" code={`# Gradient clipping by value
def clip_grad_value_(parameters, clip_value):
    """Clip gradients by value"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    for param in parameters:
        if param.grad is not None:
            param.grad.data.clamp_(-clip_value, clip_value)

# Custom gradient clipping implementation
class GradientClipper:
    def __init__(self, clip_type='norm', clip_value=1.0):
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.gradient_history = []
    
    def clip(self, model):
        if self.clip_type == 'norm':
            total_norm = utils.clip_grad_norm_(model.parameters(), self.clip_value)
        elif self.clip_type == 'value':
            clip_grad_value_(model.parameters(), self.clip_value)
            total_norm = self._compute_grad_norm(model.parameters())
        
        self.gradient_history.append(total_norm)
        return total_norm
    
    def _compute_grad_norm(self, parameters):
        total_norm = 0
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def get_statistics(self):
        if not self.gradient_history:
            return {}
        return {
            'mean_norm': torch.tensor(self.gradient_history).mean().item(),
            'max_norm': max(self.gradient_history),
            'min_norm': min(self.gradient_history)
        }

# Usage
clipper = GradientClipper(clip_type='norm', clip_value=1.0)
for epoch in range(5):
    # ... training code ...
    grad_norm = clipper.clip(model)
    print(f"Epoch {epoch}: Gradient norm = {grad_norm:.4f}")

print("Gradient statistics:", clipper.get_statistics())`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        <div id="higher-order-derivatives">
          <Title order={2} mb="xl">Higher-order Derivatives & Hessians</Title>
          
          <Paper className="p-4 bg-purple-50">
            <Title order={4} mb="sm">Computing Second-order Derivatives</Title>
            <CodeBlock language="python" code={`# Higher-order derivatives in PyTorch
def compute_hessian(func, inputs):
    """Compute Hessian matrix for scalar function"""
    inputs = inputs.clone().detach().requires_grad_(True)
    
    # First derivative
    output = func(inputs)
    first_grads = torch.autograd.grad(
        outputs=output, 
        inputs=inputs, 
        create_graph=True
    )[0]
    
    # Second derivatives (Hessian)
    hessian = torch.zeros(inputs.size(0), inputs.size(0))
    
    for i in range(inputs.size(0)):
        second_grads = torch.autograd.grad(
            outputs=first_grads[i],
            inputs=inputs,
            retain_graph=True
        )[0]
        hessian[i] = second_grads
    
    return hessian

# Example: f(x) = x₁² + x₁x₂ + x₂²
def quadratic_function(x):
    return x[0]**2 + x[0]*x[1] + x[1]**2

x = torch.tensor([1.0, 2.0], requires_grad=True)
H = compute_hessian(quadratic_function, x)

print(f"Hessian matrix:\\n{H}")
print(f"Expected Hessian:\\n{torch.tensor([[2., 1.], [1., 2.]])}")`} />
          </Paper>
        </div>

        <div id="custom-backward-passes">
          <Title order={2} mb="xl">Custom Backward Passes</Title>
          
          <Paper className="p-4 bg-green-50">
            <Title order={4} mb="sm">Implementing Custom Autograd Functions</Title>
            <CodeBlock language="python" code={`# Custom autograd function
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
custom_relu = CustomReLU.apply

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
y = custom_relu(x)
loss = y.sum()
loss.backward()

print(f"Input: {x}")
print(f"Output: {y}")
print(f"Gradients: {x.grad}")

# Compare with built-in ReLU
x.grad.zero_()
y_builtin = torch.relu(x)
loss_builtin = y_builtin.sum()
loss_builtin.backward()

print(f"Built-in ReLU gradients: {x.grad}")
print(f"Match: {torch.allclose(y, y_builtin)}")`} />
          </Paper>
        </div>

        <div>
          <Title order={2} className="mb-8">Summary: Advanced Gradient Mechanics</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} mb="md">Gradient Flow Issues</Title>
                <List spacing="md">
                  <List.Item>Vanishing gradients prevent deep layer learning</List.Item>
                  <List.Item>Exploding gradients cause training instability</List.Item>
                  <List.Item>Activation functions significantly affect gradient flow</List.Item>
                  <List.Item>Weight initialization impacts gradient propagation</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} mb="md">Solutions and Tools</Title>
                <List spacing="md">
                  <List.Item>Gradient clipping prevents explosion</List.Item>
                  <List.Item>Better architectures (ResNet, LSTM) help flow</List.Item>
                  <List.Item>Custom autograd functions enable precise control</List.Item>
                  <List.Item>Higher-order derivatives enable advanced optimization</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default AdvancedGradientMechanics;