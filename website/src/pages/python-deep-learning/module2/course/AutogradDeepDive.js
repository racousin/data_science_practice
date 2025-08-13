import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const AutogradDeepDive = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        {/* Forward and Reverse Mode */}
        <div id="forward-reverse-mode">
          <Title order={1} className="mb-6">
            Autograd Deep Dive
          </Title>
          <Text size="xl" className="mb-6">
            Forward & Reverse Mode Automatic Differentiation
          </Text>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Automatic Differentiation Theory</Title>
            <Text className="mb-4">
              Automatic differentiation computes derivatives by applying the chain rule to elementary operations. 
              There are two main modes: forward mode and reverse mode (backpropagation).
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Forward Mode AD</Title>
                  <Text className="mb-3">Computes derivatives alongside forward computation:</Text>
                  <List size="sm">
                    <List.Item>Propagates derivatives from inputs to outputs</List.Item>
                    <List.Item>Efficient for functions with few inputs, many outputs</List.Item>
                    <List.Item>Each variable carries its derivative (dual number)</List.Item>
                    <List.Item>Cost: O(n) for n input variables</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Forward mode example for f(x) = x²
class DualNumber:
    def __init__(self, value, derivative=0):
        self.value = value
        self.derivative = derivative
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # Product rule: (uv)' = u'v + uv'
            return DualNumber(
                self.value * other.value,
                self.derivative * other.value + self.value * other.derivative
            )
        else:
            return DualNumber(self.value * other, self.derivative * other)

# Compute f(x) = x² and f'(x) at x=3
x = DualNumber(3, 1)  # value=3, derivative=1 (dx/dx = 1)
result = x * x
print(f"f(3) = {result.value}")      # 9
print(f"f'(3) = {result.derivative}") # 6`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Reverse Mode AD (Backpropagation)</Title>
                  <Text className="mb-3">Computes derivatives by traversing computation graph backwards:</Text>
                  <List size="sm">
                    <List.Item>Forward pass computes function values</List.Item>
                    <List.Item>Backward pass computes derivatives</List.Item>
                    <List.Item>Efficient for functions with many inputs, few outputs</List.Item>
                    <List.Item>Cost: O(m) for m output variables</List.Item>
                  </List>
                  
                  <CodeBlock language="python" code={`# Reverse mode example
import torch

def f(x):
    """Function: f(x) = x²"""
    return x ** 2

# Forward pass
x = torch.tensor(3.0, requires_grad=True)
y = f(x)
print(f"Forward: f(3) = {y}")

# Backward pass
y.backward()  # Computes dy/dx
print(f"Backward: df/dx = {x.grad}")

# For more complex functions
x.grad.zero_()  # Reset gradient
z = torch.sin(x**2) + torch.exp(-x) + x**3
z.backward()
print(f"Complex function gradient: {x.grad}")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Computational Graph Construction */}
        <div id="computational-graph-construction">
          <Title order={2} className="mb-6">Computational Graph Construction</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Dynamic Graph Building in PyTorch</Title>
            <Text className="mb-4">
              PyTorch builds computational graphs dynamically during the forward pass. Each operation creates nodes 
              that track the operations and their inputs for gradient computation.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} className="mb-3">Graph Node Structure</Title>
                  <CodeBlock language="python" code={`import torch

# Create leaf tensors (inputs)
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

print("Leaf tensors:")
print(f"x.is_leaf: {x.is_leaf}")
print(f"x.grad_fn: {x.grad_fn}")  # None for leaf tensors

# Intermediate operations create nodes
z1 = x ** 2                    # PowBackward
z2 = torch.sin(y)             # SinBackward  
z3 = z1 + z2                  # AddBackward

print("\\nIntermediate tensors:")
print(f"z1.grad_fn: {z1.grad_fn}")
print(f"z2.grad_fn: {z2.grad_fn}")
print(f"z3.grad_fn: {z3.grad_fn}")

# Final output
loss = z3 * 2                 # MulBackward
print(f"\\nloss.grad_fn: {loss.grad_fn}")
print(f"loss.grad_fn.next_functions: {loss.grad_fn.next_functions}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-yellow-50">
                  <Title order={4} className="mb-3">Graph Traversal and Dependencies</Title>
                  <CodeBlock language="python" code={`# Visualize graph dependencies
def print_graph(tensor, depth=0):
    """Recursively print computation graph structure"""
    indent = "  " * depth
    if tensor.grad_fn is None:
        print(f"{indent}Leaf: {tensor}")
    else:
        print(f"{indent}{type(tensor.grad_fn).__name__}")
        for next_function, input_idx in tensor.grad_fn.next_functions:
            if next_function is not None:
                print(f"{indent}  Input {input_idx}:")
                # This is simplified - actual implementation would need 
                # to track tensors associated with functions

# Example with complex function
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# f(x,y) = (x² + y³) * sin(x*y)
term1 = x**2 + y**3
term2 = torch.sin(x * y)
result = term1 * term2

print("Computation graph for (x² + y³) * sin(x*y):")
print(f"Result grad_fn: {result.grad_fn}")
print("Graph structure:")
print_graph(result)`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-purple-50">
            <Title order={4} className="mb-3">Graph Memory Management</Title>
            <CodeBlock language="python" code={`# Graph lifecycle and memory management
import torch

def analyze_graph_memory():
    x = torch.randn(1000, 1000, requires_grad=True)
    y = torch.randn(1000, 1000, requires_grad=True)
    
    # Forward pass creates graph nodes
    z = x @ y  # Matrix multiplication
    loss = z.sum()
    
    print(f"Graph exists: {loss.grad_fn is not None}")
    print(f"Intermediate tensor z retains graph: {z.grad_fn is not None}")
    
    # Backward pass
    loss.backward()
    
    print(f"After backward - x.grad shape: {x.grad.shape}")
    print(f"Graph still exists: {loss.grad_fn is not None}")
    
    # Graph is freed automatically after backward pass
    # To retain graph for multiple backward passes:
    x.grad.zero_()
    z = x @ y
    loss = z.sum()
    loss.backward(retain_graph=True)  # Keep graph alive
    
    print(f"With retain_graph=True: {loss.grad_fn is not None}")
    
    # Second backward pass possible
    loss.backward()  # This works because graph was retained

analyze_graph_memory()`} />
          </Paper>
        </div>

        {/* Chain Rule and Backpropagation */}
        <div id="chain-rule-backpropagation">
          <Title order={2} className="mb-6">Chain Rule & Backpropagation Mathematics</Title>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} className="mb-4">Mathematical Foundation</Title>
            <Text className="mb-4">
              The chain rule states that for composite functions f(g(x)), the derivative is:
              (f ∘ g)'(x) = f'(g(x)) · g'(x)
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Scalar Chain Rule</Title>
                  <CodeBlock language="python" code={`import torch

# Example: f(x) = sin(x²)
# df/dx = cos(x²) * 2x

x = torch.tensor(1.0, requires_grad=True)

# Forward pass
u = x ** 2        # u = x²
y = torch.sin(u)  # y = sin(u) = sin(x²)

print(f"x = {x.item()}")
print(f"u = {u.item()}")  
print(f"y = {y.item()}")

# Backward pass
y.backward()

# Manual calculation for verification
x_val = x.item()
expected_grad = torch.cos(torch.tensor(x_val**2)) * 2 * x_val

print(f"\\nGradients:")
print(f"Computed dy/dx: {x.grad}")
print(f"Expected dy/dx: {expected_grad}")
print(f"Match: {torch.allclose(x.grad, expected_grad)}")`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-white">
                  <Title order={4} className="mb-3">Vector Chain Rule</Title>
                  <CodeBlock language="python" code={`# Multivariable chain rule
# f(x,y) = x*y + x²
# ∂f/∂x = y + 2x, ∂f/∂y = x

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)

# Function: f = sum(x*y + x²)
z = x * y + x**2
loss = z.sum()

print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")
print(f"loss: {loss}")

# Backward pass
loss.backward()

print(f"\\nGradients:")
print(f"x.grad: {x.grad}")  # Should be [y[0] + 2*x[0], y[1] + 2*x[1]]
print(f"y.grad: {y.grad}")  # Should be [x[0], x[1]]

# Manual verification
expected_x_grad = y + 2*x
expected_y_grad = x
print(f"\\nExpected x.grad: {expected_x_grad}")
print(f"Expected y.grad: {expected_y_grad}")`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-green-50">
            <Title order={4} className="mb-3">Jacobian and Vector-Jacobian Products</Title>
            <CodeBlock language="python" code={`# Understanding Jacobian matrices and VJPs
import torch

def compute_jacobian(func, inputs):
    """Compute Jacobian matrix for vector-valued function"""
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = func(inputs)
    
    jacobian = torch.zeros(outputs.shape[0], inputs.shape[0])
    
    for i in range(outputs.shape[0]):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i] = 1.0
        
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=True
        )[0]
        
        jacobian[i] = grads
    
    return jacobian

# Example: f(x) = [x₁², x₁x₂, x₂²]
def vector_function(x):
    return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

x = torch.tensor([2.0, 3.0])
J = compute_jacobian(vector_function, x)

print(f"Input: {x}")
print(f"Jacobian matrix:\\n{J}")

# Expected Jacobian:
# ∂f₁/∂x₁ = 2x₁, ∂f₁/∂x₂ = 0
# ∂f₂/∂x₁ = x₂,  ∂f₂/∂x₂ = x₁  
# ∂f₃/∂x₁ = 0,   ∂f₃/∂x₂ = 2x₂
expected_J = torch.tensor([[2*x[0], 0], [x[1], x[0]], [0, 2*x[1]]])
print(f"Expected Jacobian:\\n{expected_J}")
print(f"Match: {torch.allclose(J, expected_J)}")`} />
          </Paper>
        </div>

        {/* Gradient Accumulation */}
        <div id="gradient-accumulation">
          <Title order={2} className="mb-6">Gradient Accumulation & Zeroing</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Title order={3} className="mb-4">Gradient Lifecycle Management</Title>
            <Text className="mb-4">
              PyTorch accumulates gradients in the .grad attribute of tensors. Understanding when to zero gradients 
              is crucial for correct training.
            </Text>
            
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Paper className="p-4 bg-blue-50">
                  <Title order={4} className="mb-3">Gradient Accumulation</Title>
                  <CodeBlock language="python" code={`import torch

# Gradient accumulation example
x = torch.tensor(2.0, requires_grad=True)

print("Initial state:")
print(f"x.grad: {x.grad}")

# First backward pass
y1 = x ** 2
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

# Second backward pass (accumulates!)
y2 = x ** 3  
y2.backward()
print(f"After second backward: x.grad = {x.grad}")

# Expected: grad = 4 + 12 = 16
# dy1/dx = 2x = 4, dy2/dx = 3x² = 12

# Manual verification
expected_total_grad = 2*x + 3*x**2
print(f"Expected total gradient: {expected_total_grad}")

# Zero gradients for fresh start
x.grad.zero_()
print(f"After zeroing: x.grad = {x.grad}")

# Third backward pass
y3 = x ** 4
y3.backward()
print(f"Fresh backward: x.grad = {x.grad}")  # Should be 4x³ = 32`} />
                </Paper>
              </Grid.Col>
              
              <Grid.Col span={6}>
                <Paper className="p-4 bg-yellow-50">
                  <Title order={4} className="mb-3">Practical Gradient Management</Title>
                  <CodeBlock language="python" code={`# Practical gradient management in training loops
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Sample data
x_data = torch.randn(10, 2)
y_data = torch.randn(10, 1)

print("Training loop with proper gradient management:")

for epoch in range(3):
    # CRITICAL: Zero gradients before backward pass
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(x_data)
    loss = loss_fn(predictions, y_data)
    
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Check gradients before backward pass
    print(f"  Before backward: linear.weight.grad = {model.linear.weight.grad}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients after backward pass
    print(f"  After backward: linear.weight.grad norm = {model.linear.weight.grad.norm():.4f}")
    
    # Update parameters
    optimizer.step()

# Common mistake: forgetting to zero gradients
print("\\nCommon mistake - no gradient zeroing:")
for epoch in range(2):
    # Missing: optimizer.zero_grad()
    predictions = model(x_data)
    loss = loss_fn(predictions, y_data)
    loss.backward()
    
    grad_norm = model.linear.weight.grad.norm()
    print(f"Epoch {epoch}: Grad norm = {grad_norm:.4f} (accumulating!)")
    optimizer.step()`} />
                </Paper>
              </Grid.Col>
            </Grid>
          </Paper>

          <Paper className="p-4 bg-purple-50">
            <Title order={4} className="mb-3">Advanced Gradient Manipulation</Title>
            <CodeBlock language="python" code={`# Advanced gradient manipulation techniques
import torch

# 1. Selective gradient computation
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Compute gradients only for x
z = (x * y).sum()
z.backward(inputs=[x])  # Only compute gradients for x

print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")  # None because not computed

# 2. Custom gradient values
x.grad.zero_()
y.requires_grad_(True)

z = (x ** 2 + y ** 2).sum()
z.backward()

print(f"\\nOriginal gradients:")
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

# 3. Gradient clipping
torch.nn.utils.clip_grad_norm_([x, y], max_norm=1.0)

print(f"\\nAfter gradient clipping:")
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

# 4. Gradient modification with hooks
def gradient_hook(grad):
    print(f"Gradient hook called: {grad}")
    return grad * 2  # Double the gradient

x.register_hook(gradient_hook)

x.grad.zero_()
y.grad.zero_()
z = (x ** 3).sum()
z.backward()

print(f"\\nWith gradient hook (doubled):")
print(f"x.grad: {x.grad}")  # Should be doubled`} />
          </Paper>
        </div>

        {/* Summary */}
        <div>
          <Title order={2} className="mb-8">Summary: Autograd Deep Dive</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={3} className="mb-4">Key Concepts</Title>
                <List spacing="md">
                  <List.Item>Forward mode AD: efficient for few inputs, many outputs</List.Item>
                  <List.Item>Reverse mode AD (backprop): efficient for many inputs, few outputs</List.Item>
                  <List.Item>Dynamic computational graphs enable flexible architectures</List.Item>
                  <List.Item>Chain rule enables gradient computation through compositions</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={3} className="mb-4">Practical Implications</Title>
                <List spacing="md">
                  <List.Item>Always zero gradients before backward pass in training</List.Item>
                  <List.Item>Gradient accumulation enables large effective batch sizes</List.Item>
                  <List.Item>Graph memory management affects training efficiency</List.Item>
                  <List.Item>Custom gradient manipulation enables advanced techniques</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6 text-center">
            <Text size="lg" className="font-semibold">
              Mastering autograd is essential for understanding and debugging deep learning models
            </Text>
            <Text className="mt-2">
              These concepts form the mathematical foundation of all neural network training
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default AutogradDeepDive;