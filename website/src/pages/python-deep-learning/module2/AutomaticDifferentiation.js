import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Alert } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';

const AutomaticDifferentiation = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        {/* Introduction */}
        <div id="computation-graph">
          <Title order={1} mb="xl">
            Automatic Differentiation Deep Dive
          </Title>
          <Text size="xl" className="mb-6">
            Understanding PyTorch's Autograd Engine
          </Text>
          <Paper className="p-6 bg-gradient-to-r from-blue-50 to-purple-50 mb-6">
            <Text size="lg" mb="md">
              Automatic differentiation (AD) is the backbone of modern deep learning frameworks.
              It enables efficient computation of gradients through the chain rule, making
              backpropagation both automatic and computationally efficient.
            </Text>
            <BlockMath>
              {`\\frac{\\partial f}{\\partial x} = \\frac{\\partial f}{\\partial y} \\cdot \\frac{\\partial y}{\\partial x}`}
            </BlockMath>
          </Paper>
        </div>

        {/* Computation Graphs */}
        <div id="computation-graph">
          <Title order={2} mb="lg">Computation Graphs</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Dynamic vs Static Graphs</Title>
                <Text className="mb-3">
                  <strong>PyTorch (Dynamic):</strong> Computation graph is built on-the-fly during forward pass
                </Text>
                <Text className="mb-3">
                  <strong>TensorFlow 1.x (Static):</strong> Graph is defined first, then executed
                </Text>
                <CodeBlock language="python" code={`# PyTorch - Dynamic Graph
x = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    if i % 2 == 0:
        y = x * x
    else:
        y = x * x * x
    print(f"Iteration {i}: {y}")

# Graph structure can change each iteration!`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Building Computation Graphs</Title>
                <CodeBlock language="python" code={`import torch

# Create leaf nodes (inputs)
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Build computation graph
z = w * x  # Multiplication node
y = z + b  # Addition node  
loss = y ** 2  # Power node

print(f"Result: {loss}")
print(f"Graph structure:")
print(f"x.grad_fn: {x.grad_fn}")  # None (leaf)
print(f"z.grad_fn: {z.grad_fn}")  # MulBackward
print(f"y.grad_fn: {y.grad_fn}")  # AddBackward  
print(f"loss.grad_fn: {loss.grad_fn}")  # PowBackward`} />
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-yellow-50 mt-4">
            <Title order={4} mb="sm">Graph Visualization Example</Title>
            <Text className="mb-3">For the computation: <InlineMath>loss = (w \cdot x + b)^2</InlineMath></Text>
            <div className="text-center font-mono bg-white p-4 rounded">
              <div>x(3.0) ──┐</div>
              <div>          ├─→ * ──→ z(6.0) ──┐</div>
              <div>w(2.0) ──┘                   ├─→ + ──→ y(7.0) ──→ **2 ──→ loss(49.0)</div>
              <div>b(1.0) ──────────────────────┘</div>
            </div>
          </Paper>
        </div>

        {/* Gradient Computation */}
        <div id="gradient-computation">
          <Title order={2} mb="lg">Gradient Computation Process</Title>
          
          <Paper className="p-4 bg-purple-50 mb-4">
            <Title order={4} mb="sm">Forward and Backward Pass</Title>
            <CodeBlock language="python" code={`import torch

# Example: f(x) = 3x² + 2x + 1
x = torch.tensor(4.0, requires_grad=True)

# Forward pass - compute function value
y = 3 * x**2 + 2 * x + 1
print(f"f(4) = {y}")  # f(4) = 3(16) + 2(4) + 1 = 57

# Backward pass - compute gradient
y.backward()
print(f"f'(4) = {x.grad}")  # f'(x) = 6x + 2, so f'(4) = 26

# Manual verification
x_val = 4.0
manual_gradient = 6 * x_val + 2
print(f"Manual gradient: {manual_gradient}")  # Should match!`} />
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Chain Rule in Action</Title>
                <CodeBlock language="python" code={`# Composite function: h(x) = sin(x²)
x = torch.tensor(1.5, requires_grad=True)

# Forward pass
u = x**2        # u = x²
h = torch.sin(u) # h = sin(u)

print(f"x = {x}")
print(f"u = x² = {u}")  
print(f"h = sin(u) = {h}")

# Backward pass
h.backward()

# Chain rule: dh/dx = dh/du * du/dx
# dh/du = cos(u), du/dx = 2x
# So dh/dx = cos(x²) * 2x
manual_grad = torch.cos(u) * 2 * x
print(f"Computed gradient: {x.grad}")
print(f"Manual gradient: {manual_grad}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Multiple Variables</Title>
                <CodeBlock language="python" code={`# Function: f(x,y) = x²y + xy² + x + y
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass
z = x**2 * y + x * y**2 + x + y
print(f"f(2,3) = {z}")

# Backward pass
z.backward()

print(f"∂f/∂x = {x.grad}")  # 2xy + y² + 1
print(f"∂f/∂y = {y.grad}")  # x² + 2xy + 1

# Manual verification
# ∂f/∂x = 2xy + y² + 1 = 2(2)(3) + 3² + 1 = 12 + 9 + 1 = 22
# ∂f/∂y = x² + 2xy + 1 = 2² + 2(2)(3) + 1 = 4 + 12 + 1 = 17
print("Manual: ∂f/∂x = 22, ∂f/∂y = 17")`} />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Backward Pass Details */}
        <div id="backward-pass">
          <Title order={2} mb="lg">The Backward Pass in Detail</Title>
          
          <Paper className="p-4 bg-gray-50 mb-4">
            <Title order={4} mb="sm">Step-by-step Backward Computation</Title>
            <CodeBlock language="python" code={`# Example: y = (x + 2) * 3
x = torch.tensor(1.0, requires_grad=True)

# Forward pass (save intermediate values)
a = x + 2      # a = 3
y = a * 3      # y = 9

print(f"Forward: x={x}, a={a}, y={y}")

# Backward pass starts from output
# We want dy/dx, starting with dy/dy = 1
print("\\nBackward pass:")
print("dy/dy = 1 (starting point)")

# At multiplication node: y = a * 3
# dy/da = 3, dy/d3 = a = 3
print("At y = a * 3:")  
print("  dy/da = 3")

# At addition node: a = x + 2
# da/dx = 1, da/d2 = 1  
print("At a = x + 2:")
print("  da/dx = 1")

# Chain rule: dy/dx = dy/da * da/dx = 3 * 1 = 3
print("\\nFinal result: dy/dx = 3 * 1 = 3")

# Verify with automatic differentiation
y.backward()
print(f"Autograd result: {x.grad}")`} />
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} mb="sm">Gradient Functions</Title>
                <CodeBlock language="python" code={`# Each operation has a gradient function
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Build computation graph
z1 = x + y         # AddBackward
z2 = x * y         # MulBackward  
z3 = torch.sin(z2) # SinBackward
z4 = z3 ** 2       # PowBackward
loss = z1 + z4     # AddBackward

print("Gradient functions in the graph:")
print(f"z1.grad_fn: {z1.grad_fn}")
print(f"z2.grad_fn: {z2.grad_fn}")  
print(f"z3.grad_fn: {z3.grad_fn}")
print(f"z4.grad_fn: {z4.grad_fn}")
print(f"loss.grad_fn: {loss.grad_fn}")

# Each grad_fn knows how to compute gradients
loss.backward()
print(f"\\nGradients: x.grad={x.grad:.4f}, y.grad={y.grad:.4f}")`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper p="md">
                <Title order={4} mb="sm">Gradient Flow Visualization</Title>
                <div className="font-mono text-sm bg-white p-3 rounded">
                  <div>Forward Pass:</div>
                  <div>x(2) ──┐</div>
                  <div>        ├─→ +(5) ──────────┐</div>
                  <div>y(3) ──┘                   ├─→ +(loss)</div>
                  <div>x(2) ──┐                   │</div>
                  <div>        ├─→ *(6) ──→ sin ──┴─→ **2</div>
                  <div>y(3) ──┘</div>
                  <br/>
                  <div>Backward Pass (gradients flow backward):</div>
                  <div>∂L/∂x ←─┐</div>
                  <div>          ├─← + ←──────────┐</div>
                  <div>∂L/∂y ←─┘                   ├─← +</div>
                  <div>∂L/∂x ←─┐                   │</div>
                  <div>          ├─← * ←─ sin ←─┴─← **2</div>
                  <div>∂L/∂y ←─┘</div>
                </div>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Gradient Accumulation */}
        <div id="gradient-accumulation">
          <Title order={2} mb="lg">Gradient Accumulation</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} mb="sm">Why Gradients Accumulate</Title>
                <CodeBlock language="python" code={`# Gradients accumulate by default!
x = torch.tensor(3.0, requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(f"After first backward: {x.grad}")  # 6.0

# Second computation (gradients add up!)
y2 = x * 5
y2.backward()
print(f"After second backward: {x.grad}")  # 6.0 + 5.0 = 11.0

# This is often NOT what we want in training!
# Need to zero gradients between iterations
x.grad.zero_()  # Clear gradients
print(f"After zero_(): {x.grad}")  # 0.0

# Now try again
y3 = x ** 3
y3.backward() 
print(f"Fresh gradient: {x.grad}")  # 27.0`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Training Loop Pattern</Title>
                <CodeBlock language="python" code={`import torch.optim as optim

# Typical training loop structure
model_params = torch.tensor(1.0, requires_grad=True)
optimizer = optim.SGD([model_params], lr=0.01)

for epoch in range(3):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # or manually: model_params.grad.zero_()
    
    # Forward pass
    prediction = model_params ** 2
    target = torch.tensor(4.0)
    loss = (prediction - target) ** 2
    
    print(f"Epoch {epoch}: Loss = {loss:.3f}")
    
    # Backward pass
    loss.backward()
    print(f"  Gradient: {model_params.grad:.3f}")
    
    # Update parameters
    optimizer.step()
    print(f"  New param: {model_params:.3f}")
    
    # DON'T FORGET: optimizer.zero_grad() at the start!`} />
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Alert color="orange" className="mt-4">
            <Title order={4} className="mb-2">⚠️ Common Mistake</Title>
            <Text>
              Forgetting to zero gradients between training iterations is one of the most common
              bugs in PyTorch code. Always call <code>optimizer.zero_grad()</code> or 
              <code>param.grad.zero_()</code> before each backward pass in training!
            </Text>
          </Alert>
        </div>

        {/* Higher-order Gradients */}
        <div id="higher-order-gradients">
          <Title order={2} mb="lg">Higher-order Gradients</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} mb="sm">Second Derivatives</Title>
                <CodeBlock language="python" code={`# Computing second derivatives
x = torch.tensor(2.0, requires_grad=True)

# f(x) = x³
y = x ** 3

# First derivative: f'(x) = 3x²
grad_first = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"First derivative at x=2: {grad_first}")  # 12.0

# Second derivative: f''(x) = 6x
grad_second = torch.autograd.grad(grad_first, x)[0]
print(f"Second derivative at x=2: {grad_second}")  # 12.0

# Manual verification:
# f(x) = x³, f'(x) = 3x², f''(x) = 6x
# At x=2: f'(2) = 3(4) = 12, f''(2) = 6(2) = 12 ✓`} />
              </Paper>
              
              <Paper className="p-4 bg-blue-50 mt-4">
                <Title order={4} mb="sm">Hessian Matrix</Title>
                <CodeBlock language="python" code={`# Hessian for multivariate functions
def compute_hessian(f, vars):
    """Compute Hessian matrix of scalar function f w.r.t. vars"""
    grad = torch.autograd.grad(f, vars, create_graph=True)
    hessian = []
    for g in grad:
        row = []
        for v in vars:
            row.append(torch.autograd.grad(g, v, retain_graph=True)[0])
        hessian.append(row)
    return torch.stack([torch.stack(row) for row in hessian])

# Example: f(x,y) = x²y + xy²
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
f = x**2 * y + x * y**2

hessian = compute_hessian(f, [x, y])
print(f"Hessian matrix:\\n{hessian}")
# [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]]`} />
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} mb="sm">Practical Applications</Title>
                <List spacing="sm" className="mb-4">
                  <List.Item><strong>Newton's Optimization:</strong> Uses second derivatives for faster convergence</List.Item>
                  <List.Item><strong>Adversarial Training:</strong> Computing gradients of gradients</List.Item>
                  <List.Item><strong>Meta-Learning:</strong> Gradient-based meta-learning algorithms</List.Item>
                  <List.Item><strong>Physics-Informed NNs:</strong> Solving differential equations</List.Item>
                </List>
                
                <CodeBlock language="python" code={`# Example: Physics-Informed Neural Network
# Solving: d²u/dx² + π²u = 0, u(0)=0, u(1)=0
# Analytical solution: u(x) = sin(πx)

x = torch.linspace(0, 1, 100, requires_grad=True)

# Neural network approximation
u = torch.sin(torch.pi * x)  # Our NN would predict this

# Compute derivatives
u_x = torch.autograd.grad(
    u.sum(), x, create_graph=True)[0]
u_xx = torch.autograd.grad(
    u_x.sum(), x, create_graph=True)[0]

# Physics loss: ||d²u/dx² + π²u||²
physics_loss = ((u_xx + torch.pi**2 * u)**2).mean()
print(f"Physics residual: {physics_loss:.6f}")

# Should be near zero for exact solution!`} />
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Summary */}
        <div>
          <Title order={2} className="mb-8">Summary: Automatic Differentiation</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 h-full">
                <Title order={2} mb="lg"> Key Concepts</Title>
                <List spacing="md">
                  <List.Item>Dynamic computation graphs in PyTorch</List.Item>
                  <List.Item>Forward pass builds graph, backward pass computes gradients</List.Item>
                  <List.Item>Chain rule enables efficient gradient computation</List.Item>
                  <List.Item>Gradients accumulate by default - must zero between iterations</List.Item>
                  <List.Item>Higher-order derivatives for advanced applications</List.Item>
                  <List.Item>Custom functions and hooks for specialized needs</List.Item>
                </List>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-6 bg-gradient-to-br from-green-50 to-green-100 h-full">
                <Title order={2} mb="lg">Practical Impact</Title>
                <List spacing="md">
                  <List.Item>Makes deep learning accessible and debuggable</List.Item>
                  <List.Item>Enables complex architectures with gradient flow</List.Item>
                  <List.Item>Supports advanced techniques like meta-learning</List.Item>
                  <List.Item>Facilitates research in differentiable programming</List.Item>
                  <List.Item>Powers all modern deep learning frameworks</List.Item>
                  <List.Item>Foundation for optimization and training algorithms</List.Item>
                </List>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-6 bg-gradient-to-r from-purple-50 to-pink-50 mt-6 text-center">
            <Text size="lg" className="font-semibold">
              Automatic differentiation is the mathematical engine that makes deep learning possible!
            </Text>
            <Text className="mt-2">
              Understanding autograd gives you the power to debug, optimize, and innovate in deep learning.
            </Text>
          </Paper>
        </div>

      </Stack>
    </Container>
  );
};

export default AutomaticDifferentiation;