import React from 'react';
import { Container, Title, Text, Space, List, Flex, Image, Paper, Alert } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const AutogradTorchPerspective = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">Autograd: PyTorch's Automatic Differentiation Engine</Title>
      
      <Text>
        In the previous section, we explored the mathematical foundations of backpropagation. Now, let's see how PyTorch's 
        autograd engine implements these concepts automatically, tracking operations and computing gradients for us.
      </Text>

      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module2/autograd-overview.png"
          alt="Autograd Overview"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
      </Flex>

      <Title order={2} mt="xl">1. Basic Autograd Mechanics</Title>
      
      <Title order={3} mt="md">Understanding requires_grad</Title>
      
      <Text>
        Every tensor in PyTorch has a <code>requires_grad</code> attribute that tells autograd whether to track operations on it:
      </Text>
      
      <CodeBlock language="python" code={`import torch

# By default, tensors don't track gradients
a = torch.tensor(2.0)
print(f"requires_grad: {a.requires_grad}")  # False`} />

      <Text mt="md">
        To enable gradient tracking, set <code>requires_grad=True</code>:
      </Text>

      <CodeBlock language="python" code={`# Enable gradient tracking
b = torch.tensor(3.0, requires_grad=True)
print(f"requires_grad: {b.requires_grad}")  # True`} />

      <Title order={3} mt="xl">The Computational Graph</Title>
      
      <Text>
        When we perform operations on tensors with <code>requires_grad=True</code>, PyTorch builds a computational graph:
      </Text>

      <CodeBlock language="python" code={`# Create input tensors
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)`} />

      <Text mt="md">
        Each operation creates a new node in the graph:
      </Text>

      <CodeBlock language="python" code={`# First operation: addition
z = x + y  # z = 2 + 3 = 5
print(f"z = {z.item()}")
print(f"z.grad_fn: {z.grad_fn}")  # <AddBackward0>`} />

      <Text mt="md">
        The <code>grad_fn</code> attribute stores the function that created this tensor. It will be used during backpropagation 
        to compute gradients.
      </Text>

      <CodeBlock language="python" code={`# Chain more operations
w = z * z  # w = 5 * 5 = 25
print(f"w = {w.item()}")
print(f"w.grad_fn: {w.grad_fn}")  # <MulBackward0>`} />

      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module2/computation-graph.png"
          alt="Computation Graph"
          style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
          fluid
        />
      </Flex>

      <Title order={3} mt="xl">Leaf Tensors vs Intermediate Tensors</Title>
      
      <Text>
        PyTorch distinguishes between <strong>leaf tensors</strong> (inputs we created) and <strong>intermediate tensors</strong> 
        (results of operations):
      </Text>

      <CodeBlock language="python" code={`print(f"x is leaf: {x.is_leaf}")  # True - we created it
print(f"z is leaf: {z.is_leaf}")  # False - result of x + y`} />

      <Text mt="md">
        Only leaf tensors with <code>requires_grad=True</code> will have their gradients accumulated:
      </Text>

      <CodeBlock language="python" code={`# Compute gradients
w.backward()

print(f"x.grad: {x.grad}")  # tensor(10.) = dw/dx
print(f"y.grad: {y.grad}")  # tensor(10.) = dw/dy
print(f"z.grad: {z.grad}")  # None - intermediate tensor`} />

      <Title order={2} mt="xl">2. Understanding grad_fn Objects</Title>
      
      <Title order={3} mt="md">Each Operation Creates a Function Object</Title>
      
      <Text>
        Every operation in PyTorch has a corresponding backward function that knows how to compute gradients:
      </Text>

      <CodeBlock language="python" code={`# Clear previous gradients
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)`} />

      <Text mt="md">
        <strong>Addition</strong> creates an <code>AddBackward</code> object:
      </Text>

      <CodeBlock language="python" code={`z_add = x + y
print(f"Operation: {z_add.grad_fn}")  # <AddBackward0>
# This function knows: d(x+y)/dx = 1, d(x+y)/dy = 1`} />

      <Text mt="md">
        <strong>Multiplication</strong> creates a <code>MulBackward</code> object:
      </Text>

      <CodeBlock language="python" code={`z_mul = x * y  
print(f"Operation: {z_mul.grad_fn}")  # <MulBackward0>
# This function knows: d(x*y)/dx = y, d(x*y)/dy = x`} />

      <Text mt="md">
        <strong>Trigonometric functions</strong> have their own backward functions:
      </Text>

      <CodeBlock language="python" code={`z_sin = torch.sin(x)
print(f"Operation: {z_sin.grad_fn}")  # <SinBackward0>
# This function knows: d(sin(x))/dx = cos(x)`} />

      <Text mt="md">
        <strong>Exponential function</strong>:
      </Text>

      <CodeBlock language="python" code={`z_exp = torch.exp(x)
print(f"Operation: {z_exp.grad_fn}")  # <ExpBackward0>
# This function knows: d(exp(x))/dx = exp(x)`} />

      <Title order={3} mt="xl">Exploring the Graph Structure</Title>
      
      <Text>
        Each <code>grad_fn</code> has a <code>next_functions</code> attribute linking to its inputs' gradient functions:
      </Text>

      <CodeBlock language="python" code={`# Create a simple computation
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b
d = c + a
output = d ** 2`} />

      <Text mt="md">
        Let's trace the graph backwards from the output:
      </Text>

      <CodeBlock language="python" code={`print(f"output.grad_fn: {output.grad_fn}")  # <PowBackward0>

# Look at inputs to the power operation
for fn, _ in output.grad_fn.next_functions:
    print(f"  Input: {fn}")  # <AddBackward0>`} />

      <Text mt="md">
        This shows how PyTorch maintains the complete computational graph for backpropagation.
      </Text>

      <Title order={2} mt="xl">3. Computing Gradients Step by Step</Title>
      
      <Title order={3} mt="md">The backward() Method</Title>
      
      <Text>
        Let's build a simple function and compute its gradients manually to understand what autograd does:
      </Text>

      <Text mt="md">
        Consider the function: <InlineMath>{`f(x, y) = (x + y)^2`}</InlineMath>
      </Text>

      <CodeBlock language="python" code={`# Define inputs
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass
z = x + y  # z = 5
f = z ** 2  # f = 25`} />

      <Text mt="md">
        Mathematically, the gradients are:
      </Text>
      <BlockMath>{`\\frac{\\partial f}{\\partial x} = 2(x + y) = 10, \\quad \\frac{\\partial f}{\\partial y} = 2(x + y) = 10`}</BlockMath>

      <Text mt="md">
        Let's verify with autograd:
      </Text>

      <CodeBlock language="python" code={`# Compute gradients
f.backward()

print(f"df/dx = {x.grad}")  # tensor(10.)
print(f"df/dy = {y.grad}")  # tensor(10.)`} />

      <Title order={3} mt="xl">How Backpropagation Works in Autograd</Title>

      <Text>
        Let's trace through what happens during <code>backward()</code> for a more complex function:
      </Text>

      <Text mt="md">
        <InlineMath>{`f(x) = \\sin(x^2)`}</InlineMath>
      </Text>

      <CodeBlock language="python" code={`x = torch.tensor(1.0, requires_grad=True)

# Forward pass - building the graph
u = x ** 2        # u = x²
y = torch.sin(u)  # y = sin(u) = sin(x²)`} />

      <Text mt="md">
        During backward pass, autograd applies the chain rule:
      </Text>

      <BlockMath>{`\\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx} = \\cos(u) \\cdot 2x = \\cos(x^2) \\cdot 2x`}</BlockMath>

      <CodeBlock language="python" code={`# Backward pass
y.backward()

# Verify manually
x_val = x.item()
expected = torch.cos(torch.tensor(x_val**2)) * 2 * x_val
print(f"Computed gradient: {x.grad}")
print(f"Expected gradient: {expected}")`} />

      <Title order={3} mt="xl">Gradient Accumulation</Title>
      
      <Text>
        By default, PyTorch <strong>accumulates</strong> gradients. This means calling <code>backward()</code> multiple times 
        adds to existing gradients:
      </Text>

      <CodeBlock language="python" code={`x = torch.tensor(2.0, requires_grad=True)

# First backward pass
y1 = x ** 2
y1.backward()
print(f"After first backward: x.grad = {x.grad}")  # 4.0`} />

      <CodeBlock language="python" code={`# Second backward pass (accumulates!)
y2 = x ** 3
y2.backward()
print(f"After second backward: x.grad = {x.grad}")  # 4.0 + 12.0 = 16.0`} />

      <Text mt="md">
        Always zero gradients before a new backward pass:
      </Text>

      <CodeBlock language="python" code={`x.grad.zero_()  # Clear the gradient
y3 = x ** 2
y3.backward()
print(f"After zeroing: x.grad = {x.grad}")  # 4.0`} />

      <Title order={2} mt="xl">4. Memory Management and Graph Retention</Title>
      
      <Title order={3} mt="md">Computational Graph Lifetime</Title>
      
      <Text>
        By default, the computational graph is freed after calling <code>backward()</code> to save memory:
      </Text>

      <CodeBlock language="python" code={`x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# First backward works
y.backward()
print(f"First backward: x.grad = {x.grad}")  # 4.0`} />

      <CodeBlock language="python" code={`# Second backward fails!
try:
    y.backward()
except RuntimeError as e:
    print(f"Error: {e}")
    # Error: Trying to backward through the graph a second time`} />

      <Text mt="md">
        To reuse the graph, use <code>retain_graph=True</code>:
      </Text>

      <CodeBlock language="python" code={`x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# Retain the graph
y.backward(retain_graph=True)
print(f"First backward: x.grad = {x.grad}")  # 4.0

x.grad.zero_()
y.backward()  # Works now!
print(f"Second backward: x.grad = {x.grad}")  # 4.0`} />

      <Title order={3} mt="xl">Detaching from the Graph</Title>
      
      <Text>
        Sometimes we want to use a tensor's value without tracking gradients. Use <code>detach()</code>:
      </Text>

      <CodeBlock language="python" code={`x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# Detach y from the graph
y_detached = y.detach()
print(f"y_detached.requires_grad: {y_detached.requires_grad}")  # False`} />

      <Text mt="md">
        This is useful when you want to use computed values as constants:
      </Text>

      <CodeBlock language="python" code={`# y_detached is treated as a constant
z = y_detached * x  # Only tracks gradient w.r.t. x
z.backward()
print(f"x.grad: {x.grad}")  # 4.0 (not 8.0!)`} />

      <Title order={2} mt="xl">5. Gradient Flow Control with Hooks</Title>
      
      <Title order={3} mt="md">Registering Backward Hooks</Title>
      
      <Text>
        Hooks allow us to inspect or modify gradients during backpropagation. This is powerful for debugging and understanding 
        gradient flow:
      </Text>

      <CodeBlock language="python" code={`x = torch.tensor([2.0, 3.0], requires_grad=True)

# Define a hook function
def print_grad(grad):
    print(f"Gradient flowing through: {grad}")
    return grad  # Must return the gradient`} />

      <CodeBlock language="python" code={`# Register the hook
hook_handle = x.register_hook(print_grad)

# Perform computation
y = x ** 2
loss = y.sum()
loss.backward()
# Output: Gradient flowing through: tensor([4., 6.])`} />

      <Text mt="md">
        Hooks can also modify gradients:
      </Text>

      <CodeBlock language="python" code={`def clip_gradient(grad):
    # Clip gradients to [-1, 1]
    return torch.clamp(grad, -1, 1)

x = torch.tensor([2.0, 3.0], requires_grad=True)
x.register_hook(clip_gradient)

y = x ** 3  # Gradient would be [12, 27]
y.sum().backward()
print(f"Clipped gradient: {x.grad}")  # tensor([1., 1.])`} />

      <Title order={2} mt="xl">6. Practical Example: Simple Optimization</Title>
      
      <Text>
        Let's use autograd to minimize a simple function <InlineMath>{`f(x) = (x - 3)^2`}</InlineMath>:
      </Text>

      <CodeBlock language="python" code={`# Start with a random value
x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

# Track the optimization process
history = []`} />

      <Text mt="md">
        Perform gradient descent steps:
      </Text>

      <CodeBlock language="python" code={`for step in range(20):
    # Forward pass
    loss = (x - 3) ** 2
    history.append(loss.item())
    
    # Backward pass
    loss.backward()
    
    # Gradient descent step
    with torch.no_grad():  # Don't track this operation
        x -= learning_rate * x.grad
        x.grad.zero_()  # Clear gradient for next iteration
    
    if step % 5 == 0:
        print(f"Step {step}: x = {x.item():.3f}, loss = {loss.item():.3f}")`} />

      <Text mt="md">
        Output shows convergence to the minimum at x = 3:
      </Text>

      <CodeBlock language="python" code={`# Step 0: x = 0.000, loss = 9.000
# Step 5: x = 2.046, loss = 0.913  
# Step 10: x = 2.651, loss = 0.122
# Step 15: x = 2.893, loss = 0.011`} />

      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module2/gradient-descent.png"
          alt="Gradient Descent Visualization"
          style={{ maxWidth: 'min(600px, 90vw)', height: 'auto' }}
          fluid
        />
      </Flex>

      <Title order={2} mt="xl">7. Advanced: Custom Autograd Functions</Title>
      
      <Text>
        While PyTorch provides gradients for all standard operations, you can define custom operations with their own 
        backward passes:
      </Text>

      <CodeBlock language="python" code={`class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        # Implement ReLU: max(0, x)
        return input.clamp(min=0)`} />

      <CodeBlock language="python" code={`    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        input, = ctx.saved_tensors
        # Gradient is 1 where input > 0, else 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input`} />

      <Text mt="md">
        Use the custom function:
      </Text>

      <CodeBlock language="python" code={`# Apply custom ReLU
x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
relu = MyReLU.apply
y = relu(x)
print(f"Forward: {y}")  # tensor([0., 0., 0., 1., 2.])`} />

      <CodeBlock language="python" code={`# Check gradients
y.sum().backward()
print(f"Gradient: {x.grad}")  # tensor([0., 0., 0., 1., 1.])`} />

    </Container>
  );
};

export default AutogradTorchPerspective;