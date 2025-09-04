import React from 'react';
import { Container, Title, Text, Space, List, Flex, Image, Paper, Alert, Grid } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const AutogradTorchPerspective = () => {
  return (
    <Container fluid>
      <Title order={1} mb="lg">Autograd: PyTorch's Automatic Differentiation Engine</Title>
      
      <Text>

We saw in the previous section (/courses/python-deep-learning/module1/course/pytorch-introduction), the tensor definition, features and operations, but if PyTorch was only that, it would be just another NumPy. But where PyTorch is super useful is its capacity to compute efficiently gradients (of potentially billions of parameters) in its implementation of reverse Automatic Differentiation (named autograd), that we will describe here.
      </Text>

      <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module2/augmented_computational_graph.png"
          alt="Autograd Overview"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
                                                          <Text component="p" ta="center" mt="xs">
                                            Source: https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/
                                          </Text>
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

      <Text mt="md">
        Each operation creates a new node in the graph:
      </Text>

                    <Grid gutter="lg">
                      <Grid.Col span={6}>
      <Flex direction="column" align="center" mt="md">
              <CodeBlock language="python" code={`x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x + y
print(f"z.grad_fn: {z.grad_fn}")  # <AddBackward0>`} />
        <Image
          src="/assets/python-deep-learning/module2/graph0.png"
          alt="Autograd Overview"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
      </Flex>
                      </Grid.Col>
                      <Grid.Col span={6}>
            <Flex direction="column" align="center" mt="md">
                    <CodeBlock language="python" code={`# Chain more operations
                    x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
w = z * z
print(f"w.grad_fn: {w.grad_fn}")  # <MulBackward0>`} />
        <Image
          src="/assets/python-deep-learning/module2/graph1.png"
          alt="Autograd Overview"
          style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
          fluid
        />
      </Flex>
                      </Grid.Col>
                    </Grid>

                    <Grid gutter="lg">
                      <Grid.Col span={6}>
      <Flex direction="column" align="center" mt="md">
              <CodeBlock language="python" code={`
                x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
w = z / 2.`} />
        <Image
          src="/assets/python-deep-learning/module2/graph2.png"
          alt="Autograd Overview"
          style={{ maxWidth: 'min(300px, 40vw)', height: 'auto' }}
          fluid
        />
      </Flex>
                      </Grid.Col>
                      <Grid.Col span={6}>
            <Flex direction="column" align="center" mt="md">
                    <CodeBlock language="python" code={`x = torch.ones(3,4, requires_grad=True)
y = torch.rand(4, requires_grad=True)
w = torch.sin(x @ y)`} />
        <Image
          src="/assets/python-deep-learning/module2/graph2bis.png"
          alt="Autograd Overview"
          style={{ maxWidth: 'min(300px, 40vw)', height: 'auto' }}
          fluid
        />
      </Flex>
                      </Grid.Col>
                    </Grid>

      <Text mt="md">
        The <code>grad_fn</code> attribute stores the function that created this tensor. It will be used during backpropagation 
        to compute gradients.
      </Text>

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

      <Title order={2} mt="xl">2. Understanding grad_fn Objects</Title>
      
      <Title order={3} mt="md">Each Operation Creates a Function Object</Title>
      
      <Text>
        Every operation in PyTorch has a corresponding backward function that knows how to compute gradients:
      </Text>
      
      <CodeBlock language="python" code={`# Clear previous gradients
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)`} />
      
      <Text mt="md">
        <strong>Addition</strong> creates an <code>AddBackward</code> object:
      </Text>
      
      <CodeBlock language="python" code={`z_add = x + y
print(f"Operation: {z_add.grad_fn}")  # <AddBackward0>
# This function knows: d(x+y)/dx = 1, d(x+y)/dy = 1

# Example: Stream gradients through addition
upstream_grad = torch.tensor([0.1, 0.2, 0.3])
grad_x, grad_y = z_add.grad_fn(upstream_grad)
# grad_x = [0.1, 0.2, 0.3]  (multiplied by 1)
# grad_y = [0.1, 0.2, 0.3]  (multiplied by 1)`} />
      
      <Text mt="md">
        <strong>Multiplication</strong> creates a <code>MulBackward</code> object:
      </Text>
      
      <CodeBlock language="python" code={`z_mul = x * y   
print(f"Operation: {z_mul.grad_fn}")  # <MulBackward0>
# This function knows: d(x*y)/dx = y, d(x*y)/dy = x

# Example: Stream gradients through multiplication
upstream_grad = torch.tensor([0.1, 0.2, 0.3])
grad_x, grad_y = z_mul.grad_fn(upstream_grad)
# grad_x = [0.1*4, 0.2*5, 0.3*6] = [0.4, 1.0, 1.8]
# grad_y = [0.1*1, 0.2*2, 0.3*3] = [0.1, 0.4, 0.9]`} />
      
      <Text mt="md">
        <strong>Trigonometric functions</strong> have their own backward functions:
      </Text>
      
      <CodeBlock language="python" code={`z_sin = torch.sin(x)
print(f"Operation: {z_sin.grad_fn}")  # <SinBackward0>
# This function knows: d(sin(x))/dx = cos(x)

# Example: Stream gradients through sine
upstream_grad = torch.tensor([0.1, 0.2, 0.3])
grad_x = z_sin.grad_fn(upstream_grad)
# grad_x = upstream_grad * cos(x)
# grad_x = [0.1*cos(1), 0.2*cos(2), 0.3*cos(3)]
# grad_x ≈ [0.054, -0.083, -0.297]`} />
      
      <Text mt="md">
        <strong>Exponential function</strong>:
      </Text>
      
      <CodeBlock language="python" code={`z_exp = torch.exp(x)
print(f"Operation: {z_exp.grad_fn}")  # <ExpBackward0>
# This function knows: d(exp(x))/dx = exp(x)

# Example: Stream gradients through exponential
upstream_grad = torch.tensor([0.1, 0.2, 0.3])
grad_x = z_exp.grad_fn(upstream_grad)
# grad_x = upstream_grad * exp(x)
# grad_x = [0.1*exp(1), 0.2*exp(2), 0.3*exp(3)]
# grad_x ≈ [0.272, 1.478, 6.026]`} />

      <Title order={3} mt="xl">Exploring the Graph Structure</Title>
      
      <Text>
        Each <code>grad_fn</code> has a <code>next_functions</code> attribute linking to its inputs' gradient functions:
      </Text>


      <Title order={2} mt="xl">3. Computing Gradients Step by Step</Title>
      

<Title order={3} mt="xl">How Backpropagation Works in Autograd</Title>

<Text>
  Let's trace through what happens during <code>y.backward()</code> for a more complex function:
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

<Text mt="md">
  Here's what happens internally when we call <code>y.backward()</code>:
</Text>
            <Flex direction="column" align="center" mt="md">
        <Image
          src="/assets/python-deep-learning/module2/graph3.png"
          alt="Autograd Overview"
          style={{ maxWidth: 'min(200px, 40vw)', height: 'auto' }}
          fluid
        />
      </Flex>
<Title order={4} mt="lg">Step 1: Initialize</Title>
<CodeBlock language="python" code={`upstream_grad = torch.tensor(1.0)  # dL/dy = 1
print(f"Starting: dL/dy = {upstream_grad}")
# u = 1² = 1, y = sin(1) ≈ 0.841`} />

<Text mt="sm">
  <InlineMath>{`\\frac{dL}{dy} = 1`}</InlineMath> (treating y as final loss)
</Text>

<Title order={4} mt="lg">Step 2: Apply SinBackward</Title>
<CodeBlock language="python" code={`sin_grad = y.grad_fn(upstream_grad)
print(f"SinBackward result: {sin_grad}")  # cos(1) ≈ 0.540`} />

<Text mt="sm">
  <InlineMath>{`\\frac{dL}{du} = \\frac{dL}{dy} \\cdot \\frac{dy}{du} = 1 \\cdot \\cos(u) = \\cos(1) \\approx 0.540`}</InlineMath>
</Text>

<Title order={4} mt="lg">Step 3: Apply PowBackward</Title>
<CodeBlock language="python" code={`pow_backward = y.grad_fn.next_functions[0][0]
pow_grad = pow_backward(sin_grad[0])
print(f"PowBackward result: {pow_grad}")  # 2*1*0.540 ≈ 1.081`} />

<Text mt="sm">
  <InlineMath>{`\\frac{dL}{dx} = \\frac{dL}{du} \\cdot \\frac{du}{dx} = 0.540 \\cdot 2x = 0.540 \\cdot 2 \\cdot 1 = 1.081`}</InlineMath>
</Text>

<Title order={4} mt="lg">Step 4: Final Result</Title>
<CodeBlock language="python" code={`x_grad = pow_grad[0]  # ≈ 1.081
print(f"Final gradient: dL/dx = {x_grad}")

`} />

<Text mt="sm">
  <InlineMath>{`\\frac{dy}{dx} = \\cos(x^2) \\cdot 2x = \\cos(1) \\cdot 2 \\approx 1.081`}</InlineMath>
</Text>

<Text mt="md">
  The computational graph flows: <code>x → PowBackward → SinBackward → output</code>
  <br />
  During backprop: <code>upstream_grad → SinBackward → PowBackward → x.grad</code>


  
</Text>


      <Title order={3} mt="xl">The .backward() Method</Title>

      <Text mt="md">
        The <code>.backward()</code> method automates the entire process we described above. 
        It traverses the computational graph in reverse order, applying each grad_fn with the appropriate upstream gradients:
      </Text>

      <CodeBlock language="python" code={`import torch

x = torch.tensor(1.0, requires_grad=True)

# Forward pass - building the graph
u = x ** 2        # u = x²
y = torch.sin(u)  # y = sin(u) = sin(x²)

# Backward pass - compute gradients
y.backward()      # Automatically does everything we described above!

# Access the gradient
print(f"x.grad = {x.grad}")  # tensor(1.0806)
# This is exactly cos(1) * 2 * 1 ≈ 1.0806`} />

      <Title order={4} mt="lg">Important: Gradient Accumulation</Title>
      
      <Text mt="md">
        By default, PyTorch <strong>accumulates</strong> gradients. This means calling <code>.backward()</code> 
        multiple times will add to existing gradients rather than replacing them:
      </Text>

      <CodeBlock language="python" code={`# First backward pass
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
y1.backward()
print(f"After first backward: x.grad = {x.grad}")  # tensor(2.)

# Second backward pass - gradients accumulate!
y2 = x ** 3
y2.backward()
print(f"After second backward: x.grad = {x.grad}")  # tensor(5.)
# This is 2 (from first) + 3 (from second) = 5`} />

      <Title order={4} mt="lg">Using zero_grad() to Reset Gradients</Title>

      <CodeBlock language="python" code={`x = torch.tensor(1.0, requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(f"First backward: x.grad = {x.grad}")  # tensor(2.)

# Reset gradient to zero
x.grad.zero_()  # or x.grad = None

# Second computation (fresh gradient)
y2 = x ** 3
y2.backward()
print(f"Second backward: x.grad = {x.grad}")  # tensor(3.)
# Now it's just 3, not accumulated`} />



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

      <Title order={3} mt="xl">Using torch.no_grad() Context Manager</Title>
      
      <Text>
        While <code>detach()</code> works for individual tensors, <code>torch.no_grad()</code> disables gradient computation entirely within a context. This is more efficient when you know you won't need gradients:
      </Text>

      <CodeBlock language="python" code={`x = torch.tensor(2.0, requires_grad=True)

# Normal operation - builds graph
y = x ** 2
print(f"y.requires_grad: {y.requires_grad}")  # True

# Inside no_grad context - no graph building
with torch.no_grad():
    y_no_grad = x ** 2
    print(f"y_no_grad.requires_grad: {y_no_grad.requires_grad}")  # False
    
    # Even operations on tensors with requires_grad=True won't track gradients
    z = x * 3 + 2
    print(f"z.requires_grad: {z.requires_grad}")  # False`} />
      <Title order={2} mt="xl">5. Practical Example: Simple Optimization</Title>
      
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

    </Container>
  );
};

export default AutogradTorchPerspective;