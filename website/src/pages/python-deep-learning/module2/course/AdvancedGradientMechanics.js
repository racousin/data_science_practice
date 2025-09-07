import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Image, Flex } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const AdvancedGradientMechanics = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        
        <div id="gradient-flow">
          <Title order={2} mb="xl">
            Gradient Flow & Vanishing/Exploding Gradients
          </Title>
          
          <Paper className="p-6 bg-blue-50 mb-6">
            <Title order={3} mb="md">Mathematical Foundation of Gradient Flow</Title>
            <Text className="mb-4">
              During backpropagation, gradients flow backwards through the network. For a deep network with <InlineMath math="L" /> layers,
              the gradient of the loss with respect to parameters in layer <InlineMath math="\ell" /> involves a product of derivatives:
            </Text>
            
            <BlockMath math="\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \cdot \prod_{i=\ell+1}^{L} \frac{\partial a^{(i)}}{\partial a^{(i-1)}} \cdot \frac{\partial a^{(\ell)}}{\partial W^{(\ell)}}" />
            
            <Text className="mb-4 mt-4">
              Each factor <InlineMath math="\frac{\partial a^{(i)}}{\partial a^{(i-1)}}" /> depends on the activation function's derivative.
              This chain of multiplications is where problems arise.
            </Text>
            
          </Paper>

          <Title order={3} mb="md">The Vanishing Gradient Problem</Title>
          
          <Paper className="p-6 mb-6">
            <Text className="mb-4">
              <strong>Mathematical Analysis:</strong> Consider the sigmoid activation <InlineMath math="\sigma(z) = \frac{1}{1 + e^{-z}}" />. 
              Its derivative is <InlineMath math="\sigma'(z) = \sigma(z)(1 - \sigma(z))" />, which has maximum value:
            </Text>
            
            <BlockMath math="\max_z \sigma'(z) = \sigma(0)(1-\sigma(0)) = 0.25" />
            
            <Text className="mb-4">
              For a network with <InlineMath math="L" /> layers using sigmoid activations, the gradient involves the product:
            </Text>
            
            <BlockMath math="\left|\frac{\partial \mathcal{L}}{\partial W^{(1)}}\right| \leq \left|\frac{\partial \mathcal{L}}{\partial a^{(L)}}\right| \cdot \prod_{i=2}^{L} |\sigma'(z^{(i)})| \cdot \prod_{i=1}^{L-1} ||W^{(i+1)}||" />
            
            <Text className="mb-4">
              Since <InlineMath math="|\sigma'(z)| \leq 0.25" />, with <InlineMath math="L = 10" /> layers:
              <InlineMath math=" (0.25)^{10} \approx 9.5 \times 10^{-7}" />. Gradients vanish exponentially!
            </Text>
            
            <CodeBlock language="python" code={`# Visualizing sigmoid gradient vanishing
import torch
import torch.nn as nn

# Single layer gradient flow
x = torch.linspace(-10, 10, 100, requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()

print(f"Max gradient: {x.grad.max():.4f}")  # Max = 0.25
print(f"Gradient at x=5: {x.grad[75]:.6f}")  # Nearly zero`} />
            
            <Text className="mb-4 mt-4">
              <strong>Chain effect in deep networks:</strong> Each layer compounds the problem:
            </Text>
            
            <CodeBlock language="python" code={`# Gradient vanishing through layers
def sigmoid_chain(depth):
    x = torch.tensor([2.0], requires_grad=True)
    for _ in range(depth):
        x = torch.sigmoid(x)
    x.backward()
    return x.grad.item()

for d in [1, 5, 10, 20]:
    grad = sigmoid_chain(d)
    print(f"Depth {d:2d}: gradient = {grad:.2e}")`} />
          </Paper>

          <Title order={3} mb="md">The Exploding Gradient Problem</Title>
          
          <Paper className="p-6 mb-6">
            <Text className="mb-4">
              <strong>Mathematical Analysis:</strong> When weight matrices have large eigenvalues <InlineMath math="\lambda_{\max}(W) > 1" />,
              gradients grow exponentially through layers:
            </Text>
            
            <BlockMath math="\left|\frac{\partial \mathcal{L}}{\partial W^{(1)}}\right| \propto \prod_{i=1}^{L-1} \lambda_{\max}(W^{(i)}) \approx \lambda^{L-1}" />
            
            <Text className="mb-4">
              If <InlineMath math="\lambda_{\max} = 2" /> and <InlineMath math="L = 10" />: 
              <InlineMath math=" 2^9 = 512" /> times amplification. With poor initialization or unbounded activations (ReLU), this explodes quickly.
            </Text>
            
            <CodeBlock language="python" code={`# Weight initialization impact
W = torch.randn(10, 10) * 3  # Large initialization
eigenvalues = torch.linalg.eigvals(W).abs()
print(f"Max eigenvalue: {eigenvalues.max():.2f}")

# Gradient amplification
x = torch.randn(1, 10, requires_grad=True)
for _ in range(5):
    x = x @ W
grad_norm = torch.autograd.grad(x.sum(), x)[0].norm()
print(f"Gradient norm after 5 layers: {grad_norm:.2e}")`} />
            
            <Text className="mb-4 mt-4">
              <strong>ReLU and gradient explosion:</strong> While ReLU helps with vanishing gradients (derivative is 1 for positive inputs),
              it can cause explosion with poor initialization:
            </Text>
            
            <CodeBlock language="python" code={`# ReLU gradient behavior
def relu_chain(depth, std):
    x = torch.randn(100, requires_grad=True)
    for _ in range(depth):
        W = torch.randn(100, 100) * std
        x = torch.relu(W @ x)
    loss = x.sum()
    loss.backward()
    return x.grad.norm().item()

print(f"std=0.1: {relu_chain(10, 0.1):.2e}")
print(f"std=1.0: {relu_chain(10, 1.0):.2e}")
print(f"std=2.0: {relu_chain(10, 2.0):.2e}")`} />
          </Paper>

          <Flex direction="column" align="center" mt="md" mb="md">
            <Image
              src="/assets/python-deep-learning/module3/individualImage.png"
              alt="Gradient Flow Visualization"
              style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
              fluid
            />
            <Text size="sm" c="dimmed" mt="xs">Gradient Flow Through Deep Networks</Text>
          </Flex>

        </div>


       

      </Stack>
    </Container>
  );
};

export default AdvancedGradientMechanics;