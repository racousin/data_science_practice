import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Image, Flex } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const AdvancedGradientMechanics = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        <div data-slide>
        <div id="gradient-flow">
          <Title order={2} mb="xl">
            Gradient Flow & Vanishing/Exploding Gradients
          </Title>
           <Flex direction="column" align="center" mt="md" mb="md">
            <Image
              src="/assets/python-deep-learning/module3/histogram.png"
              alt="Gradient Flow Visualization"
              style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
              fluid
            />
                      </Flex>
          </div>
          </div>
          <div data-slide>
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
</div>
<div data-slide>
          <Title order={3} mb="md">The Vanishing Gradient Problem</Title>
          
            <Text className="mb-4">
              Consider the sigmoid activation <InlineMath math="\sigma(z) = \frac{1}{1 + e^{-z}}" />. 
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
            </div>
            <div data-slide>
            <CodeBlock language="python" code={`# Visualizing sigmoid gradient vanishing
import torch
import torch.nn as nn

# Single layer gradient flow
x = torch.linspace(-10, 10, 100, requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()

print(f"Max gradient: {x.grad.max():.4f}")  # Max = 0.25
print(f"Gradient at x=5: {x.grad[75]:.6f}")  # Nearly zero`} />
            
          <Flex direction="column" align="center" mt="md" mb="md">
            <Image
              src="/assets/python-deep-learning/module2/vanish.png"
              alt="Gradient Flow Visualization"
              style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
              fluid
            />
                      </Flex>
                      </div>
                      <div data-slide>
          <Title order={3} mb="md">The Exploding Gradient Problem</Title>
          
          <Paper className="p-6 mb-6">
            <Text className="mb-4">
              Consider ReLU activation: <InlineMath math="\text{ReLU}(x) = \max(0, x)" /> with derivative:
            </Text>
            
            <BlockMath math="\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}" />
            
            <Text className="mb-4">
              Unlike sigmoid (max derivative = 0.25), ReLU has derivative = 1 for positive inputs.
              With unscaled inputs and standard weight initialization, the gradient through each layer is:
            </Text>
            
            <BlockMath math="\frac{\partial a^{(i)}}{\partial a^{(i-1)}} = W^{(i)} \cdot \text{ReLU}'(z^{(i-1)})" />
            
            <Text className="mb-4">
              When inputs are large (e.g., magnitude ~10) and weights have std ~1, the product 
              <InlineMath math="|W \cdot x|" /> can be much greater than 1. Since ReLU doesn't shrink gradients
              (derivative = 1), each layer can amplify the gradient, causing exponential growth.
            </Text>
            
          </Paper>
          </div>
                      <div data-slide>
            <CodeBlock language="python" code={`Epoch 0, Loss: 1774099840.000000, Gradient Norm: 25415619288.704369
Training stopped at epoch 2 due to NaN loss (exploded gradients)`} />

</div>



       

      </Stack>
    </Container>
  );
};

export default AdvancedGradientMechanics;