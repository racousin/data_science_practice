import React from 'react';
import { Title, Text, Stack } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';

const ForwardGradientDemo = () => {
  return (
    <Stack spacing="md" className="w-full">
      <Title order={2} id="forward-gradients">Forward Mode Automatic Differentiation</Title>
      
      <Text>
        Forward mode automatic differentiation computes directional derivatives alongside the forward pass. 
        This is particularly efficient when you have many outputs but few inputs.
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch
from torch.autograd.forward_ad import dual_level, make_dual

def example_function(x):
    return x ** 2 + 2 * x + 1

# Create input tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Compute forward gradient
with dual_level():
    # Create dual tensor with tangent
    dual_x = make_dual(x, torch.ones_like(x))
    
    # Compute function and get primal and tangent
    dual_output = example_function(dual_x)
    primal, tangent = dual_output.primal, dual_output.tangent

print(f"Primal (function value): {primal}")
print(f"Tangent (forward gradient): {tangent}")

# Compare with backward mode
y = example_function(x)
y.sum().backward()
print(f"Backward gradient: {x.grad}")
`}
      />

      <Text>
        The forward gradient computes derivatives in the same direction as the computation flow, 
        making it efficient for functions with few inputs and many outputs. 
        In contrast, backward mode (traditional backpropagation) is more efficient for 
        functions with many inputs and few outputs, which is why it's more commonly used in deep learning.
      </Text>
    </Stack>
  );
};

export default ForwardGradientDemo;