import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, List, Table, Box, Divider,Accordion  } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const AutomaticDifferentiationMathematicalPerspective = () => {
  return (
    <Container size="xl">
      <Stack spacing="xl">
        <div data-slide>
        <Title order={2} id="gradient-methods">Gradient Computation Methods</Title>

In the previous section, we explored how to optimize parametric models using gradient-based methods. However, this raises an important practical question: when working with deep learning models that contain millions or even billions of parameters, how do we efficiently compute the gradients needed for optimization?<BlockMath>{`\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta \\mathcal{L}(\\theta_t)`}</BlockMath>
how do we compute efficiently <InlineMath>{` \\nabla_\\theta \\mathcal{L}`}</InlineMath>?
</div>
<div data-slide>
  <Table withTableBorder withColumnBorders>
    <Table.Thead>
      <Table.Tr>
        <Table.Th>Method</Table.Th>
        <Table.Th>Principle</Table.Th>
        <Table.Th>Limitations for Neural Networks</Table.Th>
      </Table.Tr>
    </Table.Thead>
    <Table.Tbody>
      <Table.Tr>
        <Table.Td>
          <strong>Symbolic Differentiation</strong>
        </Table.Td>
        <Table.Td>
          Applies differentiation rules to derive exact mathematical expressions
          <BlockMath>{`\\frac{d}{dx}(x^2) = 2x`}</BlockMath>
        </Table.Td>
        <Table.Td>
          • Expression complexity explodes with network size
          • Impractical for dynamic computations in deep networks
          • Memory intensive for large expressions
        </Table.Td>
      </Table.Tr>

      <Table.Tr>
        <Table.Td>
          <strong>Finite Differences</strong>
        </Table.Td>
        <Table.Td>
          Approximates derivatives using small perturbations
          <BlockMath>{`\\frac{\\partial L}{\\partial \\theta} \\approx \\frac{L(\\theta + h) - L(\\theta)}{h}`}</BlockMath>
        </Table.Td>
        <Table.Td>
          • Requires O(n) evaluations for n parameters
          • Numerically unstable (sensitive to h)
          • Computationally expensive for millions of parameters
        </Table.Td>
      </Table.Tr>

      <Table.Tr>
        <Table.Td>
          <strong>Automatic Differentiation</strong>
        </Table.Td>
        <Table.Td>
          Builds computation graph and applies chain rule systematically
          <BlockMath>{`\\frac{\\partial z}{\\partial x} = \\sum_i \\frac{\\partial z}{\\partial y_i}\\frac{\\partial y_i}{\\partial x}`}</BlockMath>
        </Table.Td>
        <Table.Td>
          ✓ Efficient computation
          ✓ Exact gradients
          ✓ Handles dynamic computations
        </Table.Td>
      </Table.Tr>
    </Table.Tbody>
  </Table>
</div>
<div data-slide>
  <Title order={3} mt="lg">Automatic Differentiation Modes</Title>
  
  <Grid gutter="xl">
    <Grid.Col span={6}>
      <Box className="p-4 bg-blue-50 rounded">
        <Title order={4}>Forward Mode</Title>
        <Text>
          Propagates derivatives forward through computation graph from inputs to outputs.
        </Text>
        <List>
          <List.Item>Efficient for functions with few inputs and many outputs</List.Item>
          <List.Item>Computes one input derivative at a time</List.Item>
          <List.Item>O(n) complexity for n inputs</List.Item>
        </List>
        <BlockMath>{`\\dot{y} = \\frac{\\partial f}{\\partial x}\\dot{x}`}</BlockMath>
      </Box>
    </Grid.Col>

    <Grid.Col span={6}>
      <Box className="p-4 bg-green-50 rounded">
        <Title order={4}>Reverse Mode (Backpropagation)</Title>
        <Text>
          Propagates derivatives backward from outputs to inputs.
        </Text>
        <List>
          <List.Item>Efficient for functions with many inputs and few outputs</List.Item>
          <List.Item>Computes all input derivatives in one pass</List.Item>
          <List.Item>O(1) complexity regardless of input size</List.Item>
        </List>
        <BlockMath>{`\\bar{x} = \\bar{y}\\frac{\\partial f}{\\partial x}`}</BlockMath>
      </Box>
    </Grid.Col>
  </Grid>

  <Text mt="md">
    Backpropagation is reverse-mode automatic differentiation specialized for scalar outputs (loss function),
    making it ideal for neural networks with millions of parameters but single scalar loss.
  </Text>
  </div>

      {/* Backpropagation Details */}
      <Stack spacing="md">
        </Stack>
<div data-slide>
        <Title order={2} id="backprop-details">Backpropagation: A Reverse-Mode Automatic Differentiation</Title>
        
        <Text>
          Backpropagation efficiently computes gradients by decomposing the computation graph and applying the chain rule backward from the output to inputs.
        </Text>
  
        <Table withTableBorder withColumnBorders>
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Symbol</Table.Th>
          <Table.Th>Description</Table.Th>
          <Table.Th>Formula</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        <Table.Tr>
          <Table.Td>
            <InlineMath>{`w_{ij}^k`}</InlineMath>
          </Table.Td>
          <Table.Td>Weight for node j in layer k receiving input from node i</Table.Td>
          <Table.Td>
            <InlineMath>{`w_{ij}^k \\in \\mathbb{R}`}</InlineMath>
          </Table.Td>
        </Table.Tr>
        <Table.Tr>
          <Table.Td>
            <InlineMath>{`b_i^k`}</InlineMath>
          </Table.Td>
          <Table.Td>Bias for node i in layer k</Table.Td>
          <Table.Td>
            <InlineMath>{`b_i^k \\in \\mathbb{R}`}</InlineMath>
          </Table.Td>
        </Table.Tr>
        <Table.Tr>
          <Table.Td>
            <InlineMath>{`a_i^k`}</InlineMath>
          </Table.Td>
          <Table.Td>Product sum plus bias (pre-activation) for node i in layer k</Table.Td>
          <Table.Td>
            <InlineMath>{`a_i^k = \\sum_{j=1}^{r_{k-1}} w_{ij}^k o_j^{k-1} + b_i^k = \\sum_{j=0}^{r_{k-1}} w_{ij}^k o_j^{k-1} (o_0^{k-1}=1)`}</InlineMath>
          </Table.Td>
        </Table.Tr>
        <Table.Tr>
          <Table.Td>
            <InlineMath>{`o_i^k`}</InlineMath>
          </Table.Td>
          <Table.Td>Output (post-activation) for node i in layer k</Table.Td>
          <Table.Td>
            <InlineMath>{`o_i^k = g^k(a_i^k)`}</InlineMath>
          </Table.Td>
        </Table.Tr>
        <Table.Tr>
          <Table.Td>
            <InlineMath>{`r_k`}</InlineMath>
          </Table.Td>
          <Table.Td>Number of nodes in layer k</Table.Td>
          <Table.Td>
            <InlineMath>{`r_k \\in \\mathbb{N}`}</InlineMath>
          </Table.Td>
        </Table.Tr>
        <Table.Tr>
          <Table.Td>
            <InlineMath>{`g^k`}</InlineMath>
          </Table.Td>
          <Table.Td>Activation function for layer k</Table.Td>
          <Table.Td>
            <InlineMath>{`g^k: \\mathbb{R} \\rightarrow \\mathbb{R}`}</InlineMath>
          </Table.Td>
        </Table.Tr>
        <Table.Tr>
          <Table.Td>
            <InlineMath>{`L(X,\\theta)`}</InlineMath>
          </Table.Td>
          <Table.Td>Loss function</Table.Td>
          <Table.Td>
            <InlineMath>{`L(\\theta) = L(\\hat{y}, y)`}</InlineMath>
          </Table.Td>
        </Table.Tr>
      </Table.Tbody>
    </Table>
</div>
    <Accordion variant="separated">
        {/* Weight Gradient Proof */}
        <Accordion.Item value="weight-gradient">
          <Accordion.Control>
            Proof
          </Accordion.Control>
          <Accordion.Panel>
    <Title order={3} id="error-derivatives" className="mb-4">
        Error Function Derivatives
      </Title>

      {/* Initial Chain Rule */}
      <Text>
        The derivation of the backpropagation algorithm begins by applying the chain rule to the error function partial derivative:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\frac{\\partial L}{\\partial w_{ij}^k} = 
          \\frac{\\partial L}{\\partial a_j^k}
          \\frac{\\partial a_j^k}{\\partial w_{ij}^k}
        `}</BlockMath>
      </Box>

      {/* Error Term Definition */}
      <Text>
        The first term is usually called the <strong>error</strong>, for reasons discussed below. It is denoted:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\delta_j^k \\equiv \\frac{\\partial L}{\\partial a_j^k}
        `}</BlockMath>
      </Box>

      {/* Second Term Calculation */}
      <Text>
        The second term can be calculated from the equation for <InlineMath>{`a_j^k`}</InlineMath> above:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\frac{\\partial a_j^k}{\\partial w_{ij}^k} = 
          \\frac{\\partial}{\\partial w_{ij}^k}
          \\left(\\sum_{l=0}^{r_k-1} w_{lj}^k o_l^{k-1}\\right) = 
          o_i^{k-1}
        `}</BlockMath>
      </Box>

      {/* Final Derivative Form */}
      <Text>
        Thus, the partial derivative of the error function <InlineMath>L</InlineMath> with respect to a weight <InlineMath>{`w_{ij}^k`}</InlineMath> is:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\frac{\\partial L}{\\partial w_{ij}^k} = \\delta_j^k o_i^{k-1}
        `}</BlockMath>
      </Box>

      {/* Intuitive Explanation */}
      <Text>
        Thus, the partial derivative of a weight is a product of the error term <InlineMath>{`\\delta_j^k`}</InlineMath> at node j in layer k, 
        and the output <InlineMath>{`o_i^{k-1}`}</InlineMath> of node i in layer k−1. This makes intuitive sense since the 
        weight <InlineMath>{`w_{ij}^k`}</InlineMath> connects the output of node i in layer k−1 to the input of node j in layer k 
        in the computation graph.
      </Text>
      <Title order={3} id="output-layer" className="mb-4">
        The Output Layer
      </Title>

      {/* Introduction */}
      <Text>
        Starting from the final layer (with for example MSE loss), backpropagation attempts to define the value <InlineMath>{`\\delta_1^m`}</InlineMath>, 
        where <InlineMath>m</InlineMath> is the final layer (the subscript is 1 and not j because this derivation concerns 
        a one-output neural network, so there is only one output node <InlineMath>j=1</InlineMath>).
      </Text>

      {/* Error Function Expression */}
      <Text>
        Expressing the error function <InlineMath>L</InlineMath> in terms of the value <InlineMath>{`a_1^m`}</InlineMath> 
        (since <InlineMath>{`\\delta_1^m`}</InlineMath> is a partial derivative with respect to <InlineMath>{`a_1^m`}</InlineMath>) gives:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          L = \\frac{1}{2}(\\hat{y} - y)^2 = \\frac{1}{2}(g^m(a_1^m) - y)^2
        `}</BlockMath>
      </Box>

      <Text>
        where <InlineMath>{`g^m(x)`}</InlineMath> is the activation function for the output layer.
      </Text>

      {/* Delta Calculation */}
      <Text>
        Thus, applying the partial derivative and using the chain rule gives:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\begin{align}
          \\delta_1^m &= (g^m(a_1^m) - y)g^{m\\prime}(a_1^m) \\
          &= (\\hat{y} - y)g^{m\\prime}(a_1^m)
          \\end{align}
        `}</BlockMath>
      </Box>

      {/* Final Formula */}
      <Text>
        Putting it all together, the partial derivative of the error function <InlineMath>L</InlineMath> with respect to 
        a weight in the final layer <InlineMath>{`w_{i1}^m`}</InlineMath> is:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\begin{align}
          \\frac{\\partial L}{\\partial w_{i1}^m} &= \\delta_1^m o_i^{m-1} \\\\
          &= (\\hat{y} - y)g^{m\\prime}(a_1^m) o_i^{m-1}
          \\end{align}
        `}</BlockMath>
      </Box>

      <Title order={3} id="hidden-layers" className="mb-4">
        The Hidden Layers
      </Title>

      {/* Introduction */}
      <Text>
        Now the question arises of how to calculate the partial derivatives of layers other than the output layer. 
        Luckily, the chain rule for multivariate functions comes to the rescue again. Observe the following equation 
        for the error term <InlineMath>{`\\delta_j^k`}</InlineMath> in layer <InlineMath>{`1 \\leq k < m`}</InlineMath>:
      </Text>

      {/* Initial Error Term Expression */}
      <Box className="my-4">
        <BlockMath>{`
          \\delta_j^k = \\frac{\\partial L}{\\partial a_j^k} = 
          \\sum_{l=1}^{r_{k+1}} \\frac{\\partial L}{\\partial a_l^{k+1}}
          \\frac{\\partial a_l^{k+1}}{\\partial a_j^k}
        `}</BlockMath>
      </Box>

      <Text>
        where <InlineMath>l</InlineMath> ranges from 1 to <InlineMath>{`r_{k+1}`}</InlineMath> (the number of nodes in the next layer). 
        Note that, because the bias input <InlineMath>{`o_0^k`}</InlineMath> corresponding to <InlineMath>{`w_{0j}^{k+1}`}</InlineMath> is 
        fixed, its value is not dependent on the outputs of previous layers, and thus <InlineMath>l</InlineMath> does not take on the value 0.
      </Text>

      {/* Substitution of Error Term */}
      <Text>
        Plugging in the error term <InlineMath>{`\\delta_l^{k+1}`}</InlineMath> gives:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\delta_j^k = \\sum_{l=1}^{r_{k+1}} \\delta_l^{k+1}
          \\frac{\\partial a_l^{k+1}}{\\partial a_j^k}
        `}</BlockMath>
      </Box>

      {/* Definition of Next Layer Activation */}
      <Text>
        Remembering the definition of <InlineMath>{`a_l^{k+1}`}</InlineMath>:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          a_l^{k+1} = \\sum_{j=1}^{r_k} w_{jl}^{k+1} g^k(a_j^k)
        `}</BlockMath>
      </Box>

      <Text>
        where <InlineMath>g(x)</InlineMath> is the activation function for the hidden layers:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\frac{\\partial a_l^{k+1}}{\\partial a_j^k} = 
          w_{jl}^{k+1} g^{k\\prime}(a_j^k)
        `}</BlockMath>
      </Box>

      {/* Backpropagation Formula */}
      <Text>
        Plugging this into the above equation yields the final equation for the error term <InlineMath>{`\\delta_j^k`}</InlineMath> in 
        the hidden layers, called the <strong>backpropagation formula</strong>:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\begin{align}
          \\delta_j^k &= \\sum_{l=1}^{r_{k+1}} \\delta_l^{k+1} w_{jl}^{k+1} g^{k\\prime}(a_j^k) \\\\
          &= g^{k\\prime}(a_j^k) \\sum_{l=1}^{r_{k+1}} w_{jl}^{k+1} \\delta_l^{k+1}
          \\end{align}
        `}</BlockMath>
      </Box>

      {/* Final Weight Update Formula */}
      <Text>
        Putting it all together, the partial derivative of the error function <InlineMath>L</InlineMath> with respect to a weight in 
        the hidden layers <InlineMath>{`w_{ij}^k`}</InlineMath> for <InlineMath>{`1 \\leq k < m`}</InlineMath> is:
      </Text>

      <Box className="my-4">
        <BlockMath>{`
          \\begin{align}
          \\frac{\\partial L}{\\partial w_{ij}^k} &= \\delta_j^k o_i^{k-1} \\\\
          &= g^{k\\prime}(a_j^k) o_i^{k-1} \\sum_{l=1}^{r_{k+1}} w_{jl}^{k+1} \\delta_l^{k+1}
          \\end{align}
        `}</BlockMath>
      </Box>
      </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
      <div data-slide>
      <Title order={2} id="backpropagation-algorithm" className="mb-4">
        The Backpropagation Algorithm
      </Title>

      {/* Forward Pass */}
      <Title order={3} id="forward-pass">
        Step 1: Forward Pass
      </Title>

      <Text>
        For each layer k, compute the pre-activation and activation values:
      </Text>

      {/* Pre-activation computation */}
      <Box className="my-4">
        <BlockMath>{`
          a_j^k = \\sum_{i=1}^{r_{k-1}} w_{ij}^k o_i^{k-1} + b_j^k
        `}</BlockMath>
      </Box>

      {/* Activation computation */}
      <Box className="my-4">
        <BlockMath>{`
          o_j^k = g^k(a_j^k)
        `}</BlockMath>
      </Box>

      <Divider my="lg" />
</div>
<div data-slide>
      {/* Backward Pass */}
      <Title order={3} id="backward-pass">
        Step 2: Backward Pass
      </Title>

      <Text>
        Using the terms defined earlier, the backpropagation algorithm relies on the following key equations:
      </Text>

      {/* Weight Gradients */}
      <Box className="mt-4">
        <Text weight={500}>1. Partial Derivatives for Weights:</Text>
        <BlockMath>{`
          \\frac{\\partial L_d}{\\partial w_{ij}^k} = \\delta_j^k o_i^{k-1}
        `}</BlockMath>
      </Box>

      {/* Output Layer Error */}
      <Box className="mt-4">
        <Text weight={500}>2. Output Layer Error Term:</Text>
        <BlockMath>{`
          \\delta_1^m = g^{m\\prime}(a_1^m)(\\hat{y}_d - y_d)
        `}</BlockMath>
      </Box>
      </div>
      <div data-slide>
      {/* Weight Updates */}
      <Title order={3} id="backward-pass">
        Step 3: Weight Update
      </Title>
      
      <Box className="mt-4">
        <BlockMath>{`
          \\Delta w_{ij}^k = -\\alpha \\frac{\\partial L(X,\\theta)}{\\partial w_{ij}^k}
        `}</BlockMath>
      </Box>

</div>
      </Stack>
    </Container>
  );
};

export default AutomaticDifferentiationMathematicalPerspective;