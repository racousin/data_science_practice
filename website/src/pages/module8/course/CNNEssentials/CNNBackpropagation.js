import React from 'react';
import { Title, Text, Stack, Grid, Box, List, Table, Divider, Accordion } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const CNNBackpropagation = () => {
  return (
    <Stack spacing="xl" className="w-full">
      {/* Introduction to CNN Backpropagation */}
      <Title order={1} id="cnn-backpropagation">CNN Backpropagation: The Learning Algorithm</Title>
      
      <Stack spacing="md">
        <Text>
          Convolutional Neural Networks (CNNs) learn through backpropagation like standard neural networks, but with special consideration for the convolutional and pooling operations. The key difference lies in how gradients flow through these specialized layers.
        </Text>

        <Box className="p-4 border rounded">
          <Title order={4}>CNN Architecture Components</Title>
          <List>
            <List.Item><strong>Convolutional layers:</strong> Apply filters to detect features</List.Item>
            <List.Item><strong>Pooling layers:</strong> Downsample feature maps</List.Item>
            <List.Item><strong>Fully connected layers:</strong> Process extracted features (same as in standard networks)</List.Item>
          </List>
          
          <Text mt="md">Backpropagation must compute gradients for all these layer types.</Text>
        </Box>
      </Stack>

      {/* CNN Operation Notation */}
      <Stack spacing="md">
        <Title order={2} id="cnn-notation">CNN Notation</Title>
        <Text>
          To simplify the formulation of CNN backpropagation while preserving the core logic, we will consider convolutions with zero padding and a stride of 1. The same principles apply to other padding and stride configurations, but with more complex indexing.
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
                <InlineMath>{`x_{i,j,d}^l`}</InlineMath>
              </Table.Td>
              <Table.Td>Input at position (i,j) in channel d of layer l</Table.Td>
              <Table.Td>
                <InlineMath>{`x_{i,j,d}^l \\in \\mathbb{R}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`w_{i,j,d,k}^l`}</InlineMath>
              </Table.Td>
              <Table.Td>Weight at position (i,j) in filter connecting input channel d to output channel k in layer l</Table.Td>
              <Table.Td>
                <InlineMath>{`w_{i,j,d,k}^l \\in \\mathbb{R}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`b_k^l`}</InlineMath>
              </Table.Td>
              <Table.Td>Bias for filter k in layer l</Table.Td>
              <Table.Td>
                <InlineMath>{`b_k^l \\in \\mathbb{R}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`z_{i,j,k}^l`}</InlineMath>
              </Table.Td>
              <Table.Td>Pre-activation at position (i,j) in feature map k of layer l</Table.Td>
              <Table.Td>
                <InlineMath>{`z_{i,j,k}^l \\in \\mathbb{R}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`a_{i,j,k}^l`}</InlineMath>
              </Table.Td>
              <Table.Td>Post-activation at position (i,j) in feature map k of layer l</Table.Td>
              <Table.Td>
                <InlineMath>{`a_{i,j,k}^l = g^l(z_{i,j,k}^l)`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`L`}</InlineMath>
              </Table.Td>
              <Table.Td>Loss function</Table.Td>
              <Table.Td>
                <InlineMath>{`L = Loss(\\hat{y}, y)`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`*`}</InlineMath>
              </Table.Td>
              <Table.Td>Convolution operation</Table.Td>
              <Table.Td>
                <InlineMath>{`(f * g)(t) = \\int f(\\tau)g(t-\\tau)d\\tau`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`\\circledast`}</InlineMath>
              </Table.Td>
              <Table.Td>Cross-correlation operation (used in CNN implementations)</Table.Td>
              <Table.Td>
                <InlineMath>{`(f \\circledast g)(t) = \\int f(\\tau)g(t+\\tau)d\\tau`}</InlineMath>
              </Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
        
        <Text>
          Note: In practice, most CNN implementations use cross-correlation (no flipping of the kernel) rather than true convolution, but still call it "convolution" by convention.
        </Text>
      </Stack>

      {/* Forward Propagation in CNNs */}
      <Stack spacing="md">
        <Title order={2} id="forward-propagation">Forward Propagation in CNNs</Title>
        
        <Title order={3}>Convolutional Layer</Title>
        <Text>
          The convolution operation (actually cross-correlation in most implementations) is defined as:
        </Text>
        
        <BlockMath>{`
          z_{i,j,k}^l = b_k^l + \\sum_{d=1}^{D^{l-1}} \\sum_{p=0}^{f_h-1} \\sum_{q=0}^{f_w-1} w_{p,q,d,k}^l \\cdot a_{i+p,j+q,d}^{l-1}
        `}</BlockMath>
        
        <Text>
          Where:
        </Text>
        <List>
          <List.Item><InlineMath>{`D^{l-1}`}</InlineMath> is the number of channels in the previous layer</List.Item>
          <List.Item><InlineMath>{`f_h, f_w`}</InlineMath> are the filter height and width</List.Item>
        </List>
        
        <Text>
          After applying the activation function:
        </Text>
        
        <BlockMath>{`
          a_{i,j,k}^l = g^l(z_{i,j,k}^l)
        `}</BlockMath>
        
        <Title order={3} mt="lg">Pooling Layer</Title>
        <Text>
          For max pooling with a filter of size k_h × k_w and stride 1:
        </Text>
        
        <BlockMath>{`
          a_{i,j,k}^l = \\max_{0 \\leq p < k_h, 0 \\leq q < k_w} \\{a_{i+p,j+q,k}^{l-1}\\}
        `}</BlockMath>
        
        <Text>
          For average pooling with a filter of size k_h × k_w and stride 1:
        </Text>
        
        <BlockMath>{`
          a_{i,j,k}^l = \\frac{1}{k_h \\cdot k_w} \\sum_{p=0}^{k_h-1} \\sum_{q=0}^{k_w-1} a_{i+p,j+q,k}^{l-1}
        `}</BlockMath>
      </Stack>

      {/* Backpropagation in CNNs */}
      <Stack spacing="md">
        <Title order={2} id="backprop-in-cnns">Backpropagation in CNNs</Title>
        
        <Text>
          The backpropagation process in CNNs follows the same principles as in standard neural networks: 
          compute the gradient of the loss with respect to each parameter, and update the parameters accordingly.
          However, the gradient computation differs for each type of layer.
        </Text>
        
        <Accordion variant="separated">
          <Accordion.Item value="backprop-proof">
            <Accordion.Control>
              <Title order={3}>Proof of Backpropagation for Convolutional Layers</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack spacing="md">
                <Title order={4}>Computing Gradients for Filter Weights</Title>
                
                <Text>
                  For a convolutional layer, we need to compute <InlineMath>{`\\frac{\\partial L}{\\partial w_{p,q,d,k}^l}`}</InlineMath>.
                </Text>
                
                <Text>
                  Using the chain rule:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial w_{p,q,d,k}^l} = \\sum_{i,j} \\frac{\\partial L}{\\partial z_{i,j,k}^l} \\frac{\\partial z_{i,j,k}^l}{\\partial w_{p,q,d,k}^l}
                `}</BlockMath>
                
                <Text>
                  We define the error term for CNNs as:
                </Text>
                
                <BlockMath>{`
                  \\delta_{i,j,k}^l = \\frac{\\partial L}{\\partial z_{i,j,k}^l}
                `}</BlockMath>
                
                <Text>
                  From the forward propagation equation, we have:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial z_{i,j,k}^l}{\\partial w_{p,q,d,k}^l} = a_{i+p,j+q,d}^{l-1}
                `}</BlockMath>
                
                <Text>
                  Substituting, we get:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial w_{p,q,d,k}^l} = \\sum_{i,j} \\delta_{i,j,k}^l \\cdot a_{i+p,j+q,d}^{l-1}
                `}</BlockMath>
                
                <Text>
                  Which is equivalent to a cross-correlation between the error term <InlineMath>{`\\delta_{i,j,k}^l`}</InlineMath> and the activations <InlineMath>{`a_{i,j,d}^{l-1}`}</InlineMath>:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial w_{p,q,d,k}^l} = \\sum_{i,j} \\delta_{i,j,k}^l \\cdot a_{i+p,j+q,d}^{l-1} = (\\delta_k^l \\circledast a_d^{l-1})(p,q)
                `}</BlockMath>
                
                <Title order={4} mt="lg">Computing Gradients for Biases</Title>
                
                <Text>
                  For the bias terms:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial b_k^l} = \\sum_{i,j} \\frac{\\partial L}{\\partial z_{i,j,k}^l} \\frac{\\partial z_{i,j,k}^l}{\\partial b_k^l} = \\sum_{i,j} \\delta_{i,j,k}^l
                `}</BlockMath>
                
                <Text>
                  Since <InlineMath>{`\\frac{\\partial z_{i,j,k}^l}{\\partial b_k^l} = 1`}</InlineMath> for all positions (i,j).
                </Text>
                
                <Title order={4} mt="lg">Propagating Error to Previous Layer</Title>
                
                <Text>
                  To backpropagate through a convolutional layer, we need to compute <InlineMath>{`\\delta_{i,j,d}^{l-1}`}</InlineMath>:
                </Text>
                
                <BlockMath>{`
                  \\delta_{i,j,d}^{l-1} = \\frac{\\partial L}{\\partial a_{i,j,d}^{l-1}} = \\sum_{k=1}^{K^l} \\sum_{p,q} \\frac{\\partial L}{\\partial z_{p,q,k}^l} \\frac{\\partial z_{p,q,k}^l}{\\partial a_{i,j,d}^{l-1}}
                `}</BlockMath>
                
                <Text>
                  From the forward propagation equation:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial z_{p,q,k}^l}{\\partial a_{i,j,d}^{l-1}} = w_{i-p,j-q,d,k}^l
                `}</BlockMath>
                
                <Text>
                  Note that this is only non-zero when <InlineMath>{`i-p \\in [0, f_h-1]`}</InlineMath> and <InlineMath>{`j-q \\in [0, f_w-1]`}</InlineMath>.
                </Text>
                
                <Text>
                  Substituting:
                </Text>
                
                <BlockMath>{`
                  \\delta_{i,j,d}^{l-1} = \\sum_{k=1}^{K^l} \\sum_{p,q} \\delta_{p,q,k}^l \\cdot w_{i-p,j-q,d,k}^l
                `}</BlockMath>
                
                <Text>
                  This is equivalent to a full convolution (not cross-correlation) between the error term <InlineMath>{`\\delta_{p,q,k}^l`}</InlineMath> and the 180° rotated filters <InlineMath>{`w_{p,q,d,k}^l`}</InlineMath>:
                </Text>
                
                <BlockMath>{`
                  \\delta_{i,j,d}^{l-1} = \\sum_{k=1}^{K^l} (\\delta_k^l * w_{d,k}^l)(i,j)
                `}</BlockMath>
                
                <Text>
                  We must also account for the activation function by using the chain rule:
                </Text>
                
                <BlockMath>{`
                  \\delta_{i,j,d}^{l-1} = \\frac{\\partial L}{\\partial z_{i,j,d}^{l-1}} = \\frac{\\partial L}{\\partial a_{i,j,d}^{l-1}} \\frac{\\partial a_{i,j,d}^{l-1}}{\\partial z_{i,j,d}^{l-1}} = \\frac{\\partial L}{\\partial a_{i,j,d}^{l-1}} \\cdot g^{l-1\\prime}(z_{i,j,d}^{l-1})
                `}</BlockMath>
                
                <Text>
                  So the full expression becomes:
                </Text>
                
                <BlockMath>{`
                  \\delta_{i,j,d}^{l-1} = g^{l-1\\prime}(z_{i,j,d}^{l-1}) \\cdot \\sum_{k=1}^{K^l} (\\delta_k^l * w_{d,k}^l)(i,j)
                `}</BlockMath>
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="pooling-backprop">
            <Accordion.Control>
              <Title order={3}>Backpropagation Through Pooling Layers</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack spacing="md">
                <Title order={4}>Max Pooling</Title>
                
                <Text>
                  For max pooling, gradients are passed only to the neuron that achieved the maximum value during forward pass:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial a_{i,j,k}^{l-1}} = 
                  \\begin{cases} 
                  \\frac{\\partial L}{\\partial a_{\\lfloor i/k_h \\rfloor, \\lfloor j/k_w \\rfloor, k}^{l}} & \\text{if } a_{i,j,k}^{l-1} = \\max\\text{ in its pooling region} \\\\
                  0 & \\text{otherwise}
                  \\end{cases}
                `}</BlockMath>
                
                <Text>
                  This means that during backpropagation, max pooling acts as a switch or router that directs the gradient only to the neuron that provided the maximum value.
                </Text>
                
                <Title order={4} mt="lg">Average Pooling</Title>
                
                <Text>
                  For average pooling, gradients are distributed evenly among all neurons in the pooling region:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial a_{i,j,k}^{l-1}} = \\frac{1}{k_h \\times k_w} \\frac{\\partial L}{\\partial a_{\\lfloor i/k_h \\rfloor, \\lfloor j/k_w \\rfloor, k}^{l}}
                `}</BlockMath>
                
                <Text>
                  For a pooling window of size k_h×k_w, each input receives 1/(k_h×k_w) of the gradient from the output.
                </Text>
  
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>

      {/* The Full CNN Backpropagation Algorithm */}
      <Stack spacing="md">
        <Title order={2} id="cnn-backprop-algorithm">The CNN Backpropagation Algorithm</Title>
        
        <Title order={3}>Step 1: Forward Pass</Title>
        <List>
          <List.Item>Perform convolution operations in convolutional layers</List.Item>
          <List.Item>Apply activation functions</List.Item>
          <List.Item>Perform pooling operations</List.Item>
          <List.Item>Process through fully connected layers</List.Item>
          <List.Item>Compute the loss</List.Item>
        </List>
        
        <Title order={3} mt="lg">Step 2: Backward Pass</Title>
        <Text>
          Working backward from the output layer:
        </Text>
        
        <Title order={4} mt="sm">For Fully Connected Layers:</Title>
        <BlockMath>{`
          \\delta_j^l = g^{l\\prime}(z_j^l) \\sum_{k} w_{jk}^{l+1} \\delta_k^{l+1}
        `}</BlockMath>
        <BlockMath>{`
          \\frac{\\partial L}{\\partial w_{ij}^l} = \\delta_j^l a_i^{l-1}
        `}</BlockMath>
        
        <Title order={4} mt="sm">For Convolutional Layers:</Title>
        
        <Text>
          Using the provided operation definition:
        </Text>
        
        <Box className="p-4 bg-white rounded-md"> 
          <BlockMath>{`G[i, j, c_{out}] = \\sigma\\left(\\sum_{c_{in}=0}^{C_{in}-1} \\sum_{k=0}^{K_h-1} \\sum_{l=0}^{K_w-1} F[s_h \\cdot i + k - p_h, s_w \\cdot j + l - p_w, c_{in}] \\cdot K[k, l, c_{in}, c_{out}] + b[c_{out}]\\right)`}</BlockMath>
          
          <Text className="text-sm text-gray-600 mt-2">
            where <InlineMath>{"G[i,j,c_{out}]"}</InlineMath> is the output, <InlineMath>F</InlineMath> is the input tensor, <InlineMath>K</InlineMath> is the kernel, 
            <InlineMath>\sigma</InlineMath> is the activation function, with stride <InlineMath>s_h, s_w</InlineMath> and padding <InlineMath>p_h, p_w</InlineMath>.
          </Text>
        </Box>
        
        <Title order={5} mt="md">1. Gradient with respect to pre-activation output:</Title>
        <Text>
          Let's define the pre-activation output as <InlineMath>{`Z[i, j, c_{out}]`}</InlineMath> such that <InlineMath>{`G[i, j, c_{out}] = \\sigma(Z[i, j, c_{out}])`}</InlineMath>.
          The error term (gradient of loss with respect to pre-activation) at position <InlineMath>{`(i,j)`}</InlineMath> for output channel <InlineMath>{`c_{out}`}</InlineMath> is:
        </Text>
        
        <BlockMath>{`
          \\delta_{i,j,c_{out}}^l = \\frac{\\partial L}{\\partial Z[i, j, c_{out}]} = \\frac{\\partial L}{\\partial G[i, j, c_{out}]} \\cdot \\sigma'(Z[i, j, c_{out}])
        `}</BlockMath>
        
        <Title order={5} mt="md">2. Gradient with respect to kernel weights:</Title>
        <Text>
          For each weight <InlineMath>{`K[k, l, c_{in}, c_{out}]`}</InlineMath> in the kernel:
        </Text>
        
        <BlockMath>{`
          \\frac{\\partial L}{\\partial K[k, l, c_{in}, c_{out}]} = \\sum_i \\sum_j \\delta_{i,j,c_{out}}^l \\cdot F[s_h \\cdot i + k - p_h, s_w \\cdot j + l - p_w, c_{in}]
        `}</BlockMath>
        
        <Text>
          This can be expressed as a cross-correlation between the error terms and the input activations:
        </Text>
        
        <BlockMath>{`
          \\frac{\\partial L}{\\partial w_{p,q,d,k}^l} = (\\delta_k^l \\circledast a_d^{l-1})(p,q)
        `}</BlockMath>
        
        <Title order={5} mt="md">3. Gradient with respect to biases:</Title>
        <Text>
          For each bias <InlineMath>{`b[c_{out}]`}</InlineMath>:
        </Text>
        
        <BlockMath>{`
          \\frac{\\partial L}{\\partial b[c_{out}]} = \\sum_i \\sum_j \\delta_{i,j,c_{out}}^l
        `}</BlockMath>
        
        <Title order={5} mt="md">4. Gradient with respect to input:</Title>
        <Text>
          For each input element <InlineMath>{`F[i', j', c_{in}]`}</InlineMath>, considering stride and padding:
        </Text>
        
        <BlockMath>{`
          \\frac{\\partial L}{\\partial F[i', j', c_{in}]} = \\sum_{c_{out}} \\sum_k \\sum_l \\delta_{\\frac{i'+p_h-k}{s_h}, \\frac{j'+p_w-l}{s_w}, c_{out}}^l \\cdot K[k, l, c_{in}, c_{out}]
        `}</BlockMath>
        
        <Text>
          Where the indices <InlineMath>{`\\frac{i'+p_h-k}{s_h}`}</InlineMath> and <InlineMath>{`\\frac{j'+p_w-l}{s_w}`}</InlineMath> must be integers for the term to be included.
          This can be expressed using the convolution notation:
        </Text>
        
        <BlockMath>{`
          \\delta_{i,j,d}^{l-1} = g^{l-1\\prime}(z_{i,j,d}^{l-1}) \\cdot \\sum_{k=1}^{K^l} (\\delta_k^l * w_{d,k}^l)(i,j)
        `}</BlockMath>
        
        <Title order={4} mt="sm">For Pooling Layers:</Title>
        <Text>
          For max pooling, route gradients only to neurons that had the maximum value.
          For average pooling, distribute gradients evenly.
        </Text>
        
        <Title order={3} mt="lg">Step 3: Parameter Update</Title>
        <BlockMath>{`
          w_{p,q,d,k}^l \\leftarrow w_{p,q,d,k}^l - \\alpha \\frac{\\partial L}{\\partial w_{p,q,d,k}^l}
        `}</BlockMath>
        <BlockMath>{`
          b_k^l \\leftarrow b_k^l - \\alpha \\frac{\\partial L}{\\partial b_k^l}
        `}</BlockMath>
      </Stack>

    </Stack>
  );
};

export default CNNBackpropagation;