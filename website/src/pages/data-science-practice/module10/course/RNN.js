import React from 'react';
import { Title, Text, Stack, Grid, Box, List, Table, Divider, Accordion } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';
import RNNOutro from './RNNOutro';
const RNN = () => {
  return (
    <Stack spacing="xl" className="w-full">
      {/* Introduction to RNNs */}
      <Title order={1} id="rnn-introduction">Recurrent Neural Networks</Title>
      
      <Text>
        Recurrent Neural Networks (RNNs) learn from sequential data. Unlike feedforward networks, RNNs share parameters across different time steps, allowing them to process variable-length sequences efficiently.
      </Text>

      <Stack spacing="md">


        <Box className="p-4 border rounded">
          <Title order={4}>RNN Architecture units</Title>
          <List>
            <List.Item><strong>Standard RNN:</strong> Simple recurrent cells with a single state</List.Item>
            <List.Item><strong>LSTM (Long Short-Term Memory):</strong> Complex units with gates to control information flow</List.Item>
            <List.Item><strong>GRU (Gated Recurrent Unit):</strong> Simplified gating mechanism compared to LSTM</List.Item>
          </List>
          
        </Box>
      </Stack>

      {/* RNN Operation Notation */}
      <Stack spacing="md">
        <Title order={2} id="rnn-notation">RNN Notation</Title>
        <Text>
          The following notation will be used consistently across all RNN variants to describe forward and backward passes.
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
                <InlineMath>{`x^{(t)}`}</InlineMath>
              </Table.Td>
              <Table.Td>Input vector at time step t</Table.Td>
              <Table.Td>
                <InlineMath>{`x^{(t)} \\in \\mathbb{R}^{d_x}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`h^{(t)}`}</InlineMath>
              </Table.Td>
              <Table.Td>Hidden state at time step t</Table.Td>
              <Table.Td>
                <InlineMath>{`h^{(t)} \\in \\mathbb{R}^{d_h}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`c^{(t)}`}</InlineMath>
              </Table.Td>
              <Table.Td>Cell state at time step t (for LSTM)</Table.Td>
              <Table.Td>
                <InlineMath>{`c^{(t)} \\in \\mathbb{R}^{d_h}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`y^{(t)}`}</InlineMath>
              </Table.Td>
              <Table.Td>Output at time step t</Table.Td>
              <Table.Td>
                <InlineMath>{`y^{(t)} \\in \\mathbb{R}^{d_y}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`W_{xh}`}</InlineMath>
              </Table.Td>
              <Table.Td>Weight matrix connecting input to hidden state</Table.Td>
              <Table.Td>
                <InlineMath>{`W_{xh} \\in \\mathbb{R}^{d_h \\times d_x}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`W_{hh}`}</InlineMath>
              </Table.Td>
              <Table.Td>Weight matrix connecting previous hidden state to current hidden state</Table.Td>
              <Table.Td>
                <InlineMath>{`W_{hh} \\in \\mathbb{R}^{d_h \\times d_h}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`W_{hy}`}</InlineMath>
              </Table.Td>
              <Table.Td>Weight matrix connecting hidden state to output</Table.Td>
              <Table.Td>
                <InlineMath>{`W_{hy} \\in \\mathbb{R}^{d_y \\times d_h}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`b_h`}</InlineMath>
              </Table.Td>
              <Table.Td>Bias vector for hidden state</Table.Td>
              <Table.Td>
                <InlineMath>{`b_h \\in \\mathbb{R}^{d_h}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`b_y`}</InlineMath>
              </Table.Td>
              <Table.Td>Bias vector for output</Table.Td>
              <Table.Td>
                <InlineMath>{`b_y \\in \\mathbb{R}^{d_y}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`\\sigma`}</InlineMath>
              </Table.Td>
              <Table.Td>Sigmoid activation function</Table.Td>
              <Table.Td>
                <InlineMath>{`\\sigma(x) = \\frac{1}{1+e^{-x}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`\\tanh`}</InlineMath>
              </Table.Td>
              <Table.Td>Hyperbolic tangent activation function</Table.Td>
              <Table.Td>
                <InlineMath>{`\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`L`}</InlineMath>
              </Table.Td>
              <Table.Td>Loss function (typically averaged over time steps)</Table.Td>
              <Table.Td>
                <InlineMath>{`L = \\frac{1}{T}\\sum_{t=1}^{T}L^{(t)}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`\\odot`}</InlineMath>
              </Table.Td>
              <Table.Td>Element-wise (Hadamard) product</Table.Td>
              <Table.Td>
                <InlineMath>{`(a \\odot b)_i = a_i \\cdot b_i`}</InlineMath>
              </Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
        
      </Stack>
{/* RNN Input/Output Formal Definition */}
<Stack spacing="md">
  
  <Box className="p-4 border rounded bg-gray-50">
    <Title order={4}>RNN Function Definition</Title>
    <Text mt="md">
      An RNN can be formally defined as a function:
    </Text>
    
    <BlockMath>{`
      \\text{RNN}: \\mathbb{R}^{d_x \\times T} \\times \\mathbb{R}^{d_h} \\rightarrow \\mathbb{R}^{d_y \\times T} \\times \\mathbb{R}^{d_h}
    `}</BlockMath>
    
    <Text mt="md">
      That maps an input sequence and initial hidden state to an output sequence and final hidden state:
    </Text>
    
    <BlockMath>{`
      \\text{RNN}(\\{x^{(1)}, x^{(2)}, \\ldots, x^{(T)}\\}, h^{(0)}) = (\\{y^{(1)}, y^{(2)}, \\ldots, y^{(T)}\\}, h^{(T)})
    `}</BlockMath>
    
    <Text mt="md">
      Where:
    </Text>
    <List spacing="xs">
      <List.Item>
        <InlineMath>{`\\{x^{(1)}, x^{(2)}, \\ldots, x^{(T)}\\} \\in \\mathbb{R}^{d_x \\times T}`}</InlineMath> is the input sequence of length T with each <InlineMath>{`x^{(t)} \\in \\mathbb{R}^{d_x}`}</InlineMath>
      </List.Item>
      <List.Item>
        <InlineMath>{`h^{(0)} \\in \\mathbb{R}^{d_h}`}</InlineMath> is the initial hidden state
      </List.Item>
      <List.Item>
        <InlineMath>{`\\{y^{(1)}, y^{(2)}, \\ldots, y^{(T)}\\} \\in \\mathbb{R}^{d_y \\times T}`}</InlineMath> is the sequence of outputs with each <InlineMath>{`y^{(t)} \\in \\mathbb{R}^{d_y}`}</InlineMath>
      </List.Item>
      <List.Item>
        <InlineMath>{`h^{(T)} \\in \\mathbb{R}^{d_h}`}</InlineMath> is the final hidden state
      </List.Item>
    </List>
  </Box>
  
</Stack>

{/* RNN Unfolded Illustration */}
<Stack spacing="md">
  <Text>
    An RNN can be visualized as the same cell repeated for each time step, with information flowing from one step to the next through the hidden state.
  </Text>
  
  <Box className="flex justify-center p-4">
    <svg width="600" height="240" viewBox="0 0 600 240">
      {/* Arrow definitions */}
      <defs>
        <marker id="blue-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#007acc" />
        </marker>
        <marker id="red-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#d63031" />
        </marker>
        <marker id="green-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#00b894" />
        </marker>
      </defs>
      
      {/* Time step t=0 */}
      <rect x="50" y="100" width="120" height="80" rx="5" fill="#f0f0f0" stroke="#000" />
      <text x="110" y="145" textAnchor="middle" fontWeight="bold">RNN Cell</text>

      {/* Time step t=1 */}
      <rect x="250" y="100" width="120" height="80" rx="5" fill="#f0f0f0" stroke="#000" />
      <text x="310" y="145" textAnchor="middle" fontWeight="bold">RNN Cell</text>

      {/* Time step t=2 */}
      <rect x="450" y="100" width="120" height="80" rx="5" fill="#f0f0f0" stroke="#000" />
      <text x="510" y="145" textAnchor="middle" fontWeight="bold">RNN Cell</text>

      {/* Input arrows x^(t) */}
      <path d="M 110 60 L 110 100" stroke="#007acc" strokeWidth="2" fill="none" markerEnd="url(#blue-arrow)" />
      <text x="120" y="70" fill="#007acc" fontWeight="bold">x0</text>
      
      <path d="M 310 60 L 310 100" stroke="#007acc" strokeWidth="2" fill="none" markerEnd="url(#blue-arrow)" />
      <text x="320" y="70" fill="#007acc" fontWeight="bold">x1</text>
      
      <path d="M 510 60 L 510 100" stroke="#007acc" strokeWidth="2" fill="none" markerEnd="url(#blue-arrow)" />
      <text x="520" y="70" fill="#007acc" fontWeight="bold">x2</text>

      {/* Hidden state connections h^(t) */}
      <path d="M 170 140 L 250 140" stroke="#d63031" strokeWidth="2" fill="none" markerEnd="url(#red-arrow)" />
      <text x="210" y="130" fill="#d63031" fontWeight="bold">h0</text>
      
      <path d="M 370 140 L 450 140" stroke="#d63031" strokeWidth="2" fill="none" markerEnd="url(#red-arrow)" />
      <text x="410" y="130" fill="#d63031" fontWeight="bold">h1</text>
      
      <path d="M 570 140 L 590 140" stroke="#d63031" strokeWidth="2" fill="none" markerEnd="url(#red-arrow)" />
      <text x="580" y="130" fill="#d63031" fontWeight="bold">h2</text>
      
      {/* Output arrows y^(t) */}
      <path d="M 110 180 L 110 220" stroke="#00b894" strokeWidth="2" fill="none" markerEnd="url(#green-arrow)" />
      <text x="120" y="210" fill="#00b894" fontWeight="bold">y0</text>
      
      <path d="M 310 180 L 310 220" stroke="#00b894" strokeWidth="2" fill="none" markerEnd="url(#green-arrow)" />
      <text x="320" y="210" fill="#00b894" fontWeight="bold">y1</text>
      
      <path d="M 510 180 L 510 220" stroke="#00b894" strokeWidth="2" fill="none" markerEnd="url(#green-arrow)" />
      <text x="520" y="210" fill="#00b894" fontWeight="bold">y2</text>
    </svg>
  </Box>
  <Text>
    The same weights and biases are used at each time step, which allows RNNs to handle sequences of variable length. This is a key feature that makes RNNs well-suited for sequential data processing.
  </Text>
</Stack>
      {/* Forward Propagation in RNNs */}
      <Stack spacing="md">
        <Title order={2} id="units-architecture">Units Architecture</Title>
        
        <Title order={3}>Standard RNN</Title>
        <Text>
          The forward pass in a standard RNN cell is defined as:
        </Text>
        
        <BlockMath>{`
          h^{(t)} = \\tanh(W_{xh}x^{(t)} + W_{hh}h^{(t-1)} + b_h)
        `}</BlockMath>
        
        <BlockMath>{`
          y^{(t)} = f(W_{hy}h^{(t)} + b_y)
        `}</BlockMath>
        
        <Text>
          Where f is the output activation function (often softmax for classification or identity for regression).
        </Text>
        
        <Title order={3} mt="lg">LSTM (Long Short-Term Memory)</Title>
        <Text>
          LSTM introduces a cell state <InlineMath>{`c^{(t)}`}</InlineMath> and three gates to control information flow:
        </Text>
        
        <List>
          <List.Item><strong>Forget gate</strong> <InlineMath>{`f^{(t)}`}</InlineMath>: Controls what information to discard from cell state</List.Item>
          <List.Item><strong>Input gate</strong> <InlineMath>{`i^{(t)}`}</InlineMath>: Controls what new information to store in cell state</List.Item>
          <List.Item><strong>Output gate</strong> <InlineMath>{`o^{(t)}`}</InlineMath>: Controls what information to output based on cell state</List.Item>
        </List>
        
        <Text>
          The forward pass in an LSTM is defined as:
        </Text>
        
        <BlockMath>{`
          f^{(t)} = \\sigma(W_{xf}x^{(t)} + W_{hf}h^{(t-1)} + b_f)
        `}</BlockMath>
        
        <BlockMath>{`
          i^{(t)} = \\sigma(W_{xi}x^{(t)} + W_{hi}h^{(t-1)} + b_i)
        `}</BlockMath>
        
        <BlockMath>{`
          \\tilde{c}^{(t)} = \\tanh(W_{xc}x^{(t)} + W_{hc}h^{(t-1)} + b_c)
        `}</BlockMath>
        
        <BlockMath>{`
          c^{(t)} = f^{(t)} \\odot c^{(t-1)} + i^{(t)} \\odot \\tilde{c}^{(t)}
        `}</BlockMath>
        
        <BlockMath>{`
          o^{(t)} = \\sigma(W_{xo}x^{(t)} + W_{ho}h^{(t-1)} + b_o)
        `}</BlockMath>
        
        <BlockMath>{`
          h^{(t)} = o^{(t)} \\odot \\tanh(c^{(t)})
        `}</BlockMath>
        
        <BlockMath>{`
          y^{(t)} = f(W_{hy}h^{(t)} + b_y)
        `}</BlockMath>
        
        <Title order={3} mt="lg">GRU (Gated Recurrent Unit)</Title>
        <Text>
          GRU simplifies the LSTM architecture by combining the forget and input gates into a single "update gate" and merging the cell state and hidden state:
        </Text>
        
        <List>
          <List.Item><strong>Reset gate</strong> <InlineMath>{`r^{(t)}`}</InlineMath>: Controls how much of the previous state to forget</List.Item>
          <List.Item><strong>Update gate</strong> <InlineMath>{`z^{(t)}`}</InlineMath>: Controls how much of the new candidate state to use</List.Item>
        </List>
        
        <Text>
          The forward pass in a GRU is defined as:
        </Text>
        
        <BlockMath>{`
          z^{(t)} = \\sigma(W_{xz}x^{(t)} + W_{hz}h^{(t-1)} + b_z)
        `}</BlockMath>
        
        <BlockMath>{`
          r^{(t)} = \\sigma(W_{xr}x^{(t)} + W_{hr}h^{(t-1)} + b_r)
        `}</BlockMath>
        
        <BlockMath>{`
          \\tilde{h}^{(t)} = \\tanh(W_{xh}x^{(t)} + W_{hh}(r^{(t)} \\odot h^{(t-1)}) + b_h)
        `}</BlockMath>
        
        <BlockMath>{`
          h^{(t)} = (1 - z^{(t)}) \\odot h^{(t-1)} + z^{(t)} \\odot \\tilde{h}^{(t)}
        `}</BlockMath>
        
        <BlockMath>{`
          y^{(t)} = f(W_{hy}h^{(t)} + b_y)
        `}</BlockMath>
      </Stack>

      {/* Backpropagation Through Time */}
      <Stack spacing="md">
        <Title order={2} id="backpropagation-through-time">Backpropagation Through Time (BPTT)</Title>
        
        <Text>
          BPTT is the algorithm used for training RNNs. It extends standard backpropagation by unfolding the network through time and propagating gradients backward through the temporal dimension.
        </Text>
        
        <Box className="p-4 border rounded">
          <Title order={4}>Key Challenges in BPTT</Title>
          <List>
            <List.Item><strong>Vanishing gradients:</strong> Gradients can become extremely small when backpropagated through many time steps</List.Item>
            <List.Item><strong>Exploding gradients:</strong> Gradients can become extremely large when backpropagated through many time steps</List.Item>
            <List.Item><strong>Shared parameters:</strong> The same weights are used at each time step, requiring gradient accumulation</List.Item>
          </List>
        </Box>
        
        <Accordion variant="separated">
          <Accordion.Item value="general-bptt">
            <Accordion.Control>
              <Title order={3}>General BPTT Algorithm</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack spacing="md">
                <Text>
                  The general steps for BPTT are:
                </Text>
                
                <List ordered>
                  <List.Item>Perform forward propagation for all time steps t = 1 to T</List.Item>
                  <List.Item>Compute the loss L</List.Item>
                  <List.Item>Initialize gradients for all parameters to zero</List.Item>
                  <List.Item>For each time step t = T down to 1:
                    <List withPadding>
                      <List.Item>Compute gradients of loss with respect to output: <InlineMath>{`\\frac{\\partial L}{\\partial y^{(t)}}`}</InlineMath></List.Item>
                      <List.Item>Compute gradients of loss with respect to hidden state: <InlineMath>{`\\frac{\\partial L}{\\partial h^{(t)}}`}</InlineMath></List.Item>
                      <List.Item>Compute gradients for all parameters</List.Item>
                      <List.Item>Accumulate gradients across time steps</List.Item>
                    </List>
                  </List.Item>
                  <List.Item>Update parameters using accumulated gradients</List.Item>
                </List>
                
                <Text>
                  A key insight in BPTT is that we need to account for how the hidden state at time step t affects all future time steps:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial h^{(t)}} = \\frac{\\partial L^{(t)}}{\\partial h^{(t)}} + \\frac{\\partial L}{\\partial h^{(t+1)}} \\frac{\\partial h^{(t+1)}}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <Text>
                  Where <InlineMath>{`\\frac{\\partial L^{(t)}}{\\partial h^{(t)}}`}</InlineMath> is the direct impact on the current time step and the second term represents the impact on future time steps.
                </Text>
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="standard-rnn-backprop">
            <Accordion.Control>
              <Title order={3}>BPTT for Standard RNN</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack spacing="md">
                <Title order={4}>Computing Gradients</Title>
                
                <Text>
                  Let's define the error term at time step t as:
                </Text>
                
                <BlockMath>{`
                  \\delta^{(t)} = \\frac{\\partial L}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <Text>
                  For the output layer:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W_{hy}} = \\sum_{t=1}^{T} \\frac{\\partial L}{\\partial y^{(t)}} \\frac{\\partial y^{(t)}}{\\partial W_{hy}} = \\sum_{t=1}^{T} \\frac{\\partial L}{\\partial y^{(t)}} h^{(t)}
                `}</BlockMath>
                {/* #h^{(t)T  the T is for transpose but it was not clear */}
                <Accordion variant="separated" mb="md">
                  <Accordion.Item value="output-weight-derivation">
                    <Accordion.Control>
                      <Text size="sm" fw={500}>Detailed Proof of Output Weight Gradient</Text>
                    </Accordion.Control>
                    <Accordion.Panel>
                      <Stack spacing="md">
                        <Title order={5}>Step 1: Applying the Chain Rule</Title>
                        <Text>
                          The total loss L for a sequence is the sum of losses at each time step:
                        </Text>
                        <BlockMath>{`L = \\sum_{t=1}^{T} L^{(t)}`}</BlockMath>
                        
                        <Text>
                          Since the same weight matrix W_hy is shared across all time steps, we apply the chain rule:
                        </Text>
                        <BlockMath>{`\\frac{\\partial L}{\\partial W_{hy}} = \\frac{\\partial}{\\partial W_{hy}}\\left(\\sum_{t=1}^{T} L^{(t)}\\right)`}</BlockMath>
                        
                        <Text>
                          Since differentiation is a linear operation, we can move the derivative inside:
                        </Text>
                        <BlockMath>{`\\frac{\\partial L}{\\partial W_{hy}} = \\sum_{t=1}^{T} \\frac{\\partial L^{(t)}}{\\partial W_{hy}}`}</BlockMath>
                        
                        <Text>
                          For each time step t, W_hy affects L^(t) only through y^(t), so we apply the chain rule again:
                        </Text>
                        <BlockMath>{`\\frac{\\partial L^{(t)}}{\\partial W_{hy}} = \\frac{\\partial L^{(t)}}{\\partial y^{(t)}} \\frac{\\partial y^{(t)}}{\\partial W_{hy}}`}</BlockMath>
                        
                        <Text>
                          Since L = ∑L^(t), we have ∂L/∂y^(t) = ∂L^(t)/∂y^(t), giving us:
                        </Text>
                        <BlockMath>{`\\frac{\\partial L}{\\partial W_{hy}} = \\sum_{t=1}^{T} \\frac{\\partial L}{\\partial y^{(t)}} \\frac{\\partial y^{(t)}}{\\partial W_{hy}}`}</BlockMath>
                        
                        <Title order={5} mt="md">Step 2: Calculating the Derivative of Output</Title>
                        <Text>
                          The output at time step t is:
                        </Text>
                        <BlockMath>{`y^{(t)} = f(W_{hy}h^{(t)} + b_y)`}</BlockMath>
                        
                        <Text>
                          When we calculate the derivative of y^(t) with respect to any element W_hy[i,j]:
                        </Text>
                        <BlockMath>{`\\frac{\\partial y^{(t)}[i]}{\\partial W_{hy}[i,j]} = \\frac{\\partial f}{\\partial z^{(t)}[i]} \\cdot h^{(t)}[j]`}</BlockMath>
                        
                        <Text>
                          When organized into matrix form, and typically incorporating the activation function's derivative into the ∂L/∂y^(t) term:
                        </Text>
                        <BlockMath>{`\\frac{\\partial y^{(t)}}{\\partial W_{hy}} = h^{(t)T}`}</BlockMath>
                        
                        <Text>
                          Therefore:
                        </Text>
                        <BlockMath>{`\\frac{\\partial L}{\\partial W_{hy}} = \\sum_{t=1}^{T} \\frac{\\partial L}{\\partial y^{(t)}} h^{(t)T}`}</BlockMath>
                        
                        <Text>
                          This represents a sum of outer products between the output error vectors and the hidden state vectors across all time steps.
                        </Text>
                      </Stack>
                    </Accordion.Panel>
                  </Accordion.Item>
                </Accordion>
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial b_y} = \\sum_{t=1}^{T} \\frac{\\partial L}{\\partial y^{(t)}}
                `}</BlockMath>
                
                <Text>
                  For the hidden layer, we need to backpropagate through time. For each time step t, we compute:
                </Text>
                
                <BlockMath>{`
                  \\delta^{(t)} = \\frac{\\partial L}{\\partial y^{(t)}} \\frac{\\partial y^{(t)}}{\\partial h^{(t)}} + \\delta^{(t+1)} \\frac{\\partial h^{(t+1)}}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <Text>
                  Where <InlineMath>{`\\frac{\\partial y^{(t)}}{\\partial h^{(t)}} = W_{hy}^T`}</InlineMath> and <InlineMath>{`\\frac{\\partial h^{(t+1)}}{\\partial h^{(t)}} = W_{hh}^T \\text{diag}(1 - (h^{(t+1)})^2)`}</InlineMath> for tanh activation.
                </Text>
                
                <Text>
                  For the recurrent weights:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W_{xh}} = \\sum_{t=1}^{T} \\delta^{(t)} \\text{diag}(1 - (h^{(t)})^2) x^{(t)T}
                `}</BlockMath>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W_{hh}} = \\sum_{t=1}^{T} \\delta^{(t)} \\text{diag}(1 - (h^{(t)})^2) h^{(t-1)T}
                `}</BlockMath>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial b_h} = \\sum_{t=1}^{T} \\delta^{(t)} \\text{diag}(1 - (h^{(t)})^2)
                `}</BlockMath>
                
                <Title order={4} mt="lg">Vanishing and Exploding Gradient Problem</Title>
                
                <Text>
                  In standard RNNs, as we backpropagate through time, the gradient contribution from earlier time steps can become exponentially small or large:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial h^{(t)}}{\\partial h^{(t-k)}} = \\prod_{i=t-k+1}^{t} \\frac{\\partial h^{(i)}}{\\partial h^{(i-1)}} = \\prod_{i=t-k+1}^{t} W_{hh}^T \\text{diag}(1 - (h^{(i)})^2)
                `}</BlockMath>
                
                <Text>
                  This product of matrices can have eigenvalues greater than 1 (causing exploding gradients) or less than 1 (causing vanishing gradients). This is the primary motivation for developing architectures like LSTM and GRU.
                </Text>
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="lstm-backprop">
            <Accordion.Control>
              <Title order={3}>BPTT for LSTM</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack spacing="md">
                <Title order={4}>LSTM Gradient Flow</Title>
                
                <Text>
                  LSTM addresses the vanishing gradient problem through its cell state, which creates a highway for gradient flow. During backpropagation, we need to compute gradients for all gates and states.
                </Text>
                
                <Text>
                  Let's define the error terms:
                </Text>
                
                <BlockMath>{`
                  \\delta_h^{(t)} = \\frac{\\partial L}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <BlockMath>{`
                  \\delta_c^{(t)} = \\frac{\\partial L}{\\partial c^{(t)}}
                `}</BlockMath>
                
                <Text>
                  Starting with the output, we backpropagate to the hidden state:
                </Text>
                
                <BlockMath>{`
                  \\delta_h^{(t)} = \\frac{\\partial L}{\\partial y^{(t)}} \\frac{\\partial y^{(t)}}{\\partial h^{(t)}} + \\delta_h^{(t+1)} \\frac{\\partial h^{(t+1)}}{\\partial h^{(t)}} + \\delta_c^{(t+1)} \\frac{\\partial c^{(t+1)}}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <Text>
                  The gradient flow to the cell state is:
                </Text>
                
                <BlockMath>{`
                  \\delta_c^{(t)} = \\delta_h^{(t)} \\frac{\\partial h^{(t)}}{\\partial c^{(t)}} + \\delta_c^{(t+1)} \\frac{\\partial c^{(t+1)}}{\\partial c^{(t)}}
                `}</BlockMath>
                
                <Text>
                  Where <InlineMath>{`\\frac{\\partial h^{(t)}}{\\partial c^{(t)}} = o^{(t)} \\odot (1 - \\tanh^2(c^{(t)}))`}</InlineMath> and <InlineMath>{`\\frac{\\partial c^{(t+1)}}{\\partial c^{(t)}} = f^{(t+1)}`}</InlineMath>.
                </Text>
                
                <Text>
                  For the forget gate:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial f^{(t)}} = \\delta_c^{(t)} \\odot c^{(t-1)}
                `}</BlockMath>
                
                <Text>
                  For the input gate:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial i^{(t)}} = \\delta_c^{(t)} \\odot \\tilde{c}^{(t)}
                `}</BlockMath>
                
                <Text>
                  For the cell candidate:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial \\tilde{c}^{(t)}} = \\delta_c^{(t)} \\odot i^{(t)}
                `}</BlockMath>
                
                <Text>
                  For the output gate:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial o^{(t)}} = \\delta_h^{(t)} \\odot \\tanh(c^{(t)})
                `}</BlockMath>
                
                <Text>
                  These gradients are then backpropagated to the weights for each gate. For example, for the forget gate weights:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W_{xf}} = \\sum_{t=1}^{T} \\frac{\\partial L}{\\partial f^{(t)}} \\frac{\\partial f^{(t)}}{\\partial W_{xf}} = \\sum_{t=1}^{T} (\\delta_c^{(t)} \\odot c^{(t-1)}) \\odot (f^{(t)} \\odot (1 - f^{(t)})) x^{(t)T}
                `}</BlockMath>
                
                <Title order={4} mt="lg">Constant Error Carousel</Title>
                
                <Text>
                  A key insight of the LSTM is the "constant error carousel" formed by the cell state. As we backpropagate through time:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial c^{(t)}}{\\partial c^{(t-k)}} = \\prod_{i=t-k+1}^{t} \\frac{\\partial c^{(i)}}{\\partial c^{(i-1)}} = \\prod_{i=t-k+1}^{t} f^{(i)}
                `}</BlockMath>
                
                <Text>
                  If the forget gates <InlineMath>{`f^{(i)}`}</InlineMath> are close to 1, this product doesn't vanish, allowing gradients to flow through many time steps.
                </Text>
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
          
          <Accordion.Item value="gru-backprop">
            <Accordion.Control>
              <Title order={3}>BPTT for GRU</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack spacing="md">
                <Title order={4}>GRU Gradient Flow</Title>
                
                <Text>
                  The GRU uses a simpler architecture than the LSTM but still effectively mitigates the vanishing gradient problem. During backpropagation, we need to compute gradients for the reset and update gates.
                </Text>
                
                <Text>
                  Let's define the error term:
                </Text>
                
                <BlockMath>{`
                  \\delta_h^{(t)} = \\frac{\\partial L}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <Text>
                  The backpropagation through time follows:
                </Text>
                
                <BlockMath>{`
                  \\delta_h^{(t)} = \\frac{\\partial L}{\\partial y^{(t)}} \\frac{\\partial y^{(t)}}{\\partial h^{(t)}} + \\delta_h^{(t+1)} \\frac{\\partial h^{(t+1)}}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <Text>
                  For the GRU, the term <InlineMath>{`\\frac{\\partial h^{(t+1)}}{\\partial h^{(t)}}`}</InlineMath> includes both the direct path through <InlineMath>{`(1 - z^{(t+1)})`}</InlineMath> and the path through <InlineMath>{`\\tilde{h}^{(t+1)}`}</InlineMath>:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial h^{(t+1)}}{\\partial h^{(t)}} = (1 - z^{(t+1)}) + z^{(t+1)} \\frac{\\partial \\tilde{h}^{(t+1)}}{\\partial h^{(t)}}
                `}</BlockMath>
                
                <Text>
                  For the update gate:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial z^{(t)}} = \\delta_h^{(t)} \\odot (\\tilde{h}^{(t)} - h^{(t-1)})
                `}</BlockMath>
                
                <Text>
                  For the reset gate:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial r^{(t)}} = \\delta_h^{(t)} \\odot z^{(t)} \\odot (1 - \\tilde{h}^{(t)2}) \\odot (W_{hh} h^{(t-1)})
                `}</BlockMath>
                
                <Text>
                  For the candidate hidden state:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial \\tilde{h}^{(t)}} = \\delta_h^{(t)} \\odot z^{(t)}
                `}</BlockMath>
                
                <Text>
                  These gradients are then backpropagated to the corresponding weights. For example, for the update gate weights:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W_{xz}} = \\sum_{t=1}^{T} \\frac{\\partial L}{\\partial z^{(t)}} \\frac{\\partial z^{(t)}}{\\partial W_{xz}} = \\sum_{t=1}^{T} (\\delta_h^{(t)} \\odot (\\tilde{h}^{(t)} - h^{(t-1)})) \\odot (z^{(t)} \\odot (1 - z^{(t)})) x^{(t)T}
                `}</BlockMath>
                
                <Title order={4} mt="lg">Vanishing Gradient Mitigation in GRU</Title>
                
                <Text>
                  Similar to LSTM, GRU mitigates the vanishing gradient problem through its update gate:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial h^{(t)}}{\\partial h^{(t-k)}} = \\prod_{i=t-k+1}^{t} \\frac{\\partial h^{(i)}}{\\partial h^{(i-1)}}
                `}</BlockMath>
                
                <Text>
                  When the update gates <InlineMath>{`z^{(i)}`}</InlineMath> are close to 0, this product includes terms close to 1, creating a highway for gradient flow through time. This mechanism allows GRU to capture long-term dependencies effectively while using fewer parameters than LSTM.
                </Text>
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>
<RNNOutro />
    </Stack>
  );
};

export default RNN;