import React from 'react';
import { Title, Text, Stack, Grid, Box, List, Table, Divider, Accordion } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

const TransformerComponents = () => {
  return (
    <Stack spacing="xl" className="w-full">
      {/* Introduction to Transformers */}
      <Title order={1} id="transformer-introduction">Transformers: Components</Title>
      
      <Stack spacing="md">
        <Text>
          Transformers are a type of neural network architecture that revolutionized natural language processing and many other sequence-based tasks. Unlike RNNs, transformers process entire sequences in parallel using self-attention mechanisms, allowing them to capture long-range dependencies more effectively while enabling highly parallelized computation.
        </Text>

        <Box className="p-4 border rounded">
          <Title order={4}>Transformer Architecture Components</Title>
          <List>
            <List.Item><strong>Self-Attention:</strong> Mechanisms that weigh the importance of different elements in a sequence</List.Item>
            <List.Item><strong>Multi-Head Attention:</strong> Multiple parallel attention computations for richer representations</List.Item>
            <List.Item><strong>Position Encoding:</strong> Information about token positions in the sequence</List.Item>
            <List.Item><strong>Feed-Forward Networks:</strong> Position-wise fully connected networks</List.Item>
            <List.Item><strong>Layer Normalization:</strong> Normalization applied to stabilize training</List.Item>
            <List.Item><strong>Residual Connections:</strong> Skip connections that help gradient flow</List.Item>
          </List>
          
          <Text mt="md">The transformer consists of an encoder and a decoder stack, though many modern implementations use encoder-only (BERT), decoder-only (GPT), or encoder-decoder (T5) architectures.</Text>
        </Box>
      </Stack>

      {/* Transformer Mathematical Notation */}
      <Stack spacing="md">
        <Title order={2} id="transformer-notation">Transformer Mathematical Notation</Title>
        <Text>
          The following notation will be used to describe the forward and backward passes in a transformer:
        </Text>
        <Table withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Symbol</Table.Th>
              <Table.Th>Description</Table.Th>
              <Table.Th>Formula/Dimensions</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`X`}</InlineMath>
              </Table.Td>
              <Table.Td>Input token embeddings</Table.Td>
              <Table.Td>
                <InlineMath>{`X \\in \\mathbb{R}^{n \\times d_{model}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`P_E`}</InlineMath>
              </Table.Td>
              <Table.Td>Positional encodings</Table.Td>
              <Table.Td>
                <InlineMath>{`P_E \\in \\mathbb{R}^{n \\times d_{model}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`Z^l`}</InlineMath>
              </Table.Td>
              <Table.Td>Output of layer l</Table.Td>
              <Table.Td>
                <InlineMath>{`Z^l \\in \\mathbb{R}^{n \\times d_{model}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`Q, K, V`}</InlineMath>
              </Table.Td>
              <Table.Td>Query, Key, and Value matrices</Table.Td>
              <Table.Td>
                <InlineMath>{`Q, K, V \\in \\mathbb{R}^{n \\times d_k}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`W^Q_i, W^K_i, W^V_i`}</InlineMath>
              </Table.Td>
              <Table.Td>Projection matrices for head i</Table.Td>
              <Table.Td>
                <InlineMath>{`W^Q_i, W^K_i, W^V_i \\in \\mathbb{R}^{d_{model} \\times d_k}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`W^O`}</InlineMath>
              </Table.Td>
              <Table.Td>Output projection matrix for multi-head attention</Table.Td>
              <Table.Td>
                <InlineMath>{`W^O \\in \\mathbb{R}^{h \\cdot d_k \\times d_{model}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`W^{FF}_1, W^{FF}_2`}</InlineMath>
              </Table.Td>
              <Table.Td>Feed-forward network weight matrices</Table.Td>
              <Table.Td>
                <InlineMath>{`W^{FF}_1 \\in \\mathbb{R}^{d_{model} \\times d_{ff}}, W^{FF}_2 \\in \\mathbb{R}^{d_{ff} \\times d_{model}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`\\gamma, \\beta`}</InlineMath>
              </Table.Td>
              <Table.Td>Layer normalization parameters</Table.Td>
              <Table.Td>
                <InlineMath>{`\\gamma, \\beta \\in \\mathbb{R}^{d_{model}}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`A`}</InlineMath>
              </Table.Td>
              <Table.Td>Attention weight matrix</Table.Td>
              <Table.Td>
                <InlineMath>{`A \\in \\mathbb{R}^{n \\times n}`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`h`}</InlineMath>
              </Table.Td>
              <Table.Td>Number of attention heads</Table.Td>
              <Table.Td>
                <InlineMath>{`h \\in \\mathbb{Z}^+`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`n`}</InlineMath>
              </Table.Td>
              <Table.Td>Sequence length</Table.Td>
              <Table.Td>
                <InlineMath>{`n \\in \\mathbb{Z}^+`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`d_{model}`}</InlineMath>
              </Table.Td>
              <Table.Td>Model dimension</Table.Td>
              <Table.Td>
                <InlineMath>{`d_{model} \\in \\mathbb{Z}^+`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`d_k, d_v`}</InlineMath>
              </Table.Td>
              <Table.Td>Dimensions of keys/queries and values</Table.Td>
              <Table.Td>
                <InlineMath>{`d_k, d_v \\in \\mathbb{Z}^+, \\ d_k = d_v = d_{model}/h`}</InlineMath>
              </Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>
                <InlineMath>{`d_{ff}`}</InlineMath>
              </Table.Td>
              <Table.Td>Feed-forward network inner dimension</Table.Td>
              <Table.Td>
                <InlineMath>{`d_{ff} \\in \\mathbb{Z}^+`}</InlineMath>
              </Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>
        
        <Text>
          Note: In multi-head attention, we typically have <InlineMath>{`d_k = d_v = d_{model}/h`}</InlineMath>, where h is the number of attention heads.
        </Text>
      </Stack>

      {/* Forward Propagation in Transformers */}
      <Stack spacing="md">
        <Title order={2} id="forward-propagation">Forward Propagation in Transformers</Title>
        
        <Title order={3}>Input Embedding and Positional Encoding</Title>
        <Text>
          The first step is to convert input tokens into embeddings and add positional information:
        </Text>
        
        <BlockMath>{`
          Z^0 = X + P_E
        `}</BlockMath>
        
        <Text>
          Where X is the token embeddings and P_E is the positional encoding. The original transformer uses sinusoidal position embeddings:
        </Text>
        
        <BlockMath>{`
          P_{E_{(pos, 2i)}} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)
        `}</BlockMath>
        
        <BlockMath>{`
          P_{E_{(pos, 2i+1)}} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)
        `}</BlockMath>
        
        <Title order={3} mt="lg">Self-Attention Mechanism</Title>
        <Text>
          Self-attention allows the model to weigh the importance of different tokens in the sequence:
        </Text>
        
        <BlockMath>{`
          \\begin{align}
          \\text{Attention}(Q, K, V) &= \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V \\\\
          &= \\text{softmax}\\left(\\frac{\\text{Similarity Scores}}{\\text{Scaling Factor}}\\right) \\times \\text{Values} \\\\
          &= \\text{Attention Weights} \\times \\text{Values}
          \\end{align}
        `}</BlockMath>

<Text>
  Raw similarity/attention scores are represented as a matrix <InlineMath>{`S = QK^T \\in \\mathbb{R}^{n \\times n}`}</InlineMath>, where each element <InlineMath>{`S_{ij}`}</InlineMath> quantifies how much position <InlineMath>{`i`}</InlineMath> attends to position <InlineMath>{`j`}</InlineMath>. Higher values of <InlineMath>{`S_{ij}`}</InlineMath> indicate stronger attention from token <InlineMath>{`i`}</InlineMath> to token <InlineMath>{`j`}</InlineMath>.
</Text>
        
        <Text mt="md">
          Where:
        </Text>
        <List>
          <List.Item><InlineMath math="Q \in \mathbb{R}^{n \times d_k}"/>: Query matrix (what we're looking for)</List.Item>
          <List.Item><InlineMath math="K \in \mathbb{R}^{n \times d_k}"/>: Key matrix (what information is available)</List.Item>
          <List.Item><InlineMath math="V \in \mathbb{R}^{n \times d_v}"/>: Value matrix (actual content to retrieve)</List.Item>
          <List.Item><InlineMath math="QK^T \in \mathbb{R}^{n \times n}"/>: Matrix of similarity scores between positions</List.Item>
          <List.Item><InlineMath math="\sqrt{d_k}"/>: Scaling factor that stabilizes gradients</List.Item>
          <List.Item><InlineMath math="\text{softmax}(\frac{QK^T}{\sqrt{d_k}}) \in \mathbb{R}^{n \times n}"/>: Attention weights (probabilities)</List.Item>
        </List>
        
        <Text>
          Where Q, K, and V are the query, key, and value matrices derived from the input:
        </Text>
        
        <BlockMath>{`
          Q = Z^{l-1}W^Q
        `}</BlockMath>
        
        <BlockMath>{`
          K = Z^{l-1}W^K
        `}</BlockMath>
        
        <BlockMath>{`
          V = Z^{l-1}W^V
        `}</BlockMath>
        
        <Title order={3} mt="lg">Multi-Head Attention</Title>
        <Text>
          Multi-head attention performs attention multiple times in parallel, allowing the model to attend to information from different representation subspaces:
        </Text>
        
        <BlockMath>{`
          \\text{MultiHead}(Z^{l-1}) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)W^O
        `}</BlockMath>
        
        <BlockMath>{`
          \\text{head}_i = \\text{Attention}(Z^{l-1}W^Q_i, Z^{l-1}W^K_i, Z^{l-1}W^V_i)
        `}</BlockMath>
        
        <Title order={3} mt="lg">Layer Normalization</Title>
        <Text>
          Layer normalization helps stabilize the training of deep networks:
        </Text>
        
        <BlockMath>{`
          \\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta
        `}</BlockMath>
        
        <Text>
          Where <InlineMath>{`\\mu`}</InlineMath> and <InlineMath>{`\\sigma`}</InlineMath> are the mean and standard deviation computed across the feature dimension.
        </Text>
        
        <Title order={3} mt="lg">Position-wise Feed-Forward Network</Title>
        <Text>
          Each position in the sequence passes through the same feed-forward network:
        </Text>
        
        <BlockMath>{`
          \\text{FFN}(x) = \\max(0, xW^{FF}_1 + b_1)W^{FF}_2 + b_2
        `}</BlockMath>
        
        <Text>
          This is equivalent to a two-layer neural network with a ReLU activation function in between:
        </Text>
        
        <List>
          <List.Item>First layer: Linear transformation <InlineMath math="xW^{FF}_1 + b_1"/> </List.Item>
          <List.Item>ReLU activation: <InlineMath math="\max(0, xW^{FF}_1 + b_1)"/> </List.Item>
          <List.Item>Second layer: Linear transformation <InlineMath math="\max(0, xW^{FF}_1 + b_1)W^{FF}_2 + b_2"/> </List.Item>
        </List>
        
        <Title order={3} mt="lg">Encoder Layer</Title>
        <Text>
          The complete forward pass through one encoder layer is:
        </Text>
        
        <BlockMath>{`
          Z^{l'} = \\text{LayerNorm}(Z^{l-1} + \\text{MultiHead}(Z^{l-1}))
        `}</BlockMath>
        
        <BlockMath>{`
          Z^l = \\text{LayerNorm}(Z^{l'} + \\text{FFN}(Z^{l'}))
        `}</BlockMath>

        <Title order={3} mt="lg">Masked Multi-Head Attention</Title>
        <Text>
          Masked Multi-Head Attention is used in the decoder to prevent positions from attending to future positions (auto-regressive property). This is achieved by masking out (setting to -∞) all positions in the future before the softmax step:
        </Text>
        
        <BlockMath>{`
          \\text{MaskedMultiHead}(Y^{l-1}) = \\text{Concat}(\\text{masked\\_head}_1, \\ldots, \\text{masked\\_head}_h)W^O
        `}</BlockMath>
        
        <BlockMath>{`
          \\text{masked\\_head}_i = \\text{MaskedAttention}(Y^{l-1}W^Q_i, Y^{l-1}W^K_i, Y^{l-1}W^V_i)
        `}</BlockMath>
        
        <BlockMath>{`
          \\text{MaskedAttention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T + M}{\\sqrt{d_k}}\\right)V
        `}</BlockMath>
        
        <Text>
          Where <InlineMath>{`M \\in \\mathbb{R}^{n \\times n}`}</InlineMath> is the mask matrix with:
        </Text>
        <BlockMath>{`
          M_{ij} = 
          \\begin{cases} 
          0, & \\text{if } i \\geq j \\\\
          -\\infty, & \\text{if } i < j
          \\end{cases}
        `}</BlockMath>
        
        <Title order={3} mt="lg">Cross-Attention</Title>
        <Text>
          Cross-Attention connects the decoder with the encoder, allowing the decoder to attend to all positions in the encoder output. The queries come from the previous decoder layer, and the keys and values come from the encoder output:
        </Text>
        
        <BlockMath>{`
          \\text{CrossAttention}(Y^{l'}, Z^L) = \\text{Concat}(\\text{cross\\_head}_1, \\ldots, \\text{cross\\_head}_h)W^O
        `}</BlockMath>
        
        <BlockMath>{`
          \\text{cross\\_head}_i = \\text{Attention}(Y^{l'}W^Q_i, Z^LW^K_i, Z^LW^V_i)
        `}</BlockMath>
        
        <Text>
          In this case, queries <InlineMath>{`Q = Y^{l'}W^Q_i`}</InlineMath> come from the decoder, while keys <InlineMath>{`K = Z^LW^K_i`}</InlineMath> and values <InlineMath>{`V = Z^LW^V_i`}</InlineMath> come from the encoder output.
        </Text>
        
        <Title order={3} mt="lg">Decoder Layer</Title>
        <Text>
          The decoder layer includes a masked self-attention mechanism and cross-attention:
        </Text>
        
        <BlockMath>{`
          Y^{l'} = \\text{LayerNorm}(Y^{l-1} + \\text{MaskedMultiHead}(Y^{l-1}))
        `}</BlockMath>
        
        <BlockMath>{`
          Y^{l''} = \\text{LayerNorm}(Y^{l'} + \\text{CrossAttention}(Y^{l'}, Z^L))
        `}</BlockMath>
        
        <BlockMath>{`
          Y^l = \\text{LayerNorm}(Y^{l''} + \\text{FFN}(Y^{l''}))
        `}</BlockMath>
        
        <Text>
          Where <InlineMath>{`Z^L`}</InlineMath> is the output from the last encoder layer.
        </Text>
        
        <Title order={3} mt="lg">Output Layer</Title>
        <Text>
          The final output layer converts the decoder's output to logits:
        </Text>
        
        <BlockMath>{`
          \\text{Output} = Y^LW^{out} + b^{out}
        `}</BlockMath>
      </Stack>

      {/* Backpropagation in Transformers */}
      <Stack spacing="md">
        <Title order={2} id="backprop-transformers">Backpropagation in Transformers</Title>
        
        <Text>
          Backpropagation in transformers follows the standard gradient descent procedure with a few key differences due to the attention mechanisms and layer normalization.
        </Text>
        
        <Box className="p-4 border rounded">
          <Title order={4}>Key Characteristics of Transformer Backpropagation</Title>
          <List>
            <List.Item><strong>Parallelization:</strong> Unlike RNN backpropagation, transformer backpropagation can be fully parallelized across sequence elements</List.Item>
            <List.Item><strong>Attention gradient flow:</strong> Gradients flow between all positions, creating rich paths for backpropagation</List.Item>
            <List.Item><strong>Residual connections:</strong> Help mitigate vanishing gradients by providing direct gradient paths</List.Item>
            <List.Item><strong>Layer normalization:</strong> Stabilizes gradient magnitudes during backpropagation</List.Item>
          </List>
        </Box>
        
        <Accordion variant="separated">
          
          <Accordion.Item value="attention-backprop">
            <Accordion.Control>
              <Title order={3}>Backpropagation Through Attention</Title>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack spacing="md">
                <Title order={4}>Computing Gradients in Self-Attention</Title>
                
                <Text>
                  Let's define the steps in the attention mechanism:
                </Text>
                
                <BlockMath>{`
                  S = QK^T
                `}</BlockMath>
                
                <BlockMath>{`
                  S_{scaled} = \\frac{S}{\\sqrt{d_k}}
                `}</BlockMath>
                
                <BlockMath>{`
                  A = \\text{softmax}(S_{scaled})
                `}</BlockMath>
                
                <BlockMath>{`
                  O = AV
                `}</BlockMath>
                
                <Text>
                  Given <InlineMath>{`\\frac{\\partial L}{\\partial O}`}</InlineMath>, we need to compute gradients with respect to A, S, Q, K, and V.
                </Text>
                
                <Text>
                  First, the gradient with respect to V:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial V} = A^T \\frac{\\partial L}{\\partial O}
                `}</BlockMath>
                
                <Text>
                  Next, the gradient with respect to A:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial A} = \\frac{\\partial L}{\\partial O} V^T
                `}</BlockMath>
                
                <Text>
                  The gradient through the softmax function is a bit complex:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial S_{scaled}} = \\frac{\\partial L}{\\partial A} \\odot \\frac{\\partial A}{\\partial S_{scaled}} = \\frac{\\partial L}{\\partial A} \\odot (A - A \\odot A^T\\mathbf{1})
                `}</BlockMath>
                
                <Text>
                  The gradient with respect to the scaled scores:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial S} = \\frac{1}{\\sqrt{d_k}} \\frac{\\partial L}{\\partial S_{scaled}}
                `}</BlockMath>
                
                <Text>
                  Finally, gradients with respect to Q and K:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial Q} = \\frac{\\partial L}{\\partial S} K
                `}</BlockMath>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial K} = \\left(\\frac{\\partial L}{\\partial S}\\right)^T Q
                `}</BlockMath>
                
                <Text>
                  These gradients are then used to compute the gradients with respect to the weight matrices:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W^Q} = Z^{l-1^T} \\frac{\\partial L}{\\partial Q}
                `}</BlockMath>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W^K} = Z^{l-1^T} \\frac{\\partial L}{\\partial K}
                `}</BlockMath>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W^V} = Z^{l-1^T} \\frac{\\partial L}{\\partial V}
                `}</BlockMath>
                
                <Title order={4} mt="lg">Multi-Head Attention Gradients</Title>
                
                <Text>
                  For multi-head attention, we compute gradients separately for each head and then aggregate them:
                </Text>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial \\text{head}_i} = \\frac{\\partial L}{\\partial \\text{MultiHead}} W^{O^T}[:, i \\cdot d_k : (i+1) \\cdot d_k]
                `}</BlockMath>
                
                <BlockMath>{`
                  \\frac{\\partial L}{\\partial W^O} = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)^T \\frac{\\partial L}{\\partial \\text{MultiHead}}
                `}</BlockMath>
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>

        </Accordion>
      </Stack>

      {/* The Full Transformer Backpropagation Algorithm */}
      <Stack spacing="md">
        <Title order={2} id="transformer-backprop-algorithm">The Complete Transformer Backpropagation Algorithm</Title>
        
        <Title order={3}>Step 1: Forward Pass</Title>
        <Text>
          The forward pass proceeds through all layers of the encoder and decoder as described in the forward propagation section.
        </Text>
        <List>
          <List.Item>Compute token embeddings and add positional encodings</List.Item>
          <List.Item>Process through each encoder layer (self-attention, feed-forward network)</List.Item>
          <List.Item>Process through each decoder layer (masked self-attention, cross-attention, feed-forward network)</List.Item>
          <List.Item>Compute final outputs and loss</List.Item>
        </List>
        
        <Title order={3} mt="lg">Step 2: Backward Pass</Title>
        <Text>
          Initialize gradients for all parameters to zero.
        </Text>
        <Text>
          For each layer from the last to the first:
        </Text>
        
        <Title order={4} mt="sm">Decoder Output Layer:</Title>
        <List>
          <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial \\text{Output}}`}</InlineMath> based on the loss function</List.Item>
          <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial W^{out}}`}</InlineMath> and <InlineMath>{`\\frac{\\partial L}{\\partial b^{out}}`}</InlineMath></List.Item>
          <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Y^L}`}</InlineMath></List.Item>
        </List>
        
        <Title order={4} mt="sm">For Each Decoder Layer (from L to 1):</Title>
        <List>
          <List.Item>Backpropagate through the layer normalization after FFN:
            <List withPadding>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Y^{l''}}`}</InlineMath> and <InlineMath>{`\\frac{\\partial L}{\\partial \\text{FFN}(Y^{l''})}`}</InlineMath></List.Item>
              <List.Item>Update gradients for <InlineMath>{`\\gamma`}</InlineMath> and <InlineMath>{`\\beta`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the feed-forward network:
            <List withPadding>
              <List.Item>Compute gradients for <InlineMath>{`W^{FF}_1`}</InlineMath>, <InlineMath>{`W^{FF}_2`}</InlineMath>, <InlineMath>{`b_1`}</InlineMath>, and <InlineMath>{`b_2`}</InlineMath></List.Item>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Y^{l''}}`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the layer normalization after cross-attention:
            <List withPadding>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Y^{l'}}`}</InlineMath> and <InlineMath>{`\\frac{\\partial L}{\\partial \\text{CrossAttention}(Y^{l'}, Z^L)}`}</InlineMath></List.Item>
              <List.Item>Update gradients for <InlineMath>{`\\gamma`}</InlineMath> and <InlineMath>{`\\beta`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the cross-attention:
            <List withPadding>
              <List.Item>Compute gradients for attention weights and projection matrices</List.Item>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Y^{l'}}`}</InlineMath> and <InlineMath>{`\\frac{\\partial L}{\\partial Z^L}`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the layer normalization after masked self-attention:
            <List withPadding>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Y^{l-1}}`}</InlineMath> and <InlineMath>{`\\frac{\\partial L}{\\partial \\text{MaskedMultiHead}(Y^{l-1})}`}</InlineMath></List.Item>
              <List.Item>Update gradients for <InlineMath>{`\\gamma`}</InlineMath> and <InlineMath>{`\\beta`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the masked self-attention:
            <List withPadding>
              <List.Item>Compute gradients for attention weights and projection matrices</List.Item>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Y^{l-1}}`}</InlineMath></List.Item>
            </List>
          </List.Item>
        </List>
        
        <Title order={4} mt="sm">For Each Encoder Layer (from L to 1):</Title>
        <List>
          <List.Item>Backpropagate through the layer normalization after FFN:
            <List withPadding>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Z^{l'}}`}</InlineMath> and <InlineMath>{`\\frac{\\partial L}{\\partial \\text{FFN}(Z^{l'})}`}</InlineMath></List.Item>
              <List.Item>Update gradients for <InlineMath>{`\\gamma`}</InlineMath> and <InlineMath>{`\\beta`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the feed-forward network:
            <List withPadding>
              <List.Item>Compute gradients for <InlineMath>{`W^{FF}_1`}</InlineMath>, <InlineMath>{`W^{FF}_2`}</InlineMath>, <InlineMath>{`b_1`}</InlineMath>, and <InlineMath>{`b_2`}</InlineMath></List.Item>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Z^{l'}}`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the layer normalization after self-attention:
            <List withPadding>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Z^{l-1}}`}</InlineMath> and <InlineMath>{`\\frac{\\partial L}{\\partial \\text{MultiHead}(Z^{l-1})}`}</InlineMath></List.Item>
              <List.Item>Update gradients for <InlineMath>{`\\gamma`}</InlineMath> and <InlineMath>{`\\beta`}</InlineMath></List.Item>
            </List>
          </List.Item>
          <List.Item>Backpropagate through the self-attention:
            <List withPadding>
              <List.Item>Compute gradients for attention weights and projection matrices</List.Item>
              <List.Item>Compute <InlineMath>{`\\frac{\\partial L}{\\partial Z^{l-1}}`}</InlineMath></List.Item>
            </List>
          </List.Item>
        </List>
        
        <Title order={4} mt="sm">Input Embeddings:</Title>
        <List>
          <List.Item>Backpropagate to the input embeddings, computing <InlineMath>{`\\frac{\\partial L}{\\partial X}`}</InlineMath></List.Item>
          <List.Item>If the embeddings are learnable, update their gradients</List.Item>
        </List>
        
        <Title order={3} mt="lg">Step 3: Parameter Update</Title>
        <Text>
          Apply gradient descent (or a variant) to update all parameters:
        </Text>
        
        <BlockMath>{`
          W^{(t+1)} = W^{(t)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(t)}}
        `}</BlockMath>
        
        <Text>
          Where <InlineMath>{`\\eta`}</InlineMath> is the learning rate and W represents any parameter in the model.
        </Text>
      </Stack>

    </Stack>
  );
};

export default TransformerComponents;