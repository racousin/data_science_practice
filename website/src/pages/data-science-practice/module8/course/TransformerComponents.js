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
        <Title order={2} id="transformer-notation">Transformer Notation</Title>
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
        <Title order={2} id="transformer-components-layers">Transformers Components Layers</Title>
        
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
          Backpropagation in transformers follows the standard gradient descent procedure with a few key differences due to the attention mechanisms.
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
          
          <Accordion.Item value="attention-backprop" id="attention-backprop">
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
      <Title order={2} id="minimal-transformer-implementation">Minimal Transformer Implementation</Title>
      <Text>
        Below is a simple example of Transformer Implementation and training:
      </Text>
      
      <CodeBlock
        language="python"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding described in the Transformer paper.
    """
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill the matrix with sine and cosine values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register the positional encoding as a buffer (persistent state)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]
        Returns:
            x + positional_encoding
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in "Attention is All You Need".
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: Query, Key, Value tensors [batch_size, seq_length, d_model]
            mask: Optional mask tensor [batch_size, 1, 1, seq_length]
        Returns:
            Output tensor after multi-head attention [batch_size, seq_length, d_model]
        """
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network from the Transformer paper.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
        Returns:
            Transformed tensor [batch_size, seq_length, d_model]
        """
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """
    Single encoder layer for the Transformer.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional attention mask
        Returns:
            Output tensor after one encoder layer
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer for the Transformer.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            enc_output: Output from the encoder [batch_size, src_seq_length, d_model]
            src_mask: Mask for encoder attention
            tgt_mask: Mask for decoder self-attention (ensures causality)
        Returns:
            Output tensor after one decoder layer
        """
        # Self-attention with residual connection and layer norm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with encoder outputs
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Full Transformer model as described in "Attention is All You Need".
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer to produce logits
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Hyperparameters
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz):
        """
        Generate a causal mask for the decoder self-attention.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source sequence [batch_size, src_seq_length]
            tgt: Target sequence [batch_size, tgt_seq_length]
            src_mask: Mask for encoder attention
            tgt_mask: Mask for decoder self-attention
        Returns:
            Output logits [batch_size, tgt_seq_length, tgt_vocab_size]
        """
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        # Encode source sequence
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        enc_output = src_embedded
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        # Decode target sequence
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # Project to vocabulary space
        output = self.final_linear(dec_output)
        return output


        # Example usage with a toy translation task
        def train_example():
            # Define a very small toy dataset
            src_vocab_size = 100  # Source vocabulary size
            tgt_vocab_size = 120  # Target vocabulary size
            d_model = 64         # Model dimension (reduced for example)
            num_heads = 4        # Number of attention heads
            num_layers = 2       # Number of encoder/decoder layers (reduced for example)
            
            # Create a minimal transformer model
            model = Transformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                d_ff=d_model * 4,
                dropout=0.1
            )
            
            # Create a small batch of synthetic data
            batch_size = 5
            src_seq_length = 10
            tgt_seq_length = 12
            
            src = torch.randint(1, src_vocab_size, (batch_size, src_seq_length))
            tgt_input = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_length-1))
            tgt_output = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_length-1))
            
            # Prepare for training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            
            # Forward pass
            model.train()
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            
            # Calculate loss
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_output.contiguous().view(-1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Training loss: {loss.item()}")
            
            return model
        
        model = train_example()    `}
      />
    </Stack>
  );
};

export default TransformerComponents;