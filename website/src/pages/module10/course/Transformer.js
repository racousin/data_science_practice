import React from "react";
import { Container, Title, Text, Space, Divider, Grid, Card, Table, Accordion, Box } from "@mantine/core";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";

const Transformer = () => {
  return (
    <Container fluid>
      <Title order={1} mb="md" id="transformer-architecture">
        Transformer Architecture
      </Title>
      <Text>
        The Transformer architecture, introduced in the seminal paper "Attention Is All You Need" (Vaswani et al., 2017), 
        revolutionized natural language processing by eliminating recurrence and convolutions while relying solely on 
        attention mechanisms. This architecture forms the foundation for modern NLP models like BERT, GPT, T5, and others.
      </Text>

      <Space h="md" />
      
      {/* Architecture Overview Section */}
      <Title order={2} mb="sm" id="architecture">
        Transformer Model Architecture
      </Title>
      <Card shadow="sm" p="lg" radius="md" withBorder mb="md">
        <Text>
          The Transformer follows an encoder-decoder architecture designed for sequence-to-sequence tasks. 
          The key innovations include:
        </Text>
        <Grid mt="md">
          <Grid.Col span={6}>
            <ul>
              <li>Self-attention mechanisms instead of recurrence</li>
              <li>Parallel processing of the entire sequence</li>
              <li>Multi-head attention for capturing different relationship types</li>
              <li>Positional encodings to preserve sequence order</li>
              <li>Residual connections and layer normalization</li>
            </ul>
          </Grid.Col>
          <Grid.Col span={6}>
            <Box className="visual-placeholder" sx={{ height: 300, backgroundColor: "#f5f5f5", display: "flex", justifyContent: "center", alignItems: "center" }}>
              <Text size="sm" color="dimmed">Transformer Architecture Diagram</Text>
              {/* Replace with VisualComponent for actual implementation */}
            </Box>
          </Grid.Col>
        </Grid>
      </Card>

      <Text>
        At a high level, the Transformer consists of stacked encoder and decoder layers:
      </Text>
      <ul>
        <li><strong>Encoder stack:</strong> Processes the input sequence in parallel</li>
        <li><strong>Decoder stack:</strong> Generates the output sequence autoregressively</li>
      </ul>

      <Space h="md" />

      {/* Self-Attention Section */}
      <Title order={2} mb="sm" id="self-attention">
        Self-Attention Mechanism
      </Title>
      <Text>
        Self-attention, also known as intra-attention, allows the model to weigh the importance of different tokens 
        within the same sequence when encoding a specific token. This mechanism enables the model to capture 
        long-range dependencies regardless of their distance in the input sequence.
      </Text>

      <Card shadow="sm" p="lg" radius="md" withBorder my="md">
        <Title order={3}>Self-Attention Computation</Title>
        <Text mt="md">
          For each token in a sequence, self-attention computes attention scores with respect to all other tokens:
        </Text>
        <Space h="sm" />
        <BlockMath math={`
          \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
        `} />
        <Space h="sm" />
        <Text>
          Where Q (query), K (key), and V (value) are different linear projections of the input embeddings, 
          and <InlineMath math="d_k" /> is the dimension of the key vectors.
        </Text>
      </Card>

      <Accordion variant="contained" mb="md">
        <Accordion.Item value="attention-key-components">
          <Accordion.Control>Key Components of Self-Attention</Accordion.Control>
          <Accordion.Panel>
            <Grid>
              <Grid.Col span={6}>
                <Title order={4}>Query, Key, Value Projections</Title>
                <Text>
                  Input embeddings are linearly projected to create:
                </Text>
                <ul>
                  <li><strong>Query (Q):</strong> What the current token is looking for</li>
                  <li><strong>Key (K):</strong> What each token in the sequence offers</li>
                  <li><strong>Value (V):</strong> The actual content of each token</li>
                </ul>
                <Text>
                  These projections are computed as:
                </Text>
                <BlockMath math={`
                  Q = XW^Q \\quad K = XW^K \\quad V = XW^V
                `} />
                <Text>
                  Where <InlineMath math="X" /> is the input embedding matrix, and <InlineMath math="W^Q, W^K, W^V" /> are learned parameter matrices.
                </Text>
              </Grid.Col>
              <Grid.Col span={6}>
                <Title order={4}>Scaling Factor</Title>
                <Text>
                  The dot products are scaled by <InlineMath math="\\sqrt{d_k}" /> to prevent the softmax function from 
                  having extremely small gradients when the dot products become large.
                </Text>
                <Space h="sm" />
                <Title order={4}>Softmax Normalization</Title>
                <Text>
                  The softmax function converts attention scores to probabilities, ensuring they sum to 1:
                </Text>
                <BlockMath math={`
                  \\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}
                `} />
                <Text>
                  This creates a probability distribution over all tokens in the sequence.
                </Text>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Card shadow="sm" p="lg" radius="md" withBorder>
        <Text>
          <strong>Self-Attention in PyTorch:</strong>
        </Text>
        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure the embedding size is divisible by the number of heads
        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by number of heads"

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        
        # Final output projection
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_length = query.shape[1]
        
        # Linear projections
        q = self.q_linear(query)  # (batch_size, seq_length, embed_size)
        k = self.k_linear(key)    # (batch_size, seq_length, embed_size)
        v = self.v_linear(value)  # (batch_size, seq_length, embed_size)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_length, self.heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.heads, self.head_dim)
        
        # Transpose to make shapes: (batch_size, heads, seq_length, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Calculate attention scores
        attention = torch.matmul(q, k.permute(0, 1, 3, 2))  # (batch_size, heads, seq_length, seq_length)
        
        # Scale attention scores
        attention = attention / (self.head_dim ** 0.5)
        
        # Apply mask for padding or causal attention (if provided)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=-1)
        
        # Multiply attention weights with values
        out = torch.matmul(attention, v)  # (batch_size, heads, seq_length, head_dim)
        
        # Reshape and concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embed_size)
        
        # Final linear projection
        out = self.out(out)
        
        return out, attention  # Return attention weights for visualization
`}
        />
      </Card>

      <Space h="xl" />

      {/* Multi-Head Attention Section */}
      <Title order={2} mb="sm" id="multi-head-attention">
        Multi-Head Attention
      </Title>
      <Text>
        Rather than performing a single attention function, the Transformer uses multiple attention heads in parallel. 
        This allows the model to jointly attend to information from different representation subspaces at different positions.
      </Text>

      <Card shadow="sm" p="lg" radius="md" withBorder my="md">
        <Grid>
          <Grid.Col span={6}>
            <Title order={3}>Multi-Head Attention Formulation</Title>
            <BlockMath math={`
              \\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)W^O
            `} />
            <BlockMath math={`
              \\text{where } \\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
            `} />
            <Text>
              Where <InlineMath math="W_i^Q, W_i^K, W_i^V" /> are parameter matrices for each head <InlineMath math="i" />, 
              and <InlineMath math="W^O" /> is the output projection matrix.
            </Text>
          </Grid.Col>
          <Grid.Col span={6}>
            <Box className="visual-placeholder" sx={{ height: 240, backgroundColor: "#f5f5f5", display: "flex", justifyContent: "center", alignItems: "center" }}>
              <Text size="sm" color="dimmed">Multi-Head Attention Diagram</Text>
              {/* Replace with VisualComponent for actual implementation */}
            </Box>
          </Grid.Col>
        </Grid>
      </Card>

      <Accordion variant="contained" mb="md">
        <Accordion.Item value="multihead-benefits">
          <Accordion.Control>Benefits of Multi-Head Attention</Accordion.Control>
          <Accordion.Panel>
            <Grid>
              <Grid.Col span={12}>
                <ul>
                  <li><strong>Capturing different relationships:</strong> Each attention head can focus on different patterns in the data</li>
                  <li><strong>Increased representation power:</strong> Different heads can attend to different semantic aspects</li>
                  <li><strong>Improved stability:</strong> Multiple heads provide redundancy and distribute the learning</li>
                </ul>
                <Text>
                  For example, in language tasks, different heads might specialize in:
                </Text>
                <ul>
                  <li>Syntactic relationships (subject-verb agreement)</li>
                  <li>Coreference resolution (pronouns and their antecedents)</li>
                  <li>Entity relationships</li>
                  <li>Long-distance dependencies</li>
                </ul>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Card shadow="sm" p="lg" radius="md" withBorder>
        <Text>
          <strong>Hyperparameters for Multi-Head Attention:</strong>
        </Text>
        <Table>
          <thead>
            <tr>
              <th>Hyperparameter</th>
              <th>Description</th>
              <th>Typical Values</th>
              <th>Impact</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Number of heads (h)</td>
              <td>The number of parallel attention mechanisms</td>
              <td>8 (base), 16 (large)</td>
              <td>Increases representational capacity but also computation</td>
            </tr>
            <tr>
              <td>Head dimension (d<sub>k</sub>)</td>
              <td>Dimension of each attention head (d<sub>model</sub>/h)</td>
              <td>64, 128</td>
              <td>Determines the granularity of attention patterns</td>
            </tr>
            <tr>
              <td>Attention dropout</td>
              <td>Dropout applied to attention weights</td>
              <td>0.1, 0.2</td>
              <td>Prevents over-reliance on specific attention patterns</td>
            </tr>
          </tbody>
        </Table>
      </Card>

      <Space h="xl" />

      {/* Positional Encodings Section */}
      <Title order={2} mb="sm" id="positional-encodings">
        Positional Encodings
      </Title>
      <Text>
        Unlike recurrent models, the Transformer processes all tokens in parallel, losing information about their 
        positions in the sequence. Positional encodings are added to the input embeddings to provide sequential context.
      </Text>

      <Card shadow="sm" p="lg" radius="md" withBorder my="md">
        <Grid>
          <Grid.Col span={6}>
            <Title order={3}>Sinusoidal Positional Encoding</Title>
            <Text>
              The original Transformer uses fixed sinusoidal functions:
            </Text>
            <BlockMath math={`
              PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)
            `} />
            <BlockMath math={`
              PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)
            `} />
            <Text>
              Where <InlineMath math="pos" /> is the position in the sequence, <InlineMath math="i" /> is the dimension, 
              and <InlineMath math="d_{model}" /> is the embedding dimension.
            </Text>
          </Grid.Col>
          <Grid.Col span={6}>
            <Box className="visual-placeholder" sx={{ height: 240, backgroundColor: "#f5f5f5", display: "flex", justifyContent: "center", alignItems: "center" }}>
              <Text size="sm" color="dimmed">Positional Encoding Visualization</Text>
              {/* Replace with VisualComponent for actual implementation */}
            </Box>
          </Grid.Col>
        </Grid>
      </Card>

      <Accordion variant="contained" mb="md">
        <Accordion.Item value="positional-encoding-types">
          <Accordion.Control>Types of Positional Encodings</Accordion.Control>
          <Accordion.Panel>
            <Grid>
              <Grid.Col span={12}>
                <Title order={4}>1. Fixed Sinusoidal (Transformer Original)</Title>
                <ul>
                  <li><strong>Advantages:</strong> No additional parameters, works for sequences of any length</li>
                  <li><strong>Disadvantages:</strong> Not learned from data, fixed pattern</li>
                </ul>
                
                <Title order={4}>2. Learned Positional Embeddings</Title>
                <ul>
                  <li><strong>Advantages:</strong> Adaptable to data, potentially more expressive</li>
                  <li><strong>Disadvantages:</strong> Limited to maximum sequence length seen during training</li>
                </ul>
                
                <Title order={4}>3. Relative Positional Encodings</Title>
                <ul>
                  <li><strong>Advantages:</strong> Focus on relative distances between tokens rather than absolute positions</li>
                  <li><strong>Disadvantages:</strong> More complex implementation</li>
                </ul>
                
                <Title order={4}>4. Rotary Position Embeddings (RoPE)</Title>
                <ul>
                  <li><strong>Advantages:</strong> Encodes relative positions directly in attention calculations</li>
                  <li><strong>Disadvantages:</strong> Requires modification to attention mechanism</li>
                </ul>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Card shadow="sm" p="lg" radius="md" withBorder>
        <Text>
          <strong>PyTorch Implementation of Sinusoidal Positional Encoding:</strong>
        </Text>
        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a vector of shape (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create a vector of shape (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x has shape (batch_size, seq_length, d_model)
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
`}
        />
      </Card>

      <Space h="xl" />

      {/* Encoder-Decoder Section */}
      <Title order={2} mb="sm" id="encoder-decoder">
        Encoder and Decoder Components
      </Title>
      <Text>
        The Transformer consists of stacked encoder and decoder layers, each with specific sub-layers for different functions.
      </Text>

      <Grid grow gutter="lg" mb="md">
        <Grid.Col span={6}>
          <Card shadow="sm" p="lg" radius="md" withBorder h="100%">
            <Title order={3} mb="md">Encoder Layer</Title>
            <Text>
              Each encoder layer consists of:
            </Text>
            <ol>
              <li><strong>Multi-Head Self-Attention:</strong> Processes all positions in the input sequence</li>
              <li><strong>Position-wise Feed-Forward Network (FFN):</strong> Applies the same feed-forward network to each position separately</li>
            </ol>
            <Text mt="md">
              Both sub-layers employ:
            </Text>
            <ul>
              <li><strong>Residual connections:</strong> Add the input to the output of each sub-layer</li>
              <li><strong>Layer normalization:</strong> Normalizes the outputs for stable training</li>
            </ul>
            <BlockMath math={`
              \\text{LayerNorm}(x + \\text{SubLayer}(x))
            `} />
          </Card>
        </Grid.Col>
        
        <Grid.Col span={6}>
          <Card shadow="sm" p="lg" radius="md" withBorder h="100%">
            <Title order={3} mb="md">Decoder Layer</Title>
            <Text>
              Each decoder layer consists of:
            </Text>
            <ol>
              <li><strong>Masked Multi-Head Self-Attention:</strong> Ensures tokens can only attend to previous positions</li>
              <li><strong>Cross-Attention:</strong> Attends to encoder outputs</li>
              <li><strong>Position-wise Feed-Forward Network:</strong> Same as in the encoder</li>
            </ol>
            <Text mt="md">
              The masked attention is crucial for autoregressive generation:
            </Text>
            <ul>
              <li>During training, prevents tokens from attending to future positions</li>
              <li>During inference, allows generation one token at a time</li>
            </ul>
            <BlockMath math={`
              \\text{Mask}_{ij} = \\begin{cases} 
                0 & \\text{if } i < j \\\\
                1 & \\text{otherwise}
              \\end{cases}
            `} />
          </Card>
        </Grid.Col>
      </Grid>

      <Card shadow="sm" p="lg" radius="md" withBorder mb="md">
        <Title order={3}>Position-wise Feed-Forward Network</Title>
        <Text>
          The FFN consists of two linear transformations with a ReLU activation in between:
        </Text>
        <BlockMath math={`
          \\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2
        `} />
        <Text>
          This is applied to each position separately and identically. The inner dimension <InlineMath math="d_{ff}" /> 
          is typically larger than the model dimension <InlineMath math="d_{model}" />.
        </Text>
        <Grid mt="md">
          <Grid.Col span={12}>
            <Table>
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Hyperparameter</th>
                  <th>Typical Values</th>
                  <th>Function</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Encoder/Decoder</td>
                  <td>Number of layers (N)</td>
                  <td>6 (base), 12 (large)</td>
                  <td>Depth of the network, affecting model capacity</td>
                </tr>
                <tr>
                  <td>Embedding</td>
                  <td>Model dimension (d<sub>model</sub>)</td>
                  <td>512 (base), 1024 (large)</td>
                  <td>Width of embeddings and hidden representations</td>
                </tr>
                <tr>
                  <td>Feed-Forward</td>
                  <td>Inner dimension (d<sub>ff</sub>)</td>
                  <td>2048 (base), 4096 (large)</td>
                  <td>Intermediate dimension in feed-forward layers</td>
                </tr>
                <tr>
                  <td>Regularization</td>
                  <td>Dropout rate</td>
                  <td>0.1, 0.2</td>
                  <td>Controls regularization strength</td>
                </tr>
              </tbody>
            </Table>
          </Grid.Col>
        </Grid>
      </Card>

      <Space h="xl" />

      {/* Mathematical Formulation of Attention Section */}
      <Title order={2} mb="sm" id="attention-math">
        Mathematical Formulation of Attention
      </Title>
      <Text>
        The attention mechanism is the core innovation of the Transformer. Let's examine its formal mathematical representation in detail.
      </Text>

      <Card shadow="sm" p="lg" radius="md" withBorder mb="md">
        <Title order={3}>Scaled Dot-Product Attention</Title>
        <Grid>
          <Grid.Col span={7}>
            <BlockMath math={`
              \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
            `} />
            <Text mt="md">
              The attention computation can be broken down into steps:
            </Text>
            <ol>
              <li>Compute dot products between queries and keys: <InlineMath math="QK^T" /></li>
              <li>Scale by <InlineMath math="\\sqrt{d_k}" />: <InlineMath math="\\frac{QK^T}{\\sqrt{d_k}}" /></li>
              <li>Apply softmax to get attention weights: <InlineMath math="\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)" /></li>
              <li>Multiply attention weights with values: <InlineMath math="\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V" /></li>
            </ol>
          </Grid.Col>
          <Grid.Col span={5}>
            <Box className="visual-placeholder" sx={{ height: 240, backgroundColor: "#f5f5f5", display: "flex", justifyContent: "center", alignItems: "center" }}>
              <Text size="sm" color="dimmed">Attention Calculation Diagram</Text>
              {/* Replace with VisualComponent for actual implementation */}
            </Box>
          </Grid.Col>
        </Grid>
      </Card>

      <Card shadow="sm" p="lg" radius="md" withBorder mb="md">
        <Title order={3}>Matrix Dimensions in Attention Calculation</Title>
        <Text>
          For a sequence of length <InlineMath math="n" />, with <InlineMath math="d_k" /> being the dimension of keys and queries, 
          and <InlineMath math="d_v" /> being the dimension of values:
        </Text>
        <Table>
          <thead>
            <tr>
              <th>Matrix</th>
              <th>Dimensions</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><InlineMath math="Q" /></td>
              <td><InlineMath math="n \times d_k" /></td>
              <td>Query matrix</td>
            </tr>
            <tr>
              <td><InlineMath math="K" /></td>
              <td><InlineMath math="n \times d_k" /></td>
              <td>Key matrix</td>
            </tr>
            <tr>
              <td><InlineMath math="V" /></td>
              <td><InlineMath math="n \times d_v" /></td>
              <td>Value matrix</td>
            </tr>
            <tr>
              <td><InlineMath math="QK^T" /></td>
              <td><InlineMath math="n \times n" /></td>
              <td>Attention scores</td>
            </tr>
            <tr>
              <td><InlineMath math="\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)" /></td>
              <td><InlineMath math="n \times n" /></td>
              <td>Attention weights</td>
            </tr>
            <tr>
              <td><InlineMath math="\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V" /></td>
              <td><InlineMath math="n \times d_v" /></td>
              <td>Attention output</td>
            </tr>
          </tbody>
        </Table>
      </Card>

      <Accordion variant="contained" mb="md">
        <Accordion.Item value="attention-variants">
          <Accordion.Control>Attention Variants and Complexities</Accordion.Control>
          <Accordion.Panel>
            <Grid>
              <Grid.Col span={12}>
                <Title order={4}>1. Masked Attention</Title>
                <BlockMath math={`
                  \\text{MaskedAttention}(Q, K, V, M) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V
                `} />
                <Text>
                  Where <InlineMath math="M" /> is a mask with <InlineMath math="-\infty" /> (or a very large negative number) 
                  at positions we want to prevent from attending. Used in decoders for autoregressive generation.
                </Text>
                
                <Title order={4} mt="md">2. Computational Complexity</Title>
                <Text>
                  The self-attention operation has quadratic complexity with respect to sequence length:
                </Text>
                <BlockMath math={`
                  \\text{Time Complexity} = O(n^2 \\cdot d)
                `} />
                <BlockMath math={`
                  \\text{Memory Complexity} = O(n^2)
                `} />
                <Text>
                  This becomes problematic for very long sequences. Variants like Sparse Transformers, Longformer, 
                  and Reformer address this by approximating full attention.
                </Text>
                
                <Title order={4} mt="md">3. Multi-Query and Grouped-Query Attention</Title>
                <Text>
                  Efficiency variants that reduce the number of key-value projections:
                </Text>
                <ul>
                  <li><strong>Multi-Query Attention:</strong> Multiple query heads but only one key and value head</li>
                  <li><strong>Grouped-Query Attention:</strong> Multiple query heads sharing a smaller number of key-value heads</li>
                </ul>
              </Grid.Col>
            </Grid>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

      <Space h="xl" />

      {/* Information Flow Section */}
      <Title order={2} mb="sm" id="information-flow">
        Information Flow
      </Title>
      <Text>
        Understanding how information propagates through the Transformer is crucial for grasping its efficacy in NLP tasks.
      </Text>

      <Card shadow="sm" p="lg" radius="md" withBorder mb="md">
        <Title order={3}>End-to-End Processing Flow</Title>
        <Grid>
          <Grid.Col span={12}>
            <ol>
              <li>
                <strong>Input Embedding and Positional Encoding:</strong>
                <ul>
                  <li>Input tokens are converted to embeddings of dimension <InlineMath math="d_{model}" /></li>
                  <li>Positional encodings are added to preserve sequence information</li>
                </ul>
              </li>
              <li>
                <strong>Encoder Processing:</strong>
                <ul>
                  <li>The encoder processes the input sequence through N identical layers</li>
                  <li>Each layer captures different levels of linguistic patterns and relationships</li>
                  <li>Self-attention allows each token to gather context from all positions</li>
                  <li>The feed-forward network adds non-linear transformations to each position</li>
                </ul>
              </li>
              <li>
                <strong>Decoder Processing:</strong>
                <ul>
                  <li>The decoder generates output tokens one at a time</li>
                  <li>Self-attention in the decoder applies causal masking to prevent looking ahead</li>
                  <li>Cross-attention connects the decoder to the encoder's contextual representations</li>
                  <li>The feed-forward network adds further non-linearity</li>
                </ul>
              </li>
              <li>
                <strong>Output Generation:</strong>
                <ul>
                  <li>The final decoder layer output is projected to the vocabulary size</li>
                  <li>Softmax converts logits to probability distribution over vocabulary</li>
                  <li>The highest probability token is selected (or sampling is applied)</li>
                </ul>
              </li>
            </ol>
          </Grid.Col>
        </Grid>
      </Card>

      <Card shadow="sm" p="lg" radius="md" withBorder mb="md">
        <Title order={3}>Information Paths and Gradient Flow</Title>
        <Text>
          The Transformer's architecture facilitates efficient information and gradient flow:
        </Text>
        <Grid>
          <Grid.Col span={6}>
            <Title order={4}>Forward Pass Information Flow</Title>
            <ul>
              <li><strong>Direct paths:</strong> Residual connections create direct paths between layers</li>
              <li><strong>Global context:</strong> Self-attention enables each token to access all other tokens</li>
              <li><strong>Parallel processing:</strong> All tokens are processed simultaneously in each layer</li>
            </ul>
          </Grid.Col>
          <Grid.Col span={6}>
            <Title order={4}>Backward Pass Gradient Flow</Title>
            <ul>
              <li><strong>Residual connections:</strong> Enable gradients to flow more easily to lower layers</li>
              <li><strong>Layer normalization:</strong> Stabilizes gradient magnitudes across layers</li>
              <li><strong>No vanishing/exploding gradients:</strong> Unlike RNNs, gradients don't decay with sequence length</li>
            </ul>
          </Grid.Col>
        </Grid>
      </Card>

      <Card shadow="sm" p="lg" radius="md" withBorder>
        <Title order={3}>Building a Transformer with PyTorch</Title>
        <Text mb="md">
          Here's a simplified implementation of a Transformer encoder layer:
        </Text>
        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
            mask: Optional mask for self-attention
        """
        # Self-attention block with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x
        
# Usage example:
batch_size = 32
seq_len = 50
d_model = 512
n_heads = 8
d_ff = 2048

# Create a random input tensor
x = torch.randn(seq_len, batch_size, d_model)

# Create an encoder layer
encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

# Process the input
output = encoder_layer(x)
print(f"Output shape: {output.shape}")  # Should be [seq_len, batch_size, d_model]
`}
        />
      </Card>

      <Space h="xl" />

      {/* Complete Transformer Implementation */}
      <Title order={2} mb="sm" id="complete-implementation">
        Complete Transformer Implementation
      </Title>
      <Text>
        The Hugging Face Transformers library provides a streamlined way to use Transformer models in PyTorch:
      </Text>

      <CodeBlock
        language="python"
        code={`
import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Prepare input
text = "The Transformer architecture revolutionized natural language processing."
inputs = tokenizer(text, return_tensors="pt")

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)

# Access the last hidden states
last_hidden_states = outputs.last_hidden_state
print(f"Hidden state shape: {last_hidden_states.shape}")  # [batch_size, seq_len, hidden_size]

# Use the hidden states for downstream tasks
# For example, take the [CLS] token representation for classification
cls_embedding = last_hidden_states[:, 0, :]
print(f"CLS embedding shape: {cls_embedding.shape}")  # [batch_size, hidden_size]
`}
      />

      <Space h="xl" />

      {/* Summary Section */}
      <Title order={2} mb="sm" id="summary">
        Key Takeaways
      </Title>
      <Card shadow="sm" p="lg" radius="md" withBorder>
        <Grid>
          <Grid.Col span={12}>
            <ol>
              <li><strong>Architecture:</strong> The Transformer uses an encoder-decoder architecture without recurrence or convolution</li>
              <li><strong>Attention Mechanism:</strong> Self-attention allows tokens to gather context from all positions in the sequence</li>
              <li><strong>Multi-Head Attention:</strong> Multiple attention heads enable the model to capture different types of relationships</li>
              <li><strong>Positional Encodings:</strong> Necessary to preserve sequence order in a parallel processing architecture</li>
              <li><strong>Encoder-Decoder:</strong> Encoder processes input, decoder generates output autogressively</li>
              <li><strong>Mathematical Formulation:</strong> Attention is calculated as softmax-normalized dot products between queries and keys, weighted by values</li>
              <li><strong>Information Flow:</strong> Residual connections and layer normalization facilitate efficient information propagation</li>
            </ol>
          </Grid.Col>
        </Grid>
      </Card>

      <Space h="xl" />


    </Container>
  );
};

export default Transformer;