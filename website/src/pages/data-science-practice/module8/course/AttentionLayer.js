import React from "react";
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";

const AttentionLayer = () => {
  return (
    <>
      <div data-slide>
        <Title order={1}>Attention Mechanism</Title>

        <Text mt="md">
          Attention mechanisms allow models to focus on specific parts of the input when producing each output,
          rather than compressing all information into a fixed-size representation.
        </Text>

        <Text mt="md">
          This fundamental innovation solved the bottleneck problem in sequence-to-sequence models and
          became the foundation for Transformer architectures.
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014) - https://arxiv.org/abs/1409.0473
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Motivation: The Bottleneck Problem</Title>

        <Text mt="md">
          In traditional seq2seq models with RNNs:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>
            Encoder compresses entire input sequence into a single fixed-size context vector
          </List.Item>
          <List.Item>
            Decoder must generate entire output from this single vector
          </List.Item>
          <List.Item>
            Information loss increases with sequence length
          </List.Item>
          <List.Item>
            Performance degrades significantly for long sequences
          </List.Item>
        </List>

        <Text mt="lg">
          <strong>Solution:</strong> Allow decoder to "attend" to different parts of the encoder's hidden states,
          creating a dynamic context vector for each output token.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/seq2seq-bottleneck-problem.png"
            alt="Sequence-to-sequence bottleneck problem visualization"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Sequence-to-sequence bottleneck problem visualization
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Attention: Core Concept</Title>

        <Text mt="md">
          Attention computes a weighted combination of values based on the compatibility between
          a query and keys.
        </Text>

        <Title order={3} mt="lg">Intuition</Title>
        <Text mt="sm">
          Given a query "What is the capital of France?", attention assigns high weights to
          relevant parts of the input (e.g., "Paris") and low weights to irrelevant parts.
        </Text>

        <Text mt="lg">
          <strong>Core operation:</strong>
        </Text>
        <BlockMath>{`
          \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
        `}</BlockMath>

        <Text mt="lg">
          Where Q (query), K (keys), and V (values) are matrices representing different aspects
          of the input.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/attention-mechanism-diagram.png"
            alt="Attention mechanism diagram showing Q, K, V interaction"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Attention mechanism diagram showing Q, K, V interaction
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Attention Components</Title>

        <Title order={3} mt="md">Query (Q)</Title>
        <Text size="sm">
          Represents what we're looking for. Shape: <InlineMath>{`\\mathbb{R}^{n_q \\times d_k}`}</InlineMath>
        </Text>

        <Title order={3} mt="lg">Keys (K)</Title>
        <Text size="sm">
          Represent what's available to attend to. Shape: <InlineMath>{`\\mathbb{R}^{n_k \\times d_k}`}</InlineMath>
        </Text>

        <Title order={3} mt="lg">Values (V)</Title>
        <Text size="sm">
          Actual content to aggregate. Shape: <InlineMath>{`\\mathbb{R}^{n_v \\times d_v}`}</InlineMath>
        </Text>

        <Text mt="lg">
          Note: <InlineMath>{`n_k = n_v`}</InlineMath> (same number of key-value pairs)
        </Text>

        <Text mt="lg">
          <strong>Output shape:</strong> <InlineMath>{`\\mathbb{R}^{n_q \\times d_v}`}</InlineMath>
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Scaled Dot-Product Attention</Title>

        <Text mt="md">
          The standard attention mechanism computes attention in three steps:
        </Text>

        <Title order={3} mt="lg">Step 1: Compute Attention Scores</Title>
        <BlockMath>{`
          S = QK^T \\in \\mathbb{R}^{n_q \\times n_k}
        `}</BlockMath>
        <Text size="sm">Measures compatibility between each query and key</Text>

        <Title order={3} mt="lg">Step 2: Scale and Normalize</Title>
        <BlockMath>{`
          A = \\text{softmax}\\left(\\frac{S}{\\sqrt{d_k}}\\right) \\in \\mathbb{R}^{n_q \\times n_k}
        `}</BlockMath>
        <Text size="sm">Scaling factor <InlineMath>{`\\sqrt{d_k}`}</InlineMath> prevents saturation of softmax for large <InlineMath>{`d_k`}</InlineMath></Text>

        <Title order={3} mt="lg">Step 3: Weighted Sum of Values</Title>
        <BlockMath>{`
          \\text{Output} = AV \\in \\mathbb{R}^{n_q \\times d_v}
        `}</BlockMath>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/scaled-dot-product-attention.png"
            alt="Scaled dot-product attention computation steps"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Scaled dot-product attention computation steps
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Why Scale by √d_k?</Title>

        <Text mt="md">
          Without scaling, dot products grow large in magnitude as dimensionality increases.
        </Text>

        <Text mt="lg">
          For <InlineMath>{`q, k \\sim \\mathcal{N}(0, 1)`}</InlineMath>:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Expected value: <InlineMath>{`E[q \\cdot k] = 0`}</InlineMath></List.Item>
          <List.Item>Variance: <InlineMath>{`\\text{Var}(q \\cdot k) = d_k`}</InlineMath></List.Item>
        </List>

        <Text mt="lg">
          Large dot products push softmax into regions with small gradients, hindering learning.
        </Text>

        <Text mt="lg">
          <strong>Solution:</strong> Divide by <InlineMath>{`\\sqrt{d_k}`}</InlineMath> to keep variance ≈ 1
        </Text>

        <BlockMath>{`
          \\text{Var}\\left(\\frac{q \\cdot k}{\\sqrt{d_k}}\\right) = 1
        `}</BlockMath>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/scaling-factor-effect.png"
            alt="Effect of scaling factor on softmax gradient"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Effect of scaling factor on softmax gradient
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Attention Visualization</Title>

        <Flex direction="column" align="center" mt="md" mb="md">
          <Image
            src="/assets/data-science-practice/module8/attention-mechanism.png"
            alt="Attention mechanism computing weighted sum of values"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Attention mechanism: Query attends to Keys to produce attention weights, then aggregates Values
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Attention Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries, shape (batch_size, n_q, d_k)
        K: Keys, shape (batch_size, n_k, d_k)
        V: Values, shape (batch_size, n_v, d_v), where n_v = n_k
        mask: Optional mask, shape (batch_size, n_q, n_k)

    Returns:
        output: Attention output, shape (batch_size, n_q, d_v)
        attention_weights: Attention weights, shape (batch_size, n_q, n_k)
    """`}
        />

        <CodeBlock
          language="python"
          code={`    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    # Q: (batch, n_q, d_k), K: (batch, n_k, d_k)
    # scores: (batch, n_q, n_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Step 2: Apply mask if provided (set masked positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Attention Example</Title>

        <CodeBlock
          language="python"
          code={`# Example: Sequence length 10, embedding dimension 64
batch_size = 32
n_q, n_k = 10, 10  # Query and key sequence lengths
d_k, d_v = 64, 64  # Key and value dimensions

# Generate random Q, K, V
Q = torch.randn(batch_size, n_q, d_k)
K = torch.randn(batch_size, n_k, d_k)
V = torch.randn(batch_size, n_k, d_v)

# Compute attention
output, attention_weights = scaled_dot_product_attention(Q, K, V)`}
        />

        <CodeBlock
          language="python"
          code={`print(f"Output shape: {output.shape}")  # (32, 10, 64)
print(f"Attention weights shape: {attention_weights.shape}")  # (32, 10, 10)

# Verify attention weights sum to 1
print(f"Attention weights sum: {attention_weights.sum(dim=-1)[0, 0]:.4f}")  # 1.0000

# Visualize attention for first sample, first query
print("Attention distribution for first query:")
print(attention_weights[0, 0, :])
# Output: tensor([0.0823, 0.1245, 0.0634, ..., 0.1156])`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Attention Masking</Title>

        <Text mt="md">
          Masks prevent attention to certain positions by setting their scores to <InlineMath>{`-\\infty`}</InlineMath>
          before softmax.
        </Text>

        <Title order={3} mt="lg">Padding Mask</Title>
        <Text size="sm">
          Prevents attention to padding tokens in variable-length sequences.
        </Text>

        <Title order={3} mt="lg">Causal Mask (Look-ahead Mask)</Title>
        <Text size="sm">
          Prevents attention to future positions in autoregressive models.
        </Text>

        <BlockMath>{`
          M_{causal}[i, j] = \\begin{cases}
          1 & \\text{if } i \\geq j \\\\
          0 & \\text{otherwise}
          \\end{cases}
        `}</BlockMath>

        <Text mt="lg">
          Used in language modeling where position i cannot attend to positions j > i.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/attention-masking-types.png"
            alt="Different types of attention masks: padding and causal"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Different types of attention masks: padding and causal
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Causal Mask Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch

def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask, shape (seq_len, seq_len)
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1s below diagonal, 0s above`}
        />

        <CodeBlock
          language="python"
          code={`# Example: Create causal mask for sequence length 5
mask = create_causal_mask(5)
print(mask)
# Output:
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])

# Apply causal mask in attention
output, attn = scaled_dot_product_attention(Q, K, V, mask=mask.unsqueeze(0))`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention</Title>

        <Text mt="md">
          Multi-head attention runs multiple attention operations in parallel, allowing the model
          to attend to different representation subspaces.
        </Text>

        <BlockMath>{`
          \\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O
        `}</BlockMath>

        <BlockMath>{`
          \\text{where } \\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
        `}</BlockMath>

        <Text mt="lg">
          <strong>Projections:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`W_i^Q \\in \\mathbb{R}^{d_{model} \\times d_k}`}</InlineMath>: Query projection for head i</List.Item>
          <List.Item><InlineMath>{`W_i^K \\in \\mathbb{R}^{d_{model} \\times d_k}`}</InlineMath>: Key projection for head i</List.Item>
          <List.Item><InlineMath>{`W_i^V \\in \\mathbb{R}^{d_{model} \\times d_v}`}</InlineMath>: Value projection for head i</List.Item>
          <List.Item><InlineMath>{`W^O \\in \\mathbb{R}^{hd_v \\times d_{model}}`}</InlineMath>: Output projection</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/multi-head-attention-architecture.png"
            alt="Multi-head attention architecture with parallel attention heads"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Multi-head attention architecture with parallel attention heads
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention: Dimensions</Title>

        <Text mt="md">
          Typical configuration: <InlineMath>{`d_k = d_v = d_{model} / h`}</InlineMath>
        </Text>

        <Text mt="lg">
          <strong>Example:</strong> <InlineMath>{`d_{model} = 512, h = 8`}</InlineMath>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`d_k = d_v = 512 / 8 = 64`}</InlineMath></List.Item>
          <List.Item>Each head operates on 64-dimensional subspace</List.Item>
          <List.Item>8 heads capture different aspects of relationships</List.Item>
          <List.Item>Total computational cost ≈ single-head with full dimensionality</List.Item>
        </List>

        <Text mt="lg">
          <strong>Input shape:</strong> <InlineMath>{`(batch, seq\_len, d_{model})`}</InlineMath>
        </Text>
        <Text mt="sm">
          <strong>Output shape:</strong> <InlineMath>{`(batch, seq\_len, d_{model})`}</InlineMath>
        </Text>

        <Text mt="lg">
          <strong>Parameters:</strong> <InlineMath>{`4 \\times d_{model}^2`}</InlineMath> (3 projections + output)
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Vaswani et al., "Attention Is All You Need" (2017) - https://arxiv.org/abs/1706.03762
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)`}
        />

        <CodeBlock
          language="python"
          code={`    def split_heads(self, x):
        """Split last dimension into (num_heads, d_k)."""
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch, num_heads, seq_len_v, d_v)`}
        />

        <CodeBlock
          language="python"
          code={`        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(output)
        return output, attention_weights`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention Example</Title>

        <CodeBlock
          language="python"
          code={`# Initialize multi-head attention
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

# Input
batch_size, seq_len = 32, 10
x = torch.randn(batch_size, seq_len, d_model)

# Self-attention (Q = K = V)
output, attention_weights = mha(x, x, x)

print(f"Output shape: {output.shape}")  # (32, 10, 512)
print(f"Attention weights shape: {attention_weights.shape}")  # (32, 8, 10, 10)`}
        />

        <CodeBlock
          language="python"
          code={`# Calculate parameters
total_params = sum(p.numel() for p in mha.parameters())
print(f"Total parameters: {total_params:,}")
# Output: Total parameters: 1,048,576
# Calculation: 4 * (512 * 512) = 4 * 262,144 = 1,048,576

# Each head sees:
print(f"Dimension per head: {mha.d_k}")  # 64`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Self-Attention vs Cross-Attention</Title>

        <Title order={3} mt="md">Self-Attention</Title>
        <Text size="sm">
          Q, K, V all come from the same source: <InlineMath>{`Q = K = V = X`}</InlineMath>
        </Text>
        <Text size="sm">
          Each position attends to all positions in the same sequence.
        </Text>
        <Text size="sm">
          Used in: Transformer encoder, decoder self-attention
        </Text>

        <Title order={3} mt="lg">Cross-Attention</Title>
        <Text size="sm">
          Q from one source, K and V from another: <InlineMath>{`Q = X, K = V = Y`}</InlineMath>
        </Text>
        <Text size="sm">
          Each position in X attends to all positions in Y.
        </Text>
        <Text size="sm">
          Used in: Transformer decoder attending to encoder output (encoder-decoder attention)
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/self-cross-attention-comparison.png"
            alt="Self-attention vs cross-attention mechanisms"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Self-attention vs cross-attention mechanisms
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Attention Computational Complexity</Title>

        <Text mt="md">
          For sequence length <InlineMath>{`n`}</InlineMath> and model dimension <InlineMath>{`d`}</InlineMath>:
        </Text>

        <Table striped mt="lg">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Operation</Table.Th>
              <Table.Th>Complexity</Table.Th>
              <Table.Th>Description</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Q, K, V projections</Table.Td>
              <Table.Td><InlineMath>{`O(nd^2)`}</InlineMath></Table.Td>
              <Table.Td>3 linear transformations</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Attention scores</Table.Td>
              <Table.Td><InlineMath>{`O(n^2d)`}</InlineMath></Table.Td>
              <Table.Td>Matrix multiplication QK^T</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Weighted sum</Table.Td>
              <Table.Td><InlineMath>{`O(n^2d)`}</InlineMath></Table.Td>
              <Table.Td>Matrix multiplication (attn)V</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td><strong>Total</strong></Table.Td>
              <Table.Td><InlineMath>{`O(n^2d + nd^2)`}</InlineMath></Table.Td>
              <Table.Td>Quadratic in sequence length</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Text mt="lg">
          <strong>Memory:</strong> <InlineMath>{`O(n^2)`}</InlineMath> to store attention matrix
        </Text>

        <Text mt="md">
          This quadratic complexity motivates efficient attention variants for long sequences
          (e.g., sparse attention, linear attention).
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/attention-complexity-comparison.png"
            alt="Comparison of computational complexity between RNN and attention mechanisms"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Comparison of computational complexity between RNN and attention mechanisms
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Attention Parallelization</Title>

        <Text mt="md">
          Unlike RNNs, attention can be fully parallelized:
        </Text>

        <List spacing="sm" mt="lg">
          <List.Item>
            <strong>No sequential dependency:</strong> All positions computed simultaneously
          </List.Item>
          <List.Item>
            <strong>Matrix operations:</strong> Highly optimized on GPUs
          </List.Item>
          <List.Item>
            <strong>Training speedup:</strong> 10-100× faster than RNNs on modern hardware
          </List.Item>
          <List.Item>
            <strong>Inference:</strong> Autoregressive models still sequential (but faster per step)
          </List.Item>
        </List>

        <Text mt="lg">
          This parallelizability is a key advantage of attention-based models over RNNs.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/attention-parallelization-diagram.png"
            alt="Parallel computation in attention mechanism vs sequential RNN processing"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Parallel computation in attention mechanism vs sequential RNN processing
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Positional Information in Attention</Title>

        <Text mt="md">
          Attention is permutation-invariant: it treats input as a set, not a sequence.
        </Text>

        <Text mt="lg">
          Without positional information:
        </Text>
        <BlockMath>{`
          \\text{Attention}([x_1, x_2, x_3], ...) = \\text{Attention}([x_2, x_3, x_1], ...)
        `}</BlockMath>

        <Text mt="lg">
          <strong>Solution:</strong> Add positional encodings to input embeddings
        </Text>

        <Text mt="md">
          This is handled at the model level (before attention), which we'll cover in the Transformer section.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/positional-encoding-visualization.png"
            alt="Positional encoding visualization in attention mechanism"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Positional encoding visualization in attention mechanism
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Attention Patterns</Title>

        <Flex direction="column" align="center" mt="md" mb="md">
          <Image
            src="/assets/data-science-practice/module8/attention-patterns.png"
            alt="Different attention patterns in multi-head attention"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Visualization of different attention patterns learned by different heads
          </Text>
        </Flex>

        <Text mt="md">
          Different attention heads learn to capture different linguistic phenomena:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Syntactic dependencies (subject-verb agreement)</List.Item>
          <List.Item>Long-range dependencies</List.Item>
          <List.Item>Local context</List.Item>
          <List.Item>Positional patterns</List.Item>
        </List>
      </div>

    </>
  );
};

export default AttentionLayer;
