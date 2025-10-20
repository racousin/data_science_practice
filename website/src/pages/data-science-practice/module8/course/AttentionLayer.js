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
          \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right)V
        `}</BlockMath>

        <Text mt="lg">
          Where Q (query), K (keys), and V (values) are matrices representing different aspects
          of the input.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/qkv.png"
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

        <Text mt="md">
          Starting from an input sequence <InlineMath>{`X \\in \\mathbb{R}^{n \\times d}`}</InlineMath>:
        </Text>

        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`n`}</InlineMath>: sequence length (number of tokens)</List.Item>
          <List.Item><InlineMath>{`d`}</InlineMath>: embedding dimension</List.Item>
        </List>

        <Title order={3} mt="lg">Weight Matrices</Title>
        <Text mt="sm">
          We use learned projection matrices to transform X into Q, K, V:
        </Text>

        <List spacing="xs" mt="sm">
          <List.Item>Query (Q) :  what we're looking for. <InlineMath>{`W_Q \\in \\mathbb{R}^{d \\times d}`}</InlineMath>: Query projection matrix</List.Item>
          <List.Item>Key (K) :  what we have to attend to. <InlineMath>{`W_K \\in \\mathbb{R}^{d \\times d}`}</InlineMath>: Key projection matrix</List.Item>
          <List.Item>Value (V) :  the actual content. <InlineMath>{`W_V \\in \\mathbb{R}^{d \\times d}`}</InlineMath>: Value projection matrix</List.Item>
        </List>

        <Title order={3} mt="lg">Resulting Q, K, V</Title>
        <BlockMath>{`Q = XW_Q \\in \\mathbb{R}^{n \\times d}`}</BlockMath>
        <BlockMath>{`K = XW_K \\in \\mathbb{R}^{n \\times d}`}</BlockMath>
        <BlockMath>{`V = XW_V \\in \\mathbb{R}^{n \\times d}`}</BlockMath>
      </div>

      <div data-slide>
        <Title order={2}>Self-Attention:</Title>

        <Text mt="md">
          For self-attention, Q, K, V all come from the same sequence X:
        </Text>

        <BlockMath>{`
          \\begin{align*}
          S = QK^T &\\in \\mathbb{R}^{n \\times n} \\\\
          A = \\text{softmax}\\left(\\frac{S}{\\sqrt{d}}\\right) &\\in \\mathbb{R}^{n \\times n} \\\\
          \\text{Output} = AV &\\in \\mathbb{R}^{n \\times d}
          \\end{align*}
        `}</BlockMath>

        <Text mt="lg">
          Each of the <InlineMath>{`n`}</InlineMath> positions attends to all <InlineMath>{`n`}</InlineMath> positions.
        </Text>
                <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/self-attention.png"
            alt="Attention mechanism diagram showing Q, K, V interaction"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Cross-Attention: Two Sequences</Title>

        <Text mt="md">
          For cross-attention between two sequences of different lengths:
        </Text>

        <List spacing="xs" mt="sm">
          <List.Item>Sequence 1: <InlineMath>{`X_1 \\in \\mathbb{R}^{n_1 \\times d}`}</InlineMath> (queries)</List.Item>
          <List.Item>Sequence 2: <InlineMath>{`X_2 \\in \\mathbb{R}^{n_2 \\times d}`}</InlineMath> (keys and values)</List.Item>
        </List>

        <BlockMath>{`
          \\begin{align*}
          Q = X_1 W_Q &\\in \\mathbb{R}^{n_1 \\times d} \\\\
          K = X_2 W_K &\\in \\mathbb{R}^{n_2 \\times d} \\\\
          V = X_2 W_V &\\in \\mathbb{R}^{n_2 \\times d}
          \\end{align*}
        `}</BlockMath>

        <Title order={3} mt="lg">Attention Computation</Title>
        <BlockMath>{`
          \\begin{align*}
          S = QK^T &\\in \\mathbb{R}^{n_1 \\times n_2} \\\\
          A = \\text{softmax}\\left(\\frac{S}{\\sqrt{d}}\\right) &\\in \\mathbb{R}^{n_1 \\times n_2} \\\\
          \\text{Output} = AV &\\in \\mathbb{R}^{n_1 \\times d}
          \\end{align*}
        `}</BlockMath>

        <Text mt="lg">
          Each of the <InlineMath>{`n_1`}</InlineMath> positions in sequence 1 attends to all <InlineMath>{`n_2`}</InlineMath> positions in sequence 2.
        </Text>
                <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/cross_attention.webp"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Why Scale by √d?</Title>

        <Text mt="md">
          Without scaling, dot products grow large in magnitude as dimensionality increases.
        </Text>

        <Text mt="lg">
          For <InlineMath>{`q, k \\sim \\mathcal{N}(0, 1)`}</InlineMath>:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Expected value: <InlineMath>{`E[q \\cdot k] = 0`}</InlineMath></List.Item>
          <List.Item>Variance: <InlineMath>{`\\text{Var}(q \\cdot k) = d`}</InlineMath></List.Item>
        </List>

        <Text mt="lg">
          Large dot products push softmax into regions with small gradients, hindering learning.
        </Text>

        <Text mt="lg">
          <strong>Solution:</strong> Divide by <InlineMath>{`\\sqrt{d}`}</InlineMath> to keep variance ≈ 1
        </Text>

        <BlockMath>{`
          \\text{Var}\\left(\\frac{q \\cdot k}{\\sqrt{d}}\\right) = 1
        `}</BlockMath>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/softmax.webp"
            alt="Effect of scaling factor on softmax gradient"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />

        </Flex>
      </div>


      <div data-slide>
        <Title order={2}>Attention Implementation</Title>

        <Text mt="md">
          First, define the function signature:
        </Text>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):`}
        />

        <Text mt="md">
          Extract the dimension and compute attention scores:
        </Text>

        <CodeBlock
          language="python"
          code={`    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / torch.sqrt(torch.tensor(d, dtype=torch.float32))`}
        />
        <Text mt="md">
          Apply optional masking and softmax:
        </Text>

        <CodeBlock
          language="python"
          code={`    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)`}
        />
        <Text mt="md">
          Finally, compute weighted sum of values:
        </Text>

        <CodeBlock
          language="python"
          code={`    output = torch.matmul(attention_weights, V)
    return output, attention_weights`}
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
          Used in language modeling where position i cannot attend to positions j {'>'}  i.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/mask.png"
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

        <Text mt="md">
          Create a lower triangular matrix for causal masking:
        </Text>

        <CodeBlock
          language="python"
          code={`import torch

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask`}
        />

        <Text mt="md">
          Example output for sequence length 5:
        </Text>

        <CodeBlock
          language="python"
          code={`mask = create_causal_mask(5)
print(mask)
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])`}
        />

        <Text mt="md">
          Apply the mask in attention:
        </Text>

        <CodeBlock
          language="python"
          code={`output, attn = scaled_dot_product_attention(Q, K, V, mask=mask.unsqueeze(0))`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention</Title>

        <Text mt="md">
          Multi-head attention runs multiple attention operations in parallel, allowing the model
          to attend to different representation subspaces.
        </Text>

        <BlockMath>{`
          \\text{MultiHead}(X) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O
        `}</BlockMath>

        <BlockMath>{`
          \\text{where } \\text{head}_i = \\text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
        `}</BlockMath>
        <Text mt="lg">
          <strong>Weight matrices:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`W_i^Q, W_i^K, W_i^V \\in \\mathbb{R}^{d \\times d/h}`}</InlineMath>: Projections for head i</List.Item>
          <List.Item><InlineMath>{`W^O \\in \\mathbb{R}^{d \\times d}`}</InlineMath>: Output projection</List.Item>
        </List>
        <Text mt="lg">
          <strong>Dimensional flow:</strong> Starting from input <InlineMath>{`X \\in \\mathbb{R}^{n \\times d}`}</InlineMath>
        </Text>

        <List spacing="xs" mt="sm">
          <List.Item> <InlineMath>{`XW_i^Q \\in \\mathbb{R}^{n \\times d/h}`}</InlineMath> , <InlineMath>{`XW_i^K \\in \\mathbb{R}^{n \\times d/h}, XW_i^V \\in \\mathbb{R}^{n \\times d/h}`}</InlineMath>
          </List.Item>
          <List.Item> <InlineMath>{`\\text{head}_i \\in \\mathbb{R}^{n \\times d/h}`}</InlineMath>
          </List.Item>
          <List.Item> <InlineMath>{`\\text{Concat}(\\text{head}_1, ..., \\text{head}_h) \\in \\mathbb{R}^{n \\times d}`}</InlineMath>
          </List.Item>
          <List.Item> <InlineMath>{`\\text{MultiHead}(X) \\in \\mathbb{R}^{n \\times d}`}</InlineMath>
          </List.Item>
        </List>



        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/multihead.png"
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
          Each head operates on dimension <InlineMath>{`d/h`}</InlineMath> where <InlineMath>{`h`}</InlineMath> is the number of heads.
        </Text>

        <Text mt="lg">
          <strong>Example:</strong> <InlineMath>{`d = 512, h = 8`}</InlineMath>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Each head dimension: <InlineMath>{`512 / 8 = 64`}</InlineMath></List.Item>
          <List.Item>Each head operates on 64-dimensional subspace</List.Item>
          <List.Item>8 heads capture different aspects of relationships</List.Item>
        </List>

      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention Implementation</Title>

        <Text mt="md">
          Initialize with projections for Q, K, V and output:
        </Text>

        <CodeBlock
          language="python"
          code={`import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.d_head = d // num_heads`}
        />

        <CodeBlock
          language="python"
          code={`        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)`}
        />

        <Text mt="md">
          Split embeddings into multiple heads:
        </Text>

        <CodeBlock
          language="python"
          code={`    def split_heads(self, x):
        batch_size, seq_len, d = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention: Forward Pass</Title>

        <Text mt="md">
          Apply linear projections and split into heads:
        </Text>

        <CodeBlock
          language="python"
          code={`    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))`}
        />

        <Text mt="md">
          Compute attention for all heads in parallel:
        </Text>

        <CodeBlock
          language="python"
          code={`        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.d_head))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))`}
        />

        <CodeBlock
          language="python"
          code={`        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)`}
        />

        <Text mt="md">
          Concatenate heads and apply output projection:
        </Text>

        <CodeBlock
          language="python"
          code={`        batch_size = Q.shape[0]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d)
        return self.W_o(output), attention_weights`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Multi-Head Attention Example</Title>

        <Text mt="md">
          Initialize multi-head attention with 8 heads:
        </Text>

        <CodeBlock
          language="python"
          code={`d = 512
num_heads = 8
mha = MultiHeadAttention(d, num_heads)`}
        />

        <Text mt="md">
          Create input and apply self-attention:
        </Text>

        <CodeBlock
          language="python"
          code={`batch_size, n = 32, 10
x = torch.randn(batch_size, n, d)

output, attention_weights = mha(x, x, x)`}
        />

        <Text mt="md">
          Check output shapes:
        </Text>

        <CodeBlock
          language="python"
          code={`print(f"Output shape: {output.shape}")  # (32, 10, 512)
print(f"Attention weights: {attention_weights.shape}")  # (32, 8, 10, 10)
print(f"Dimension per head: {mha.d_head}")  # 64`}
        />
      </div>


      <div data-slide>
        <Title order={2}>Attention Computational Complexity</Title>

        <Text mt="md">
          For sequence length <InlineMath>{`n`}</InlineMath> and dimension <InlineMath>{`d`}</InlineMath>:
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


      </div>



      <div data-slide>
        <Title order={2}>Attention Patterns</Title>

        <Flex direction="column" align="center" mt="md" mb="md">
          <Image
            src="/assets/data-science-practice/module8/head.png"
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
