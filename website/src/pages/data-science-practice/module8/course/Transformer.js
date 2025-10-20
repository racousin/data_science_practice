import React from 'react';
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from 'components/CodeBlock';

export default function Transformer() {
  return (
    <div>
      <div data-slide>
        <Title order={1}>Transformer Architecture</Title>
        <Text size="lg" mt="md">
          The foundation of modern large language models
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Overview</Title>
        <Text mt="md">
          The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017),
          revolutionized natural language processing by replacing recurrent neural networks with a purely
          attention-based mechanism.
        </Text>
        <List mt="md" spacing="sm">
          <List.Item>Eliminates sequential processing, enabling full parallelization during training</List.Item>
          <List.Item>Captures long-range dependencies without the vanishing gradient problem of RNNs</List.Item>
          <List.Item>Scales efficiently to billions of parameters</List.Item>
          <List.Item>Forms the basis for BERT, GPT, T5, and most modern language models</List.Item>
        </List>
        <Flex justify="center" mt="xl">
          <Image
            src="/assets/data-science-practice/module8/transformer.png"
            alt="Transformer Architecture Overview"
            maw={500}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Core Components</Title>
        <Text mt="md">
          The Transformer architecture consists of several fundamental building blocks:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item><strong>Self-Attention Mechanism</strong>: Computes relationships between all positions in the sequence</List.Item>
          <List.Item><strong>Multi-Head Attention</strong>: Parallel attention operations with different learned projections</List.Item>
          <List.Item><strong>Positional Encoding</strong>: Injects sequence order information</List.Item>
          <List.Item><strong>Feed-Forward Networks</strong>: Position-wise fully connected layers</List.Item>
          <List.Item><strong>Layer Normalization</strong>: Stabilizes training</List.Item>
          <List.Item><strong>Residual Connections</strong>: Enables gradient flow through deep networks</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Token Embeddings</Title>
        <Text mt="md">
          Before processing, discrete token indices must be converted to continuous vector representations.
        </Text>
        <Text mt="lg">
          <strong>Embedding layer:</strong> Learnable lookup table mapping token indices to dense vectors
        </Text>
        <BlockMath>
          {`\\text{Embedding}: \\{0, 1, 2, ..., V-1\\} \\rightarrow \\mathbb{R}^{d}`}
        </BlockMath>

        <Title order={3} mt="lg">Dimensions</Title>
        <Text mt="sm">
          For a vocabulary of size <InlineMath>V</InlineMath> and embedding dimension <InlineMath>d</InlineMath>:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Embedding matrix: <InlineMath>{`E \\in \\mathbb{R}^{V \\times d}`}</InlineMath></List.Item>
          <List.Item>Input token indices: <InlineMath>{`[t_1, t_2, ..., t_n]`}</InlineMath> where <InlineMath>{`t_i \\in \\{0, ..., V-1\\}`}</InlineMath></List.Item>
          <List.Item>Output embeddings: <InlineMath>{`X \\in \\mathbb{R}^{n \\times d}`}</InlineMath></List.Item>
        </List>

        <Text mt="md">
          Each token <InlineMath>{`t_i`}</InlineMath> is mapped to embedding vector <InlineMath>{`E[t_i] \\in \\mathbb{R}^{d}`}</InlineMath>.
          These embeddings are learned during training to capture semantic relationships between tokens.
        </Text>
                <Flex justify="center" mt="xl">
          <Image
            src="/assets/data-science-practice/module8/embedding.png"
            alt="Residual Connections in Transformers"
            maw={400}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Embedding Implementation</Title>
        <Text mt="md">
          PyTorch implementation of token embeddings:
        </Text>
        <CodeBlock
          language="python"
          code={`import torch.nn as nn`}
        />
        <CodeBlock
          language="python"
          code={`class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Embedding lookup table: (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)`}
        />
        <CodeBlock
          language="python"
          code={`        # Scale embeddings by sqrt(d_model) as in original paper
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len) - token indices
        # output: (batch, seq_len, d_model)
        return self.embedding(x) * (self.d_model ** 0.5)`}
        />
        <Text mt="md">
          The scaling factor <InlineMath>{`\\sqrt{d}`}</InlineMath> is applied so that embeddings and positional
          encodings have similar magnitudes when combined. For vocabulary size <InlineMath>{`V = 30000`}</InlineMath> and
          <InlineMath>{`d = 512`}</InlineMath>, the embedding layer contains <InlineMath>{`30000 \\times 512 = 15.36`}</InlineMath> million
          parameters.
        </Text>
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

        <BlockMath>
          {`PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right)`}
        </BlockMath>
        <BlockMath>
          {`PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)`}
        </BlockMath>
        <Text mt="md">
          Where <InlineMath>pos</InlineMath> is the position in the sequence, <InlineMath>i</InlineMath> is
          the dimension index, and <InlineMath>d</InlineMath> is the embedding dimension (typically 512 or 768).
        </Text>

        <Title order={3} mt="lg">Dimensions</Title>
        <Text mt="sm">
          For an input sequence of length <InlineMath>n</InlineMath> with embeddings
          <InlineMath>{`X \\in \\mathbb{R}^{n \\times d}`}</InlineMath>:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Positional encoding: <InlineMath>{`PE \\in \\mathbb{R}^{n \\times d}`}</InlineMath></List.Item>
          <List.Item>Combined: <InlineMath>{`X + PE \\in \\mathbb{R}^{n \\times d}`}</InlineMath></List.Item>
        </List>
        <Text mt="sm">
          This encoding is deterministic and allows the model to extrapolate to sequence lengths longer than
          those seen during training. Different frequencies encode different positional information, with lower
          dimensions capturing coarse position and higher dimensions capturing fine-grained position.
        </Text>
        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/position.png"
            alt="Positional encoding visualization in attention mechanism"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Positional Encoding Implementation</Title>
        <Text mt="md">
          PyTorch implementation of sinusoidal positional encoding:
        </Text>
        <CodeBlock
          language="python"
          code={`import torch
import torch.nn as nn`}
        />
        <CodeBlock
          language="python"
          code={`class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()`}
        />
        <CodeBlock
          language="python"
          code={`        # Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)

        # Create dimension indices [0, 2, 4, ..., d_model-2]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            (-torch.log(torch.tensor(10000.0)) / d_model))  # (d_model/2,)`}
        />
        <CodeBlock
          language="python"
          code={`        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        self.register_buffer('pe', pe)  # (max_len, d_model)`}
        />
        <CodeBlock
          language="python"
          code={`    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # pe[:x.size(1)]: (seq_len, d_model) - broadcasted across batch
        return x + self.pe[:x.size(1)]  # (batch, seq_len, d_model)`}
        />
        <Text mt="md">
          The positional encoding <InlineMath>{`\\text{pe} \\in \\mathbb{R}^{max\\_len \\times d}`}</InlineMath> is
          precomputed and added to input embeddings. Broadcasting handles the batch dimension automatically.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Feed-Forward Networks</Title>
        <Text mt="md">
          Each Transformer layer contains a position-wise feed-forward network applied identically to each position:
        </Text>
        <BlockMath>
          {`FFN(X) = \\max(0, XW_1 + b_1)W_2 + b_2`}
        </BlockMath>

        <Title order={3} mt="lg">Weight Matrices</Title>
        <Text mt="sm">
          Starting from input <InlineMath>{`X \\in \\mathbb{R}^{n \\times d}`}</InlineMath>:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`W_1 \\in \\mathbb{R}^{d \\times d_{ff}}`}</InlineMath>: First projection matrix</List.Item>
          <List.Item><InlineMath>{`b_1 \\in \\mathbb{R}^{d_{ff}}`}</InlineMath>: First bias vector</List.Item>
          <List.Item><InlineMath>{`W_2 \\in \\mathbb{R}^{d_{ff} \\times d}`}</InlineMath>: Second projection matrix</List.Item>
          <List.Item><InlineMath>{`b_2 \\in \\mathbb{R}^{d}`}</InlineMath>: Second bias vector</List.Item>
        </List>

        <Title order={3} mt="lg">Dimensional Flow</Title>
        <Text mt="sm">
          The inner layer typically has dimension <InlineMath>{`d_{ff} = 4 \\times d`}</InlineMath>.
          For <InlineMath>{`d = 512`}</InlineMath>, this means <InlineMath>{`d_{ff} = 2048`}</InlineMath>.
        </Text>
        <BlockMath>{`
          \\begin{align*}
          X &\\in \\mathbb{R}^{n \\times d} \\\\
          XW_1 + b_1 &\\in \\mathbb{R}^{n \\times d_{ff}} \\\\
          \\max(0, XW_1 + b_1) &\\in \\mathbb{R}^{n \\times d_{ff}} \\\\
          FFN(X) &\\in \\mathbb{R}^{n \\times d}
          \\end{align*}
        `}</BlockMath>
        <Text mt="sm">
          The ReLU activation introduces non-linearity. Modern variants use GELU or SwiGLU activations.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Feed-Forward Network Implementation</Title>
        <Text mt="md">
          Position-wise feed-forward network in PyTorch:
        </Text>
        <CodeBlock
          language="python"
          code={`class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()`}
        />
        <CodeBlock
          language="python"
          code={`        self.linear1 = nn.Linear(d_model, d_ff)  # W1: (d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2: (d_ff, d_model)
        self.dropout = nn.Dropout(dropout)`}
        />
        <CodeBlock
          language="python"
          code={`    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)        # → (batch, seq_len, d_ff)
        x = torch.relu(x)
        x = self.dropout(x)`}
        />
        <CodeBlock
          language="python"
          code={`        x = self.linear2(x)        # → (batch, seq_len, d_model)
        return x`}
        />
        <Text mt="md">
          Weight matrix dimensions: <InlineMath>{`W_1 \\in \\mathbb{R}^{d \\times d_{ff}}`}</InlineMath>,
          <InlineMath>{`W_2 \\in \\mathbb{R}^{d_{ff} \\times d}`}</InlineMath>.
          For <InlineMath>{`d = 512`}</InlineMath> and <InlineMath>{`d_{ff} = 2048`}</InlineMath>,
          this layer has <InlineMath>{`512 \\times 2048 + 2048 \\times 512 = 2{,}097{,}152`}</InlineMath> parameters
          (excluding biases).
        </Text>
      </div>
      <div data-slide>
        <Title order={2}>Layer Normalization</Title>
        <Text mt="md">
          Layer normalization normalizes across the feature dimension for each example independently:
        </Text>
        <BlockMath>
          {`LayerNorm(X) = \\gamma \\odot \\frac{X - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta`}
        </BlockMath>

        <Title order={3} mt="lg">Dimensions</Title>
        <Text mt="sm">
          For input <InlineMath>{`X \\in \\mathbb{R}^{n \\times d}`}</InlineMath>:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`\\mu, \\sigma^2 \\in \\mathbb{R}^{n}`}</InlineMath>: Computed across <InlineMath>d</InlineMath> dimension for each position</List.Item>
          <List.Item><InlineMath>{`\\gamma, \\beta \\in \\mathbb{R}^{d}`}</InlineMath>: Learned scale and shift parameters</List.Item>
          <List.Item><InlineMath>{`LayerNorm(X) \\in \\mathbb{R}^{n \\times d}`}</InlineMath>: Output has same shape as input</List.Item>
        </List>

        <Text mt="lg">
          <strong>Placement variants:</strong>
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item><strong>Post-norm</strong> (original): <InlineMath>{`LayerNorm(X + Sublayer(X))`}</InlineMath></List.Item>
          <List.Item><strong>Pre-norm</strong> (modern): <InlineMath>{`X + Sublayer(LayerNorm(X))`}</InlineMath></List.Item>
        </List>
        <Text mt="sm">
          Pre-norm is now standard as it provides better gradient flow and enables training of deeper models
          without learning rate warmup.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Residual Connections</Title>
        <Text mt="md">
          Residual connections create skip paths around each sublayer:
        </Text>
        <BlockMath>
          {`output = X + Sublayer(X)`}
        </BlockMath>
        <Text mt="md">
          This design provides several benefits:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item>
            <strong>Gradient flow</strong>: Gradients can flow directly through the identity path,
            mitigating vanishing gradients in deep networks
          </List.Item>
          <List.Item>
            <strong>Identity initialization</strong>: The network can learn to use or bypass each layer
          </List.Item>
          <List.Item>
            <strong>Ensemble effect</strong>: Creates an implicit ensemble of paths of different depths
          </List.Item>
        </List>
        <Text mt="md">
          Combined with layer normalization, residual connections enable training of Transformers with
          hundreds of layers.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/assets/data-science-practice/module8/residual_connection.png"
            alt="Residual Connections in Transformers"
            maw={400}
          />
        </Flex>
      </div>


      <div data-slide>
        <Title order={2}>Encoder Architecture</Title>
        <Title order={3} mt="lg">Dimensional Flow</Title>
        <Text mt="sm">
          Starting from input <InlineMath>{`X_b \\in \\mathbb{R}^{n \\times d}`}</InlineMath> for block <InlineMath>b</InlineMath>:
        </Text>
        <BlockMath>{`
          \\begin{align*}
          \\text{1. Self-Attention:} \\\\
          \\text{attn\\_out} &= \\text{MultiHeadAttn}(X_b, X_b, X_b) \\in \\mathbb{R}^{n \\times d} \\\\
          X'_b &= \\text{LayerNorm}(X_b + \\text{attn\\_out}) \\in \\mathbb{R}^{n \\times d} \\\\
          \\\\
          \\text{2. Feed-Forward:} \\\\
          \\text{ff\\_out} &= \\text{FFN}(X'_b) \\in \\mathbb{R}^{n \\times d} \\\\
          X_{b+1} &= \\text{LayerNorm}(X'_b + \\text{ff\\_out}) \\in \\mathbb{R}^{n \\times d}
          \\end{align*}
        `}</BlockMath>

        <Text mt="md">
          Each block <InlineMath>b</InlineMath> processes <InlineMath>{`X_b`}</InlineMath> through attention to produce intermediate output <InlineMath>{`X'_b`}</InlineMath>,
          then through feed-forward to produce <InlineMath>{`X_{b+1}`}</InlineMath>. All tensors maintain shape <InlineMath>{`\\mathbb{R}^{n \\times d}`}</InlineMath>,
          enabling residual connections. This structure is repeated <InlineMath>N</InlineMath> times (typically <InlineMath>N=6</InlineMath> or <InlineMath>N=12</InlineMath>) to form the complete encoder.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Transformer Block Implementation</Title>
        <Text mt="md">
          Complete encoder block combining attention, feed-forward, and normalization:
        </Text>
        <CodeBlock
          language="python"
          code={`class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()`}
        />
        <CodeBlock
          language="python"
          code={`        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)`}
        />
        <CodeBlock
          language="python"
          code={`    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Self-attention with residual and norm (pre-norm)
        normed = self.norm1(x)  # (batch, seq_len, d_model)
        attn_out, _ = self.attention(normed, normed, normed)  # (batch, seq_len, d_model)
        x = x + attn_out  # (batch, seq_len, d_model)`}
        />
        <CodeBlock
          language="python"
          code={`        # Feed-forward with residual and norm
        normed = self.norm2(x)  # (batch, seq_len, d_model)
        ff_out = self.ff(normed)  # (batch, seq_len, d_model)
        x = x + ff_out  # (batch, seq_len, d_model)
        return x  # (batch, seq_len, d_model)`}
        />
        <Text mt="md">
          This implements pre-norm architecture. All intermediate tensors maintain shape
          <InlineMath>{`\\mathbb{R}^{n \\times d}`}</InlineMath>,
          enabling residual connections throughout.
        </Text>

      </div>
      <div data-slide>
        <Title order={2}>Decoder Architecture</Title>
        <Text mt="md">
          The Transformer decoder adds two key modifications to the encoder structure:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item>
            <strong>Masked Self-Attention</strong>: Prevents positions from attending to future positions,
            ensuring autoregressive generation
          </List.Item>
          <List.Item>
            <strong>Cross-Attention</strong>: Attends to the encoder output, allowing the decoder to access
            source sequence information
          </List.Item>
        </List>

        <Title order={3} mt="lg">Dimensional Flow</Title>
        <Text mt="sm">
          For decoder block <InlineMath>b</InlineMath> input <InlineMath>{`X_b \\in \\mathbb{R}^{m \\times d}`}</InlineMath> and
          encoder output <InlineMath>{`\\text{enc\\_out} \\in \\mathbb{R}^{n \\times d}`}</InlineMath>:
        </Text>
        <BlockMath>{`
          \\begin{align*}
          \\text{1. Masked Self-Attention:} \\\\
          \\text{masked\\_attn\\_out} &= \\text{MaskedAttn}(X_b, X_b, X_b) \\in \\mathbb{R}^{m \\times d} \\\\
          X'_b &= \\text{LayerNorm}(X_b + \\text{masked\\_attn\\_out}) \\in \\mathbb{R}^{m \\times d} \\\\
          \\\\
          \\text{2. Cross-Attention:} \\\\
          \\text{cross\\_attn\\_out} &= \\text{CrossAttn}(X'_b, \\text{enc\\_out}, \\text{enc\\_out}) \\in \\mathbb{R}^{m \\times d} \\\\
          X''_b &= \\text{LayerNorm}(X'_b + \\text{cross\\_attn\\_out}) \\in \\mathbb{R}^{m \\times d} \\\\
          \\\\
          \\text{3. Feed-Forward:} \\\\
          \\text{ff\\_out} &= \\text{FFN}(X''_b) \\in \\mathbb{R}^{m \\times d} \\\\
          X_{b+1} &= \\text{LayerNorm}(X''_b + \\text{ff\\_out}) \\in \\mathbb{R}^{m \\times d}
          \\end{align*}
        `}</BlockMath>
        <Text mt="md">
          Each decoder block <InlineMath>b</InlineMath> processes <InlineMath>{`X_b`}</InlineMath> through three sublayers:
          masked self-attention produces <InlineMath>{`X'_b`}</InlineMath>, cross-attention produces <InlineMath>{`X''_b`}</InlineMath>,
          and feed-forward produces <InlineMath>{`X_{b+1}`}</InlineMath>. The mask in step 1 ensures position <InlineMath>i</InlineMath> can only attend to positions
          <InlineMath>{`j \\leq i`}</InlineMath>. Target sequence length <InlineMath>m</InlineMath> is independent of source sequence length <InlineMath>n</InlineMath>.
        </Text>

      </div>

      <div data-slide>
        <Title order={2}>Full Encoder-Decoder Architecture</Title>
        <Text mt="md">
          The complete Transformer combines encoder and decoder stacks:
        </Text>

        <Title order={3} mt="lg">Dimensional Flow</Title>
        <Text mt="sm">
          Source sequence with length <InlineMath>n</InlineMath>, target sequence with length <InlineMath>m</InlineMath>,
          vocabulary size <InlineMath>V</InlineMath>:
        </Text>
        <BlockMath>{`
          \\begin{align*}
          \\text{Encoder:} \\\\
          \\text{src\\_emb} &= \\text{Embedding}(\\text{source}) + \\text{PosEncoding} \\in \\mathbb{R}^{n \\times d} \\\\
          \\text{enc\\_out} &= \\text{EncoderStack}(\\text{src\\_emb}) \\in \\mathbb{R}^{n \\times d} \\\\
          \\\\
          \\text{Decoder:} \\\\
          \\text{tgt\\_emb} &= \\text{Embedding}(\\text{target}) + \\text{PosEncoding} \\in \\mathbb{R}^{m \\times d} \\\\
          \\text{dec\\_out} &= \\text{DecoderStack}(\\text{tgt\\_emb}, \\text{enc\\_out}) \\in \\mathbb{R}^{m \\times d} \\\\
          \\text{logits} &= \\text{Linear}(\\text{dec\\_out}) \\in \\mathbb{R}^{m \\times V}
          \\end{align*}
        `}</BlockMath>
        <Text mt="md">
          During training, teacher forcing is used: the decoder receives the true target sequence as input.
          During inference, the decoder generates one token at a time autoregressively.
        </Text>
      </div>



      <div data-slide>
        <Title order={2}>Complete Transformer Model</Title>
        <Text mt="md">
          Full encoder-decoder Transformer implementation:
        </Text>
        <CodeBlock
          language="python"
          code={`class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()`}
        />
        <CodeBlock
          language="python"
          code={`        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)`}
        />
        <CodeBlock
          language="python"
          code={`        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)`}
        />
        <CodeBlock
          language="python"
          code={`        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: (batch, n), tgt: (batch, m)
        src_emb = self.pos_encoding(self.src_embedding(src))  # (batch, n, d_model)
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))  # (batch, m, d_model)`}
        />
        <CodeBlock
          language="python"
          code={`        enc_out = self.encoder(src_emb, mask=src_mask)  # (batch, n, d_model)
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)  # (batch, m, d_model)
        return self.output_projection(dec_out)  # (batch, m, vocab_size)`}
        />
        <Text mt="md">
          Input dimensions: source tokens of length <InlineMath>n</InlineMath>,
          target tokens of length <InlineMath>m</InlineMath>.
          Output logits <InlineMath>{`\\in \\mathbb{R}^{m \\times V}`}</InlineMath> where <InlineMath>V</InlineMath> is vocabulary size.
        </Text>
      </div>


      <div data-slide>
        <Title order={2}>Parameter Counting</Title>
        <Text mt="md">
          Detailed parameter breakdown for Transformer base model (<InlineMath>{`d = 512`}</InlineMath>,
          <InlineMath>{`d_{ff} = 2048`}</InlineMath>, <InlineMath>h = 8</InlineMath>, <InlineMath>N = 6</InlineMath>):
        </Text>
        <Table mt="md">
          <thead>
            <tr>
              <th>Component</th>
              <th>Parameters</th>
              <th>Per Layer</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Attention Q, K, V projections</td>
              <td><InlineMath>{`3 \\times 512 \\times 512 = 786{,}432`}</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Attention output projection</td>
              <td><InlineMath>{`512 \\times 512 = 262{,}144`}</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Feed-forward W1</td>
              <td><InlineMath>{`512 \\times 2048 = 1{,}048{,}576`}</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Feed-forward W2</td>
              <td><InlineMath>{`2048 \\times 512 = 1{,}048{,}576`}</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Layer norm (2 per layer)</td>
              <td><InlineMath>{`2 \\times 2 \\times 512 = 2{,}048`}</InlineMath></td>
              <td>Yes</td>
            </tr>
          </tbody>
        </Table>
        <Text mt="md">
          Total per encoder layer: <InlineMath>{`\\approx 3.1`}</InlineMath> million parameters
        </Text>
        <Text>
          Total for 6 encoder layers: <InlineMath>{`\\approx 18.6`}</InlineMath> million parameters
        </Text>

      </div>


      <div data-slide>
        <Title order={2}>Architectural Variants</Title>
        <Text mt="md">
          The original encoder-decoder Transformer has spawned three major architectural variants:
        </Text>
        <List mt="md" spacing="md">
          <List.Item>
            <strong>Encoder-only (BERT)</strong>: Bidirectional self-attention throughout, suited for
            understanding tasks like classification and question answering
          </List.Item>
          <List.Item>
            <strong>Decoder-only (GPT)</strong>: Causal (masked) self-attention only, designed for
            autoregressive generation tasks
          </List.Item>
          <List.Item>
            <strong>Encoder-Decoder (T5)</strong>: Full encoder-decoder structure, versatile for
            sequence-to-sequence tasks like translation and summarization
          </List.Item>
        </List>
        <Text mt="md">
          Each variant optimizes for different use cases while maintaining the core attention mechanism.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/assets/data-science-practice/module8/bertgpt.png"
            alt="Transformer Architecture Variants"
            maw={550}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>BERT Architecture</Title>
        <Text mt="md">
          BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2019) uses
          only the encoder stack with bidirectional attention.
        </Text>
        <Text mt="md">
          <strong>Key characteristics:</strong>
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item>
            <strong>Masked Language Modeling (MLM)</strong>: Randomly masks 15% of tokens and predicts them
            using bidirectional context
          </List.Item>
          <List.Item>
            <strong>Next Sentence Prediction (NSP)</strong>: Predicts if two segments are consecutive
            (removed in RoBERTa)
          </List.Item>
          <List.Item>
            <strong>Bidirectional attention</strong>: Each token attends to all other tokens without masking
          </List.Item>
        </List>
        <Text mt="md">
          <strong>Model sizes:</strong>
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item>BERT-Base: 12 layers, 768 dimensions, 12 heads, 110M parameters</List.Item>
          <List.Item>BERT-Large: 24 layers, 1024 dimensions, 16 heads, 340M parameters</List.Item>
        </List>
        <Text mt="md">
          BERT excels at tasks requiring understanding of full context: sentiment analysis, named entity
          recognition, question answering.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>GPT Architecture</Title>
        <Text mt="md">
          GPT (Generative Pre-trained Transformer) uses only the decoder stack with causal masking.
        </Text>
        <Text mt="md">
          <strong>Key characteristics:</strong>
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item>
            <strong>Autoregressive generation</strong>: Predicts next token given all previous tokens
          </List.Item>
          <List.Item>
            <strong>Causal masking</strong>: Position <InlineMath>i</InlineMath> can only attend to
            positions <InlineMath>{`j \\leq i`}</InlineMath>
          </List.Item>
          <List.Item>
            <strong>Unidirectional context</strong>: Only left-to-right information flow
          </List.Item>
        </List>
        <Text mt="md">
          <strong>Evolution:</strong>
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item>GPT-1: 12 layers, 768 dimensions, 117M parameters</List.Item>
          <List.Item>GPT-2: 48 layers, 1600 dimensions, 1.5B parameters</List.Item>
          <List.Item>GPT-3: 96 layers, 12288 dimensions, 175B parameters</List.Item>
        </List>
        <Text mt="md">
          GPT models are optimized for generation tasks: text completion, dialogue, creative writing.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>T5 Architecture</Title>
        <Text mt="md">
          T5 (Text-to-Text Transfer Transformer, Raffel et al., 2020) uses the full encoder-decoder
          architecture with a unified text-to-text framework.
        </Text>
        <Text mt="md">
          <strong>Text-to-text paradigm:</strong> All tasks are cast as text generation:
        </Text>
        <CodeBlock
          language="text"
          code={`Translation: "translate English to German: Hello" → "Hallo"
Classification: "sentiment: I love this!" → "positive"
Summarization: "summarize: [article]" → "[summary]"
Q&A: "question: What is AI?" → "Artificial Intelligence"`}
        />
        <Text mt="md">
          <strong>Architectural modifications:</strong>
        </Text>
        <List mt="sm" spacing="sm">
          <List.Item>Simplified layer normalization (removes bias term)</List.Item>
          <List.Item>Relative positional embeddings instead of absolute</List.Item>
          <List.Item>Span corruption objective: masks contiguous spans during pre-training</List.Item>
        </List>
        <Text mt="md">
          T5 models range from T5-Small (60M) to T5-11B (11B parameters).
        </Text>
      </div>
      <div data-slide>
        <Flex justify="center" mt="xl">
          <Image
            src="/assets/data-science-practice/module8/head-layer.gif"
            alt="Transformer Architecture Variants"
            maw={550}
          />
        </Flex>
      </div>
    </div>
  );
}
