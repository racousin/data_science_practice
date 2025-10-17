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
            src="/modules/data-science-practice/module8/transformer-architecture.png"
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
        <Title order={2}>Positional Encoding</Title>
        <Text mt="md">
          Since Transformers have no inherent notion of sequence order, positional information must be
          explicitly added. The original paper uses sinusoidal functions:
        </Text>
        <BlockMath>
          {`PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)`}
        </BlockMath>
        <BlockMath>
          {`PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)`}
        </BlockMath>
        <Text mt="md">
          Where <InlineMath>pos</InlineMath> is the position in the sequence, <InlineMath>i</InlineMath> is
          the dimension index, and <InlineMath>d_{`{model}`}</InlineMath> is the model dimension (typically 512 or 768).
        </Text>
        <Text mt="sm">
          This encoding is deterministic and allows the model to extrapolate to sequence lengths longer than
          those seen during training. Different frequencies encode different positional information, with lower
          dimensions capturing coarse position and higher dimensions capturing fine-grained position.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/positional-encoding-visualization.png"
            alt="Positional Encoding Visualization"
            maw={500}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Feed-Forward Networks</Title>
        <Text mt="md">
          Each Transformer layer contains a position-wise feed-forward network applied identically to each position:
        </Text>
        <BlockMath>
          {`FFN(x) = \\max(0, xW_1 + b_1)W_2 + b_2`}
        </BlockMath>
        <Text mt="md">
          The inner layer typically has dimension <InlineMath>d_{`{ff}`} = 4 \\times d_{`{model}`}</InlineMath>.
          For <InlineMath>d_{`{model}`} = 512</InlineMath>, this means <InlineMath>d_{`{ff}`} = 2048</InlineMath>.
        </Text>
        <Text mt="sm">
          Input shape: <InlineMath>(batch, seq\_len, d_{`{model}`})</InlineMath>
        </Text>
        <Text>
          After <InlineMath>W_1</InlineMath>: <InlineMath>(batch, seq\_len, d_{`{ff}`})</InlineMath>
        </Text>
        <Text>
          After <InlineMath>W_2</InlineMath>: <InlineMath>(batch, seq\_len, d_{`{model}`})</InlineMath>
        </Text>
        <Text mt="sm">
          The ReLU activation introduces non-linearity. Modern variants use GELU or SwiGLU activations.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/feed-forward-network.png"
            alt="Feed-Forward Network Architecture"
            maw={450}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Layer Normalization</Title>
        <Text mt="md">
          Layer normalization normalizes across the feature dimension for each example independently:
        </Text>
        <BlockMath>
          {`LayerNorm(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta`}
        </BlockMath>
        <Text mt="md">
          Where <InlineMath>\mu</InlineMath> and <InlineMath>\sigma^2</InlineMath> are computed across
          the <InlineMath>d_{`{model}`}</InlineMath> dimension. Parameters <InlineMath>\gamma</InlineMath> and
          <InlineMath>\beta</InlineMath> are learned scale and shift vectors.
        </Text>
        <Text mt="md">
          <strong>Placement variants:</strong>
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item><strong>Post-norm</strong> (original): <InlineMath>LayerNorm(x + Sublayer(x))</InlineMath></List.Item>
          <List.Item><strong>Pre-norm</strong> (modern): <InlineMath>x + Sublayer(LayerNorm(x))</InlineMath></List.Item>
        </List>
        <Text mt="sm">
          Pre-norm is now standard as it provides better gradient flow and enables training of deeper models
          without learning rate warmup.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/layer-normalization.png"
            alt="Layer Normalization Comparison"
            maw={450}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Residual Connections</Title>
        <Text mt="md">
          Residual connections create skip paths around each sublayer:
        </Text>
        <BlockMath>
          {`output = x + Sublayer(x)`}
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
            src="/modules/data-science-practice/module8/residual-connections.png"
            alt="Residual Connections in Transformers"
            maw={400}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Transformer Block</Title>
        <Text mt="md">
          A complete Transformer encoder block combines all components:
        </Text>
        <CodeBlock
          language="text"
          code={`Input: x ∈ ℝ^(seq_len × d_model)

1. Multi-Head Self-Attention
   attn_out = MultiHeadAttn(x, x, x)
   x = LayerNorm(x + attn_out)

2. Feed-Forward Network
   ff_out = FFN(x)
   x = LayerNorm(x + ff_out)

Output: x ∈ ℝ^(seq_len × d_model)`}
        />
        <Text mt="md">
          This structure is repeated N times (typically N=6 or N=12) to form the complete encoder.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/transformer_block.png"
            alt="Transformer Block Architecture"
            maw={400}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Encoder Architecture</Title>
        <Text mt="md">
          The Transformer encoder consists of N identical layers stacked sequentially. For the base model:
        </Text>
        <List mt="md" spacing="sm">
          <List.Item><InlineMath>N = 6</InlineMath> layers</List.Item>
          <List.Item><InlineMath>d_{`{model}`} = 512</InlineMath></List.Item>
          <List.Item><InlineMath>d_{`{ff}`} = 2048</InlineMath></List.Item>
          <List.Item><InlineMath>h = 8</InlineMath> attention heads</List.Item>
          <List.Item><InlineMath>d_k = d_v = 64</InlineMath> (dimension per head)</List.Item>
        </List>
        <Text mt="md">
          Input processing:
        </Text>
        <CodeBlock
          language="text"
          code={`Token Embeddings: (batch, seq_len) → (batch, seq_len, d_model)
+ Positional Encoding: (seq_len, d_model)
→ Encoder Input: (batch, seq_len, d_model)

Through N layers → Encoder Output: (batch, seq_len, d_model)`}
        />
        <Text mt="md">
          Each position in the output contains contextual information from the entire input sequence.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/transformer-encoder-stack.png"
            alt="Transformer Encoder Stack Architecture"
            maw={450}
          />
        </Flex>
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
        <Text mt="md">
          Decoder block structure:
        </Text>
        <CodeBlock
          language="text"
          code={`1. Masked Self-Attention
   x = LayerNorm(x + MaskedAttn(x, x, x))

2. Cross-Attention
   x = LayerNorm(x + CrossAttn(x, enc_out, enc_out))

3. Feed-Forward
   x = LayerNorm(x + FFN(x))`}
        />
        <Text mt="md">
          The mask in step 1 ensures position <InlineMath>i</InlineMath> can only attend to positions
          <InlineMath>j \leq i</InlineMath>.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/transformer-decoder-stack.png"
            alt="Transformer Decoder Stack Architecture"
            maw={450}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Full Encoder-Decoder Architecture</Title>
        <Text mt="md">
          The complete Transformer combines encoder and decoder stacks:
        </Text>
        <CodeBlock
          language="text"
          code={`Source: [BOS, x1, x2, ..., xn]
Target: [BOS, y1, y2, ..., ym]

Encoder:
  src_emb = Embedding(source) + PosEncoding
  enc_out = EncoderStack(src_emb)  # N layers

Decoder:
  tgt_emb = Embedding(target) + PosEncoding
  dec_out = DecoderStack(tgt_emb, enc_out)  # N layers
  logits = Linear(dec_out)  # (batch, seq_len, vocab_size)`}
        />
        <Text mt="md">
          During training, teacher forcing is used: the decoder receives the true target sequence as input.
          During inference, the decoder generates one token at a time autoregressively.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/transformer_full.png"
            alt="Full Transformer Architecture"
            maw={500}
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
                            (-torch.log(torch.tensor(10000.0)) / d_model))`}
        />
        <CodeBlock
          language="python"
          code={`        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        self.register_buffer('pe', pe)`}
        />
        <CodeBlock
          language="python"
          code={`    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1)]`}
        />
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
          code={`        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
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
          For <InlineMath>d_{`{model}`} = 512</InlineMath> and <InlineMath>d_{`{ff}`} = 2048</InlineMath>,
          this layer has <InlineMath>512 \times 2048 + 2048 \times 512 = 2{`,`}097{`,`}152</InlineMath> parameters
          (excluding biases).
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
        # Self-attention with residual and norm (pre-norm)
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out`}
        />
        <CodeBlock
          language="python"
          code={`        # Feed-forward with residual and norm
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        return x`}
        />
        <Text mt="md">
          This implements pre-norm architecture. Input and output shapes are both
          <InlineMath>(batch, seq\_len, d_{`{model}`})</InlineMath>.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/transformer-block-implementation.png"
            alt="Transformer Block Implementation Details"
            maw={450}
          />
        </Flex>
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
        src_emb = self.pos_encoding(self.src_embedding(src))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))`}
        />
        <CodeBlock
          language="python"
          code={`        enc_out = self.encoder(src_emb, mask=src_mask)
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
        return self.output_projection(dec_out)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Training Details</Title>
        <Text mt="md">
          <strong>Loss function:</strong> Cross-entropy loss with label smoothing (<InlineMath>\epsilon = 0.1</InlineMath>)
        </Text>
        <BlockMath>
          {`\\mathcal{L} = -\\sum_{i=1}^{V} y_i' \\log(\\hat{y}_i), \\quad y_i' = (1-\\epsilon)y_i + \\frac{\\epsilon}{V}`}
        </BlockMath>
        <Text mt="md">
          <strong>Optimizer:</strong> Adam with <InlineMath>\beta_1 = 0.9</InlineMath>, <InlineMath>\beta_2 = 0.98</InlineMath>,
          <InlineMath>\epsilon = 10^{`{-9}`}</InlineMath>
        </Text>
        <Text mt="md">
          <strong>Learning rate schedule:</strong> Warmup followed by inverse square root decay
        </Text>
        <BlockMath>
          {`lr = d_{model}^{-0.5} \\cdot \\min(step^{-0.5}, step \\cdot warmup\\_steps^{-1.5})`}
        </BlockMath>
        <Text mt="sm">
          Typical warmup: 4000-8000 steps. This schedule increases learning rate linearly during warmup,
          then decreases proportionally to the inverse square root of step number.
        </Text>
        <Text mt="md">
          <strong>Regularization:</strong> Dropout rate 0.1 applied to attention weights, feed-forward outputs,
          and embeddings.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/learning-rate-schedule.png"
            alt="Transformer Learning Rate Schedule"
            maw={500}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Parameter Counting</Title>
        <Text mt="md">
          Detailed parameter breakdown for Transformer base model (<InlineMath>d_{`{model}`} = 512</InlineMath>,
          <InlineMath>d_{`{ff}`} = 2048</InlineMath>, <InlineMath>h = 8</InlineMath>, <InlineMath>N = 6</InlineMath>):
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
              <td><InlineMath>3 \times 512 \times 512 = 786{`,`}432</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Attention output projection</td>
              <td><InlineMath>512 \times 512 = 262{`,`}144</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Feed-forward W1</td>
              <td><InlineMath>512 \times 2048 = 1{`,`}048{`,`}576</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Feed-forward W2</td>
              <td><InlineMath>2048 \times 512 = 1{`,`}048{`,`}576</InlineMath></td>
              <td>Yes</td>
            </tr>
            <tr>
              <td>Layer norm (2 per layer)</td>
              <td><InlineMath>2 \times 2 \times 512 = 2{`,`}048</InlineMath></td>
              <td>Yes</td>
            </tr>
          </tbody>
        </Table>
        <Text mt="md">
          Total per encoder layer: <InlineMath>\approx 3.1</InlineMath> million parameters
        </Text>
        <Text>
          Total for 6 encoder layers: <InlineMath>\approx 18.6</InlineMath> million parameters
        </Text>
        <Text>
          With decoder and embeddings: <InlineMath>\approx 65</InlineMath> million parameters
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/parameter-distribution.png"
            alt="Parameter Distribution Across Transformer Layers"
            maw={500}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Computational Complexity</Title>
        <Text mt="md">
          Complexity analysis for sequence length <InlineMath>n</InlineMath> and dimension <InlineMath>d</InlineMath>:
        </Text>
        <Table mt="md">
          <thead>
            <tr>
              <th>Operation</th>
              <th>Complexity per Layer</th>
              <th>Sequential Operations</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Self-Attention</td>
              <td><InlineMath>O(n^2 \cdot d)</InlineMath></td>
              <td><InlineMath>O(1)</InlineMath></td>
            </tr>
            <tr>
              <td>Feed-Forward</td>
              <td><InlineMath>O(n \cdot d^2)</InlineMath></td>
              <td><InlineMath>O(1)</InlineMath></td>
            </tr>
            <tr>
              <td>RNN (comparison)</td>
              <td><InlineMath>O(n \cdot d^2)</InlineMath></td>
              <td><InlineMath>O(n)</InlineMath></td>
            </tr>
            <tr>
              <td>CNN (comparison)</td>
              <td><InlineMath>O(k \cdot n \cdot d^2)</InlineMath></td>
              <td><InlineMath>O(1)</InlineMath></td>
            </tr>
          </tbody>
        </Table>
        <Text mt="md">
          Memory requirements:
        </Text>
        <List mt="sm" spacing="xs">
          <List.Item>
            Attention scores: <InlineMath>O(n^2)</InlineMath> per head, <InlineMath>O(h \cdot n^2)</InlineMath> total
          </List.Item>
          <List.Item>
            Activations: <InlineMath>O(n \cdot d)</InlineMath> per layer
          </List.Item>
          <List.Item>
            For <InlineMath>n = 512</InlineMath>, attention dominates memory usage
          </List.Item>
        </List>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/computational-complexity-comparison.png"
            alt="Computational Complexity Comparison"
            maw={500}
          />
        </Flex>
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
            src="/modules/data-science-practice/module8/transformer-variants.png"
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
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/bert-architecture.png"
            alt="BERT Architecture and Training"
            maw={500}
          />
        </Flex>
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
            positions <InlineMath>j \leq i</InlineMath>
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
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/gpt-architecture.png"
            alt="GPT Architecture and Generation"
            maw={500}
          />
        </Flex>
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
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/t5-architecture.png"
            alt="T5 Text-to-Text Architecture"
            maw={500}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Architecture Comparison</Title>
        <Table mt="md">
          <thead>
            <tr>
              <th>Aspect</th>
              <th>BERT</th>
              <th>GPT</th>
              <th>T5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Architecture</td>
              <td>Encoder-only</td>
              <td>Decoder-only</td>
              <td>Encoder-Decoder</td>
            </tr>
            <tr>
              <td>Attention</td>
              <td>Bidirectional</td>
              <td>Causal (masked)</td>
              <td>Both</td>
            </tr>
            <tr>
              <td>Training Objective</td>
              <td>MLM + NSP</td>
              <td>Next token prediction</td>
              <td>Span corruption</td>
            </tr>
            <tr>
              <td>Context</td>
              <td>Full sequence</td>
              <td>Left-to-right</td>
              <td>Full (encoder)</td>
            </tr>
            <tr>
              <td>Best For</td>
              <td>Understanding tasks</td>
              <td>Generation tasks</td>
              <td>Seq2seq tasks</td>
            </tr>
            <tr>
              <td>Examples</td>
              <td>Classification, NER, QA</td>
              <td>Text completion, dialogue</td>
              <td>Translation, summarization</td>
            </tr>
            <tr>
              <td>Position Encoding</td>
              <td>Absolute learned</td>
              <td>Absolute learned</td>
              <td>Relative</td>
            </tr>
          </tbody>
        </Table>
        <Text mt="md">
          Modern trends favor decoder-only architectures (GPT-style) for their versatility and scaling properties,
          but encoder-only and encoder-decoder models remain important for specific applications.
        </Text>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/transformer-variants-comparison.png"
            alt="BERT vs GPT vs T5 Comparison"
            maw={600}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Advantages Over RNNs</Title>
        <Text mt="md">
          Transformers provide several concrete advantages over recurrent architectures:
        </Text>
        <List mt="md" spacing="md">
          <List.Item>
            <strong>Parallelization</strong>: All positions processed simultaneously during training,
            reducing training time from days to hours for large models
          </List.Item>
          <List.Item>
            <strong>Long-range dependencies</strong>: Direct connections between all positions through
            attention, path length is <InlineMath>O(1)</InlineMath> versus <InlineMath>O(n)</InlineMath> for RNNs
          </List.Item>
          <List.Item>
            <strong>Gradient flow</strong>: Residual connections provide direct gradient paths, eliminating
            vanishing gradient problems
          </List.Item>
          <List.Item>
            <strong>Interpretability</strong>: Attention weights provide insight into model decisions
          </List.Item>
          <List.Item>
            <strong>Scalability</strong>: Architecture scales efficiently to billions of parameters, enabling
            models like GPT-3 (175B) and PaLM (540B)
          </List.Item>
          <List.Item>
            <strong>Transfer learning</strong>: Pre-trained models transfer effectively to downstream tasks
            with minimal fine-tuning
          </List.Item>
        </List>
        <Flex justify="center" mt="xl">
          <Image
            src="/modules/data-science-practice/module8/transformer-vs-rnn.png"
            alt="Transformer vs RNN Comparison"
            maw={550}
          />
        </Flex>
      </div>
    </div>
  );
}
