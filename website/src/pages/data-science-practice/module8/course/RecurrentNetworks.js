import React from "react";
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";

const RecurrentNetworks = () => {
  return (
    <>
      <div data-slide>
        <Title order={1}>Recurrent Neural Networks</Title>

        <Text mt="md">
          Recurrent Neural Networks (RNNs) are designed to process sequential data by applying
          the same parameters at each time step, similar to how CNNs share parameters across spatial locations.
        </Text>

        <Text mt="md">
          <strong>Key difference from CNNs:</strong> While CNNs apply parameters in parallel across
          spatial positions, RNNs apply parameters <strong>sequentially</strong> across time steps.
        </Text>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/rnn.png"
            alt="RNN concept: sequential parameter sharing across time steps"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Same parameters applied sequentially at each time step
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Sequential Processing</Title>

        <Text mt="md">
          At each time step t, the RNN:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Receives input <InlineMath>{'x^{(t)}'}</InlineMath></List.Item>
          <List.Item>Uses the previous hidden state <InlineMath>{'h^{(t-1)}'}</InlineMath></List.Item>
          <List.Item>Applies the <strong>same parameters</strong> to compute <InlineMath>{'h^{(t)}'}</InlineMath></List.Item>
          <List.Item>Produces output <InlineMath>{'y^{(t)}'}</InlineMath></List.Item>
        </List>

        <Text mt="lg">
          This sequential dependency means:
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Cannot process time steps in parallel</List.Item>
          <List.Item>Information flows forward through time</List.Item>
          <List.Item>Can handle variable-length sequences</List.Item>
        </List>


      </div>
<div data-slide>
        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/rnns.png"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            RNNs units
          </Text>
        </Flex>
</div>
      <div data-slide>
        <Title order={2}>Standard RNN</Title>

        <Text mt="md">
          The simplest recurrent architecture with a single hidden state update:
        </Text>

        <BlockMath>{'h^{(t)} = \\tanh(W_{xh}x^{(t)} + W_{hh}h^{(t-1)} + b)'}</BlockMath>

        <Title order={3} mt="lg">PyTorch Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch.nn as nn

# Create RNN layer
rnn = nn.RNN(input_size=300, hidden_size=128, num_layers=1, batch_first=True)`}
        />

        <CodeBlock
          language="python"
          code={`# Forward pass
x = torch.randn(32, 20, 300)  # (batch, seq_len, input_size)
output, h_n = rnn(x)
print(output.shape)  # (32, 20, 128)
print(h_n.shape)     # (1, 32, 128)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>RNN: Parameters & Properties</Title>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Property</Table.Th>
              <Table.Th>Value</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Input shape</Table.Td>
              <Table.Td>(batch_size, seq_len, input_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Output shape</Table.Td>
              <Table.Td>(batch_size, seq_len, hidden_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Hidden state shape</Table.Td>
              <Table.Td>(num_layers, batch_size, hidden_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Number of parameters</Table.Td>
              <Table.Td><InlineMath>{'d_h(d_x + d_h + 1)'}</InlineMath></Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Title order={3} mt="lg">Key Hyperparameters</Title>
        <List spacing="xs" mt="sm">
          <List.Item><strong>input_size:</strong> Dimension of input features</List.Item>
          <List.Item><strong>hidden_size:</strong> Dimension of hidden state (128-512 typical)</List.Item>
          <List.Item><strong>num_layers:</strong> Number of stacked RNN layers (1-3 typical)</List.Item>
          <List.Item><strong>batch_first:</strong> Set to True for (batch, seq, features) format</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>RNN: Pros & Cons</Title>

        <Title order={3} mt="md">Pros</Title>
        <List spacing="xs" mt="sm">
          <List.Item>Simple architecture, fast computation</List.Item>
          <List.Item>Fewer parameters than LSTM/GRU</List.Item>
          <List.Item>Works well for short sequences</List.Item>
        </List>

        <Title order={3} mt="lg">Cons</Title>
        <List spacing="xs" mt="sm">
          <List.Item>Vanishing gradient problem for long sequences</List.Item>
          <List.Item>Cannot learn long-term dependencies effectively</List.Item>
          <List.Item>Prone to exploding gradients</List.Item>
          <List.Item>Sequential processing (no parallelization)</List.Item>
        </List>

        <Text mt="lg">
          <strong>Use case:</strong> Simple tasks with short sequences where long-term memory is not critical.
        </Text>

      </div>

      <div data-slide>
        <Title order={2}>LSTM: Long Short-Term Memory</Title>

        <Text mt="md">
          LSTM addresses the vanishing gradient problem using gates and a separate cell state.
        </Text>

        <Text mt="md">
          <strong>Key innovation:</strong> A cell state <InlineMath>{'c^{(t)}'}</InlineMath> that flows
          through time with minimal transformations, preserving gradients.
        </Text>

        <Title order={3} mt="lg">PyTorch Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch.nn as nn

# Create LSTM layer
lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True)`}
        />

        <CodeBlock
          language="python"
          code={`# Forward pass
x = torch.randn(32, 20, 300)  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
print(output.shape)  # (32, 20, 128)
print(h_n.shape)     # (1, 32, 128) - hidden state
print(c_n.shape)     # (1, 32, 128) - cell state`}
        />
      </div>

      <div data-slide>
        <Title order={2}>LSTM: Parameters & Properties</Title>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Property</Table.Th>
              <Table.Th>Value</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Input shape</Table.Td>
              <Table.Td>(batch_size, seq_len, input_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Output shape</Table.Td>
              <Table.Td>(batch_size, seq_len, hidden_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Hidden state shape</Table.Td>
              <Table.Td>(num_layers, batch_size, hidden_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Cell state shape</Table.Td>
              <Table.Td>(num_layers, batch_size, hidden_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Number of parameters</Table.Td>
              <Table.Td><InlineMath>{'4 \\cdot d_h(d_x + d_h + 1)'}</InlineMath> (4 gates)</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Title order={3} mt="lg">Key Hyperparameters</Title>
        <List spacing="xs" mt="sm">
          <List.Item><strong>input_size:</strong> Dimension of input features</List.Item>
          <List.Item><strong>hidden_size:</strong> Dimension of hidden/cell state (128-512 typical)</List.Item>
          <List.Item><strong>num_layers:</strong> Number of stacked LSTM layers (1-3 typical)</List.Item>
          <List.Item><strong>dropout:</strong> Dropout between layers (0.2-0.5 for multiple layers)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>LSTM: Gates Overview</Title>

        <Text mt="md">
          LSTM uses three gates to control information flow:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Forget gate:</strong> Decides what to discard from cell state
          </List.Item>
          <List.Item>
            <strong>Input gate:</strong> Decides what new information to add
          </List.Item>
          <List.Item>
            <strong>Output gate:</strong> Decides what to output from cell state
          </List.Item>
        </List>

      </div>

      <div data-slide>
        <Title order={2}>LSTM: Pros & Cons</Title>

        <Title order={3} mt="md">Pros</Title>
        <List spacing="xs" mt="sm">
          <List.Item>Learns long-term dependencies effectively</List.Item>
          <List.Item>Stable gradients through cell state</List.Item>
          <List.Item>Most widely used RNN variant</List.Item>
          <List.Item>Well-tested across many domains</List.Item>
        </List>

        <Title order={3} mt="lg">Cons</Title>
        <List spacing="xs" mt="sm">
          <List.Item>4× more parameters than standard RNN</List.Item>
          <List.Item>Slower training and inference</List.Item>
          <List.Item>More memory required (stores cell state)</List.Item>
          <List.Item>Still sequential (no parallelization)</List.Item>
        </List>

        <Text mt="lg">
          <strong>Use case:</strong> Tasks requiring long-term memory (text generation, machine translation,
          speech recognition).
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>GRU: Gated Recurrent Unit</Title>

        <Text mt="md">
          GRU simplifies LSTM by merging cell and hidden states and using only two gates.
        </Text>

        <Text mt="md">
          <strong>Key simplification:</strong> No separate cell state, fewer parameters than LSTM.
        </Text>

        <Title order={3} mt="lg">PyTorch Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch.nn as nn

# Create GRU layer
gru = nn.GRU(input_size=300, hidden_size=128, num_layers=1, batch_first=True)`}
        />

        <CodeBlock
          language="python"
          code={`# Forward pass
x = torch.randn(32, 20, 300)  # (batch, seq_len, input_size)
output, h_n = gru(x)
print(output.shape)  # (32, 20, 128)
print(h_n.shape)     # (1, 32, 128) - only hidden state (no cell state)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>GRU: Parameters & Properties</Title>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Property</Table.Th>
              <Table.Th>Value</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Input shape</Table.Td>
              <Table.Td>(batch_size, seq_len, input_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Output shape</Table.Td>
              <Table.Td>(batch_size, seq_len, hidden_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Hidden state shape</Table.Td>
              <Table.Td>(num_layers, batch_size, hidden_size)</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Number of parameters</Table.Td>
              <Table.Td><InlineMath>{'3 \\cdot d_h(d_x + d_h + 1)'}</InlineMath> (3 gates)</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Title order={3} mt="lg">Key Hyperparameters</Title>
        <List spacing="xs" mt="sm">
          <List.Item><strong>input_size:</strong> Dimension of input features</List.Item>
          <List.Item><strong>hidden_size:</strong> Dimension of hidden state (128-512 typical)</List.Item>
          <List.Item><strong>num_layers:</strong> Number of stacked GRU layers (1-3 typical)</List.Item>
          <List.Item><strong>dropout:</strong> Dropout between layers (0.2-0.5 for multiple layers)</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>GRU: Gates Overview</Title>

        <Text mt="md">
          GRU uses two gates for simpler computation:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Reset gate:</strong> Controls how much past information to forget
          </List.Item>
          <List.Item>
            <strong>Update gate:</strong> Controls how much to update with new information
          </List.Item>
        </List>

      </div>

      <div data-slide>
        <Title order={2}>GRU: Pros & Cons</Title>

        <Title order={3} mt="md">Pros</Title>
        <List spacing="xs" mt="sm">
          <List.Item>25% fewer parameters than LSTM</List.Item>
          <List.Item>Faster training and inference than LSTM</List.Item>
          <List.Item>Performance often comparable to LSTM</List.Item>
          <List.Item>Less memory (no cell state)</List.Item>
        </List>

        <Title order={3} mt="lg">Cons</Title>
        <List spacing="xs" mt="sm">
          <List.Item>Still 3× more parameters than standard RNN</List.Item>
          <List.Item>May underperform LSTM on some tasks</List.Item>
          <List.Item>Still sequential (no parallelization)</List.Item>
        </List>

        <Text mt="lg">
          <strong>Use case:</strong> Good default choice - faster than LSTM with similar performance.
          Especially useful when computational resources are limited.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Architecture Comparison</Title>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Architecture</Table.Th>
              <Table.Th>Parameters</Table.Th>
              <Table.Th>Speed</Table.Th>
              <Table.Th>Long-term Memory</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>RNN</Table.Td>
              <Table.Td><InlineMath>{'d_h(d_x + d_h + 1)'}</InlineMath></Table.Td>
              <Table.Td>Fast</Table.Td>
              <Table.Td>Poor</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>GRU</Table.Td>
              <Table.Td><InlineMath>{'3d_h(d_x + d_h + 1)'}</InlineMath></Table.Td>
              <Table.Td>Medium</Table.Td>
              <Table.Td>Good</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>LSTM</Table.Td>
              <Table.Td><InlineMath>{'4d_h(d_x + d_h + 1)'}</InlineMath></Table.Td>
              <Table.Td>Slower</Table.Td>
              <Table.Td>Best</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Title order={3} mt="lg">Example Parameter Count</Title>
        <Text mt="sm">For input_size=300, hidden_size=128:</Text>
        <List spacing="xs" mt="sm">
          <List.Item>RNN: ~71k parameters</List.Item>
          <List.Item>GRU: ~214k parameters (3×)</List.Item>
          <List.Item>LSTM: ~285k parameters (4×)</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/rnns-diff.png"
            alt="Comparison of RNN, GRU, and LSTM architectures"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Performance vs complexity trade-offs
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Practical Example: Text Classification</Title>

        <CodeBlock
          language="python"
          code={`import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)`}
        />

        <CodeBlock
          language="python"
          code={`    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(embedded)  # h_n: (1, batch_size, hidden_dim)
        return self.fc(h_n.squeeze(0))  # (batch_size, num_classes)`}
        />

        <CodeBlock
          language="python"
          code={`# Create model
model = TextClassifier(vocab_size=10000, embed_dim=300,
                       hidden_dim=128, num_classes=2)

# Example usage
x = torch.randint(0, 10000, (32, 20))  # 32 sequences of length 20
logits = model(x)  # (32, 2)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Bidirectional RNNs</Title>
        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/birnn.png"
            alt="Bidirectional RNN concept: processing sequences in both directions"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
        </Flex>
        <Text mt="md">
          Process sequences in both forward and backward directions to capture full context.
        </Text>

        <CodeBlock
          language="python"
          code={`# Bidirectional LSTM
bi_lstm = nn.LSTM(
    input_size=300,
    hidden_size=128,
    bidirectional=True,  # Enable bidirectional
    batch_first=True
)`}
        />

        <CodeBlock
          language="python"
          code={`# Forward pass
x = torch.randn(32, 20, 300)
output, (h_n, c_n) = bi_lstm(x)

# Output concatenates forward and backward
print(output.shape)  # (32, 20, 256) - hidden_size * 2`}
        />

        <Text mt="md">
          <strong>Note:</strong> Output dimension is <InlineMath>{'2 \\times \\text{hidden\\_size}'}</InlineMath>
          due to concatenation of both directions.
        </Text>
      </div>


      <div data-slide>
        <Title order={2}>Limitations of RNNs</Title>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Sequential processing:</strong> Cannot parallelize across time steps
          </List.Item>
          <List.Item>
            <strong>Limited by hidden state size:</strong> All information must compress into fixed vector
          </List.Item>
          <List.Item>
            <strong>Computational cost:</strong> Linear in sequence length
          </List.Item>
          <List.Item>
            <strong>Still challenging for very long sequences:</strong> Even LSTM/GRU struggle with 1000+ steps
          </List.Item>
        </List>

        <Text mt="lg">
          These limitations led to the development of <strong>attention mechanisms</strong> and
          <strong>Transformers</strong>, which we'll explore next.
        </Text>


      </div>

      <div data-slide>
        <Title order={2}>Sequence-to-Sequence (Seq2Seq) Models</Title>

        <Text mt="md">
          Standard RNNs process input sequences to produce output sequences of the <strong>same length</strong>.
          Many real-world tasks require different input and output lengths:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Machine translation: English sentence → French sentence (different lengths)</List.Item>
          <List.Item>Text summarization: Long document → Short summary</List.Item>
          <List.Item>Question answering: Question → Answer (different structures)</List.Item>
          <List.Item>Speech recognition: Audio frames → Text words (different temporal scales)</List.Item>
        </List>

        <Text mt="lg">
          <strong>Problem:</strong> How do we handle sequences where input length <InlineMath>{'T_x'}</InlineMath> ≠ output length <InlineMath>{'T_y'}</InlineMath>?
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Seq2Seq Architecture</Title>

        <Text mt="md">
          The sequence-to-sequence model uses an <strong>encoder-decoder</strong> architecture:
        </Text>

        <List spacing="sm" mt="lg">
          <List.Item>
            <strong>Encoder:</strong> Processes the entire input sequence and compresses it into a fixed-size context vector
          </List.Item>
          <List.Item>
            <strong>Context vector:</strong> Captures the semantic meaning of the input (bottleneck)
          </List.Item>
          <List.Item>
            <strong>Decoder:</strong> Generates the output sequence one token at a time, conditioned on the context
          </List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/seq2seq.png"
            alt="Sequence-to-sequence encoder-decoder architecture"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Encoder compresses input into context vector, decoder generates output
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Encoder-Decoder Process</Title>

        <Title order={3} mt="md">Encoding Phase</Title>
        <Text mt="sm">
          Process input sequence <InlineMath>{'x^{(1)}, x^{(2)}, ..., x^{(T_x)}'}</InlineMath>:
        </Text>
        <BlockMath>{'h^{(t)} = \\text{LSTM}(x^{(t)}, h^{(t-1)})'}</BlockMath>
        <Text mt="sm">
          Final hidden state <InlineMath>{'h^{(T_x)}'}</InlineMath> becomes the context vector.
        </Text>

        <Title order={3} mt="lg">Decoding Phase</Title>
        <Text mt="sm">
          Generate output sequence <InlineMath>{'y^{(1)}, y^{(2)}, ..., y^{(T_y)}'}</InlineMath>:
        </Text>
        <BlockMath>{'s^{(t)} = \\text{LSTM}(y^{(t-1)}, s^{(t-1)})'}</BlockMath>
        <Text mt="sm">
          Initial decoder state <InlineMath>{'s^{(0)} = h^{(T_x)}'}</InlineMath> (context from encoder).
        </Text>

        <Text mt="lg">
          This allows <InlineMath>{'T_y'}</InlineMath> to be different from <InlineMath>{'T_x'}</InlineMath>.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>PyTorch Seq2Seq Implementation</Title>

        <CodeBlock
          language="python"
          code={`class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)`}
        />

        <CodeBlock
          language="python"
          code={`    def forward(self, src, tgt):
        # Encode: src is (batch, src_len, input_dim)
        _, (h_n, c_n) = self.encoder(src)

        # Decode: tgt is (batch, tgt_len, output_dim)
        decoder_out, _ = self.decoder(tgt, (h_n, c_n))

        # Generate predictions: (batch, tgt_len, output_dim)
        return self.fc(decoder_out)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>The Bottleneck Problem</Title>

        <Text mt="md">
          The context vector must compress <strong>all information</strong> from the input sequence
          into a fixed-size vector, regardless of input length.
        </Text>

        <Title order={3} mt="lg">Challenges</Title>
        <List spacing="sm" mt="sm">
          <List.Item>Long sequences lose information during compression</List.Item>
          <List.Item>Fixed vector size limits representational capacity</List.Item>
          <List.Item>Early input tokens may be forgotten by the end</List.Item>
          <List.Item>Performance degrades with longer sequences</List.Item>
        </List>

        <Text mt="lg">
          <strong>Solution:</strong> This bottleneck motivated the development of <strong>transformer architectures</strong>,
          which allow the decoder to access all encoder states rather than just the final one.
        </Text>
      </div>

    </>
  );
};

export default RecurrentNetworks;
