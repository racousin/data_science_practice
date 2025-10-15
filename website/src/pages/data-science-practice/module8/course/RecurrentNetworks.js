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
          Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state
          that captures information about previous elements in the sequence.
        </Text>

        <Text mt="md">
          Unlike feedforward networks, RNNs share parameters across different time steps,
          allowing them to process variable-length sequences efficiently.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>RNN Architecture Overview</Title>

        <Text mt="md">
          Three main RNN architectures have been developed to address different challenges:
        </Text>

        <List spacing="sm" mt="lg">
          <List.Item><strong>Standard RNN:</strong> Simple recurrent cells with a single hidden state</List.Item>
          <List.Item><strong>LSTM (Long Short-Term Memory):</strong> Complex units with gates to control information flow</List.Item>
          <List.Item><strong>GRU (Gated Recurrent Unit):</strong> Simplified gating mechanism compared to LSTM</List.Item>
        </List>

        <Flex direction="column" align="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/rnn-architecture-comparison.png"
            alt="Comparison of RNN, LSTM, and GRU architectures"
            style={{ maxWidth: 'min(700px, 70vw)', height: 'auto' }}
            fluid
            mb="sm"
          />
          <Text size="sm">
            Comparison of RNN, LSTM, and GRU cell architectures
          </Text>
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>RNN Mathematical Framework</Title>

        <Title order={3} mt="md">Function Signature</Title>
        <Text mt="sm">
          An RNN maps an input sequence and initial hidden state to an output sequence and final hidden state:
        </Text>

        <BlockMath>{`
          \\text{RNN}: \\mathbb{R}^{T \\times d_x} \\times \\mathbb{R}^{d_h} \\rightarrow \\mathbb{R}^{T \\times d_y} \\times \\mathbb{R}^{d_h}
        `}</BlockMath>

        <Text mt="lg">
          <strong>Input:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Sequence: <InlineMath>{`\\{x^{(1)}, x^{(2)}, \\ldots, x^{(T)}\\}`}</InlineMath> where each <InlineMath>{`x^{(t)} \\in \\mathbb{R}^{d_x}`}</InlineMath></List.Item>
          <List.Item>Initial hidden state: <InlineMath>{`h^{(0)} \\in \\mathbb{R}^{d_h}`}</InlineMath></List.Item>
        </List>

        <Text mt="lg">
          <strong>Output:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>Output sequence: <InlineMath>{`\\{y^{(1)}, y^{(2)}, \\ldots, y^{(T)}\\}`}</InlineMath> where each <InlineMath>{`y^{(t)} \\in \\mathbb{R}^{d_y}`}</InlineMath></List.Item>
          <List.Item>Final hidden state: <InlineMath>{`h^{(T)} \\in \\mathbb{R}^{d_h}`}</InlineMath></List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Standard RNN Cell</Title>

        <Title order={3} mt="md">Forward Pass</Title>

        <BlockMath>{`h^{(t)} = \\tanh(W_{xh}x^{(t)} + W_{hh}h^{(t-1)} + b_h)`}</BlockMath>
        <BlockMath>{`y^{(t)} = W_{hy}h^{(t)} + b_y`}</BlockMath>

        <Text mt="lg">
          <strong>Parameters:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`W_{xh} \\in \\mathbb{R}^{d_h \\times d_x}`}</InlineMath>: Input-to-hidden weights</List.Item>
          <List.Item><InlineMath>{`W_{hh} \\in \\mathbb{R}^{d_h \\times d_h}`}</InlineMath>: Hidden-to-hidden weights</List.Item>
          <List.Item><InlineMath>{`W_{hy} \\in \\mathbb{R}^{d_y \\times d_h}`}</InlineMath>: Hidden-to-output weights</List.Item>
          <List.Item><InlineMath>{`b_h \\in \\mathbb{R}^{d_h}`}</InlineMath>, <InlineMath>{`b_y \\in \\mathbb{R}^{d_y}`}</InlineMath>: Bias vectors</List.Item>
        </List>

        <Text mt="lg">
          <strong>Total parameters:</strong> <InlineMath>{`d_h(d_x + d_h + 1) + d_y(d_h + 1)`}</InlineMath>
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Standard RNN Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Parameters
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape

        if h_0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_0`}
        />

        <CodeBlock
          language="python"
          code={`        outputs = []
        for t in range(seq_len):
            # h^(t) = tanh(W_xh @ x^(t) + W_hh @ h^(t-1) + b_h)
            h = torch.tanh(self.W_xh(x[:, t, :]) + self.W_hh(h))

            # y^(t) = W_hy @ h^(t) + b_y
            y = self.W_hy(h)
            outputs.append(y)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        return outputs, h  # Return outputs and final hidden state`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Standard RNN Usage Example</Title>

        <CodeBlock
          language="python"
          code={`# Example: Sentiment classification
input_size = 300   # Word embedding dimension
hidden_size = 128  # Hidden state dimension
output_size = 2    # Binary classification (negative/positive)

model = SimpleRNN(input_size, hidden_size, output_size)

# Input: batch of embedded sequences
batch_size, seq_len = 32, 20
x = torch.randn(batch_size, seq_len, input_size)

# Forward pass
outputs, final_hidden = model(x)

print(f"Outputs shape: {outputs.shape}")  # (32, 20, 2)
print(f"Final hidden state: {final_hidden.shape}")  # (32, 128)`}
        />

        <CodeBlock
          language="python"
          code={`# For sequence classification, typically use only final output
logits = outputs[:, -1, :]  # Take last time step: (32, 2)
predictions = torch.argmax(logits, dim=1)  # (32,)

# Calculate number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: Total parameters: 56,834`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Vanishing and Exploding Gradients</Title>

        <Text mt="md">
          Standard RNNs suffer from gradient flow problems during backpropagation through time (BPTT).
        </Text>

        <Title order={3} mt="lg">Gradient Chain</Title>
        <Text mt="sm">
          The gradient from time step t to time step t-k involves a product of Jacobian matrices:
        </Text>

        <BlockMath>{`
          \\frac{\\partial h^{(t)}}{\\partial h^{(t-k)}} = \\prod_{i=t-k+1}^{t} \\frac{\\partial h^{(i)}}{\\partial h^{(i-1)}} = \\prod_{i=t-k+1}^{t} W_{hh}^T \\text{diag}(1 - (h^{(i)})^2)
        `}</BlockMath>

        <Text mt="lg">
          When <InlineMath>{`||W_{hh}|| > 1`}</InlineMath>: Gradients explode (grow exponentially)
        </Text>
        <Text mt="sm">
          When <InlineMath>{`||W_{hh}|| < 1`}</InlineMath>: Gradients vanish (shrink exponentially)
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>LSTM: Long Short-Term Memory</Title>

        <Text mt="md">
          LSTM addresses vanishing gradients through a gating mechanism and a separate cell state.
        </Text>

        <Title order={3} mt="lg">Forward Pass</Title>

        <BlockMath>{`f^{(t)} = \\sigma(W_{xf}x^{(t)} + W_{hf}h^{(t-1)} + b_f)`}</BlockMath>
        <BlockMath>{`i^{(t)} = \\sigma(W_{xi}x^{(t)} + W_{hi}h^{(t-1)} + b_i)`}</BlockMath>
        <BlockMath>{`\\tilde{c}^{(t)} = \\tanh(W_{xc}x^{(t)} + W_{hc}h^{(t-1)} + b_c)`}</BlockMath>
        <BlockMath>{`c^{(t)} = f^{(t)} \\odot c^{(t-1)} + i^{(t)} \\odot \\tilde{c}^{(t)}`}</BlockMath>
        <BlockMath>{`o^{(t)} = \\sigma(W_{xo}x^{(t)} + W_{ho}h^{(t-1)} + b_o)`}</BlockMath>
        <BlockMath>{`h^{(t)} = o^{(t)} \\odot \\tanh(c^{(t)})`}</BlockMath>
      </div>

      <div data-slide>
        <Title order={2}>LSTM Gates Explained</Title>

        <Text mt="md"><strong>Forget Gate</strong> <InlineMath>{`f^{(t)} \\in (0, 1)^{d_h}`}</InlineMath></Text>
        <Text size="sm">Controls what information to discard from the cell state</Text>

        <Text mt="md"><strong>Input Gate</strong> <InlineMath>{`i^{(t)} \\in (0, 1)^{d_h}`}</InlineMath></Text>
        <Text size="sm">Controls what new information to store in the cell state</Text>

        <Text mt="md"><strong>Cell Candidate</strong> <InlineMath>{`\\tilde{c}^{(t)} \\in (-1, 1)^{d_h}`}</InlineMath></Text>
        <Text size="sm">New candidate values that could be added to the cell state</Text>

        <Text mt="md"><strong>Output Gate</strong> <InlineMath>{`o^{(t)} \\in (0, 1)^{d_h}`}</InlineMath></Text>
        <Text size="sm">Controls what information to output based on the cell state</Text>

        <Text mt="lg">
          <strong>Key insight:</strong> The cell state <InlineMath>{`c^{(t)}`}</InlineMath> creates a "highway" for gradient flow,
          as <InlineMath>{`\\frac{\\partial c^{(t)}}{\\partial c^{(t-1)}} = f^{(t)}`}</InlineMath>
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>LSTM Parameters and Complexity</Title>

        <Text mt="md">
          <strong>Parameters per gate:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item><InlineMath>{`W_{x*} \\in \\mathbb{R}^{d_h \\times d_x}`}</InlineMath>: Input weights (4 gates)</List.Item>
          <List.Item><InlineMath>{`W_{h*} \\in \\mathbb{R}^{d_h \\times d_h}`}</InlineMath>: Hidden weights (4 gates)</List.Item>
          <List.Item><InlineMath>{`b_* \\in \\mathbb{R}^{d_h}`}</InlineMath>: Biases (4 gates)</List.Item>
        </List>

        <Text mt="lg">
          <strong>Total parameters:</strong> <InlineMath>{`4 \\cdot d_h(d_x + d_h + 1) + d_y(d_h + 1)`}</InlineMath>
        </Text>

        <Text mt="lg">
          <strong>Computational complexity per time step:</strong> <InlineMath>{`O(4d_h(d_x + d_h))`}</InlineMath>
        </Text>

        <Text mt="lg">
          <strong>Memory:</strong> Stores both cell state <InlineMath>{`c^{(t)}`}</InlineMath> and hidden state <InlineMath>{`h^{(t)}`}</InlineMath>
        </Text>

        <Text mt="lg">
          <strong>Parallelization:</strong> Cannot parallelize across time steps (sequential dependency)
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) - https://www.bioinf.jku.at/publications/older/2604.pdf
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>LSTM Implementation</Title>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Combine all 4 gates for efficiency
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, states=None):
        # x: (batch_size, input_size)
        batch_size = x.shape[0]

        if states is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = states`}
        />

        <CodeBlock
          language="python"
          code={`        # Compute all gates in parallel
        gates = self.W_ih(x) + self.W_hh(h)  # (batch_size, 4*hidden_size)

        # Split into 4 gates
        i, f, g, o = gates.chunk(4, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate

        # Update cell state and hidden state
        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, (h, c)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>LSTM Usage Example</Title>

        <CodeBlock
          language="python"
          code={`# PyTorch built-in LSTM
import torch.nn as nn

input_size = 300
hidden_size = 128
num_layers = 2
batch_size, seq_len = 32, 20

# Create LSTM
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# Input
x = torch.randn(batch_size, seq_len, input_size)

# Forward pass
output, (h_n, c_n) = lstm(x)`}
        />

        <CodeBlock
          language="python"
          code={`print(f"Output shape: {output.shape}")  # (32, 20, 128)
print(f"Final hidden state: {h_n.shape}")  # (2, 32, 128) - 2 layers
print(f"Final cell state: {c_n.shape}")    # (2, 32, 128)

# Calculate parameters
total_params = sum(p.numel() for p in lstm.parameters())
print(f"Total parameters: {total_params:,}")
# For 2-layer LSTM: ~394k parameters
# Layer 1: 4 * 128 * (300 + 128 + 1) = 219,648
# Layer 2: 4 * 128 * (128 + 128 + 1) = 131,584
# Total: 351,232`}
        />
      </div>

      <div data-slide>
        <Title order={2}>GRU: Gated Recurrent Unit</Title>

        <Text mt="md">
          GRU simplifies LSTM by combining the forget and input gates into a single update gate,
          and merging the cell state and hidden state.
        </Text>

        <Title order={3} mt="lg">Forward Pass</Title>

        <BlockMath>{`z^{(t)} = \\sigma(W_{xz}x^{(t)} + W_{hz}h^{(t-1)} + b_z)`}</BlockMath>
        <BlockMath>{`r^{(t)} = \\sigma(W_{xr}x^{(t)} + W_{hr}h^{(t-1)} + b_r)`}</BlockMath>
        <BlockMath>{`\\tilde{h}^{(t)} = \\tanh(W_{xh}x^{(t)} + W_{hh}(r^{(t)} \\odot h^{(t-1)}) + b_h)`}</BlockMath>
        <BlockMath>{`h^{(t)} = (1 - z^{(t)}) \\odot h^{(t-1)} + z^{(t)} \\odot \\tilde{h}^{(t)}`}</BlockMath>

        <Text mt="lg">
          <strong>Reset Gate</strong> <InlineMath>{`r^{(t)}`}</InlineMath>: Controls how much past information to forget
        </Text>
        <Text mt="sm">
          <strong>Update Gate</strong> <InlineMath>{`z^{(t)}`}</InlineMath>: Controls how much to update with new candidate
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>GRU Parameters and Complexity</Title>

        <Text mt="md">
          <strong>Total parameters:</strong> <InlineMath>{`3 \\cdot d_h(d_x + d_h + 1) + d_y(d_h + 1)`}</InlineMath>
        </Text>

        <Text mt="lg">
          <strong>Computational complexity per time step:</strong> <InlineMath>{`O(3d_h(d_x + d_h))`}</InlineMath>
        </Text>

        <Text mt="lg">
          <strong>Comparison with LSTM:</strong>
        </Text>
        <List spacing="xs" mt="sm">
          <List.Item>~25% fewer parameters than LSTM</List.Item>
          <List.Item>Faster to train and run</List.Item>
          <List.Item>Performance often comparable to LSTM</List.Item>
          <List.Item>No separate cell state to maintain</List.Item>
        </List>

        <Text mt="lg">
          <strong>Parallelization:</strong> Still sequential (cannot parallelize across time)
        </Text>

        <Text mt="sm" size="sm" fs="italic">
          Reference: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014) - https://arxiv.org/abs/1406.1078
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>RNN Training: Backpropagation Through Time</Title>

        <Text mt="md">
          RNNs are trained using Backpropagation Through Time (BPTT), which unfolds the network
          through the temporal dimension.
        </Text>

        <Title order={3} mt="lg">Key Steps</Title>
        <List ordered spacing="sm" mt="sm">
          <List.Item>Forward propagation for all time steps t = 1 to T</List.Item>
          <List.Item>Compute loss L (typically averaged over time steps)</List.Item>
          <List.Item>Backward propagation from t = T to t = 1</List.Item>
          <List.Item>Accumulate gradients across time steps (parameters are shared)</List.Item>
          <List.Item>Update parameters using accumulated gradients</List.Item>
        </List>

        <Text mt="lg">
          <strong>Loss function:</strong>
        </Text>
        <BlockMath>{`L = \\frac{1}{T}\\sum_{t=1}^{T}L^{(t)} = \\frac{1}{T}\\sum_{t=1}^{T} \\text{CrossEntropy}(y^{(t)}, \\hat{y}^{(t)})`}</BlockMath>
      </div>

      <div data-slide>
        <Title order={2}>Truncated BPTT</Title>

        <Text mt="md">
          For very long sequences, computing gradients through all time steps is computationally expensive
          and memory-intensive.
        </Text>

        <Text mt="lg">
          <strong>Solution:</strong> Truncate backpropagation to k time steps
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Forward pass through entire sequence</List.Item>
          <List.Item>Backward pass only through last k steps</List.Item>
          <List.Item>Maintain hidden state across chunks</List.Item>
          <List.Item>Typical k values: 20-50 time steps</List.Item>
        </List>

        <Text mt="lg">
          <strong>Trade-off:</strong> Reduced memory and computation, but gradients don't flow as far back
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>RNN Training Example</Title>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn as nn
import torch.optim as optim

# Model setup
vocab_size = 10000
embed_dim = 300
hidden_dim = 128
num_classes = 2

model = nn.Sequential(
    nn.Embedding(vocab_size, embed_dim),
    nn.LSTM(embed_dim, hidden_dim, batch_first=True),
)
classifier = nn.Linear(hidden_dim, num_classes)

optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()`}
        />

        <CodeBlock
          language="python"
          code={`# Training loop
for epoch in range(10):
    for batch_tokens, batch_labels in dataloader:
        # batch_tokens: (batch_size, seq_len)
        # batch_labels: (batch_size,)

        optimizer.zero_grad()

        # Forward pass
        embeddings = model[0](batch_tokens)  # (batch_size, seq_len, embed_dim)
        lstm_output, (h_n, _) = model[1](embeddings)

        # Use final hidden state for classification
        logits = classifier(h_n[-1])  # (batch_size, num_classes)

        # Compute loss and backward
        loss = criterion(logits, batch_labels)
        loss.backward()

        # Gradient clipping (important for RNNs!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Gradient Clipping</Title>

        <Text mt="md">
          Gradient clipping prevents exploding gradients by limiting the magnitude of gradient updates.
        </Text>

        <Title order={3} mt="lg">Norm-based Clipping</Title>
        <BlockMath>{`
          \\mathbf{g} \\leftarrow \\begin{cases}
          \\mathbf{g} & \\text{if } ||\\mathbf{g}|| \\leq \\theta \\\\
          \\theta \\cdot \\frac{\\mathbf{g}}{||\\mathbf{g}||} & \\text{otherwise}
          \\end{cases}
        `}</BlockMath>

        <Text mt="lg">
          Where <InlineMath>{`\\theta`}</InlineMath> is the clipping threshold (typically 1.0 or 5.0).
        </Text>

        <CodeBlock
          language="python"
          code={`# Apply gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Or clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>RNN Architecture Comparison</Title>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Architecture</Table.Th>
              <Table.Th>Parameters</Table.Th>
              <Table.Th>Complexity</Table.Th>
              <Table.Th>Strengths</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Standard RNN</Table.Td>
              <Table.Td><InlineMath>{`d_h(d_x + d_h + 1)`}</InlineMath></Table.Td>
              <Table.Td><InlineMath>{`O(d_h(d_x + d_h))`}</InlineMath></Table.Td>
              <Table.Td>Simple, fast</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>LSTM</Table.Td>
              <Table.Td><InlineMath>{`4d_h(d_x + d_h + 1)`}</InlineMath></Table.Td>
              <Table.Td><InlineMath>{`O(4d_h(d_x + d_h))`}</InlineMath></Table.Td>
              <Table.Td>Long-term memory, stable gradients</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>GRU</Table.Td>
              <Table.Td><InlineMath>{`3d_h(d_x + d_h + 1)`}</InlineMath></Table.Td>
              <Table.Td><InlineMath>{`O(3d_h(d_x + d_h))`}</InlineMath></Table.Td>
              <Table.Td>Efficient, good performance</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Text mt="lg">
          <strong>Typical hidden dimension:</strong> 128-512 for standard tasks, 1024-2048 for large models
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Bidirectional RNNs</Title>

        <Text mt="md">
          Bidirectional RNNs process sequences in both forward and backward directions,
          capturing context from both past and future.
        </Text>

        <BlockMath>{`
          \\overrightarrow{h}^{(t)} = \\text{RNN}_{forward}(x^{(t)}, \\overrightarrow{h}^{(t-1)})
        `}</BlockMath>
        <BlockMath>{`
          \\overleftarrow{h}^{(t)} = \\text{RNN}_{backward}(x^{(t)}, \\overleftarrow{h}^{(t+1)})
        `}</BlockMath>
        <BlockMath>{`
          h^{(t)} = [\\overrightarrow{h}^{(t)}; \\overleftarrow{h}^{(t)}]
        `}</BlockMath>

        <Text mt="lg">
          <strong>Output dimension:</strong> <InlineMath>{`2d_h`}</InlineMath> (concatenation of forward and backward states)
        </Text>
        <Text mt="sm">
          <strong>Parameters:</strong> 2× compared to unidirectional RNN
        </Text>
        <Text mt="sm">
          <strong>Use case:</strong> Tasks where future context is available (e.g., text classification, NER)
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Bidirectional LSTM Example</Title>

        <CodeBlock
          language="python"
          code={`import torch.nn as nn

# Bidirectional LSTM
input_size = 300
hidden_size = 128

bi_lstm = nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    batch_first=True,
    bidirectional=True  # Enable bidirectional processing
)

# Input
batch_size, seq_len = 32, 20
x = torch.randn(batch_size, seq_len, input_size)

# Forward pass
output, (h_n, c_n) = bi_lstm(x)`}
        />

        <CodeBlock
          language="python"
          code={`print(f"Output shape: {output.shape}")  # (32, 20, 256)
# Note: hidden_size is doubled (128 * 2 = 256)

print(f"Final hidden state: {h_n.shape}")  # (2, 32, 128)
# h_n[0]: forward direction
# h_n[1]: backward direction

# For classification, concatenate both directions
final_state = torch.cat([h_n[0], h_n[1]], dim=1)  # (32, 256)
print(f"Combined final state: {final_state.shape}")`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Limitations of RNNs</Title>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Sequential computation:</strong> Cannot parallelize across time steps,
            limiting training speed on modern hardware
          </List.Item>
          <List.Item>
            <strong>Long-range dependencies:</strong> Despite LSTM/GRU improvements, very long
            sequences still pose challenges
          </List.Item>
          <List.Item>
            <strong>Fixed hidden state:</strong> All information must be compressed into a
            fixed-size hidden state
          </List.Item>
          <List.Item>
            <strong>Computational cost:</strong> Linear in sequence length, expensive for long documents
          </List.Item>
        </List>

        <Text mt="lg">
          These limitations motivated the development of attention mechanisms and Transformer architectures,
          which we'll explore in the next sections.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Summary</Title>

        <List spacing="sm" mt="md">
          <List.Item>
            RNNs process sequences by maintaining hidden states, with shared parameters across time steps
          </List.Item>
          <List.Item>
            LSTM and GRU architectures use gating mechanisms to enable learning of long-term dependencies
          </List.Item>
          <List.Item>
            Training uses BPTT with gradient clipping to prevent exploding gradients
          </List.Item>
          <List.Item>
            RNNs have <InlineMath>{`O(Td_h(d_x + d_h))`}</InlineMath> complexity, cannot parallelize across time
          </List.Item>
          <List.Item>
            Bidirectional RNNs capture context from both directions at cost of 2× parameters
          </List.Item>
        </List>
      </div>
    </>
  );
};

export default RecurrentNetworks;
