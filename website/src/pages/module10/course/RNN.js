import React from "react";
import { Title, Text, Space, Table, Alert, List, Tabs, Grid, Card } from "@mantine/core";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";
import { IconAlertCircle } from "@tabler/icons-react";

const RNN = () => {
  return (
    <div>
      <Title order={1} mb="lg">
        Recurrent Neural Networks for NLP
      </Title>

      <Text>
        Recurrent Neural Networks (RNNs) were the dominant architecture for
        sequence modeling tasks before the rise of Transformers. This section
        covers the theory, implementation, and practical considerations of RNNs
        and their variants for NLP tasks.
      </Text>

      <Space h="md" />

      {/* Sequential Nature of Text Data */}
      <Title order={2} id="sequential-data" mb="sm">
        Sequential Nature of Text Data
      </Title>
      
      <Text>
        Text data is inherently sequential - words derive meaning from their
        context and position within a sentence. Traditional feedforward neural
        networks don't naturally handle this sequential structure as they:
      </Text>
      
      <List withPadding>
        <List.Item>Expect fixed-size inputs</List.Item>
        <List.Item>Don't share parameters across different positions</List.Item>
        <List.Item>Can't naturally model dependencies between positions</List.Item>
      </List>
      
      <Text mt="md">
        RNNs address these limitations by maintaining a hidden state that captures
        information about previous inputs, enabling them to process variable-length
        sequences and model dependencies between tokens.
      </Text>

      <Card withBorder shadow="sm" mt="md" mb="md">
        <Text fw={700}>Key Insight</Text>
        <Text>
          The core innovation of RNNs is processing tokens sequentially while maintaining 
          a state vector that carries information forward, creating an implicit "memory" 
          of the sequence seen so far.
        </Text>
      </Card>

      {/* Basic RNN Architecture */}
      <Title order={2} id="basic-rnn" mt="lg" mb="sm">
        Basic RNN Architecture
      </Title>
      
      <Text>
        The vanilla RNN processes input sequences one element at a time, updating
        its hidden state at each step.
      </Text>

      <Grid>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Title order={3} mt="md" mb="sm">
            Mathematical Formulation
          </Title>
          
          <Text>
            At each time step <InlineMath>t</InlineMath>, the RNN computes:
          </Text>
          
          <BlockMath>
            {`h_t = \\sigma(W_{xh} x_t + W_{hh} h_{t-1} + b_h)`}
          </BlockMath>
          
          <BlockMath>
            {`y_t = W_{hy} h_t + b_y`}
          </BlockMath>
          
          <Text mt="md">
            Where:
          </Text>
          
          <List withPadding>
            <List.Item>
              <InlineMath>x_t</InlineMath> is the input at time step <InlineMath>t</InlineMath>
            </List.Item>
            <List.Item>
              <InlineMath>h_t</InlineMath> is the hidden state at time step <InlineMath>t</InlineMath>
            </List.Item>
            <List.Item>
              <InlineMath>{"h_{t-1}"}</InlineMath> is the previous hidden state
            </List.Item>
            <List.Item>
              <InlineMath>{"W_{xh}"}</InlineMath> is the input-to-hidden weight matrix
            </List.Item>
            <List.Item>
              <InlineMath>{"W_{hh}"}</InlineMath> is the hidden-to-hidden weight matrix
            </List.Item>
            <List.Item>
              <InlineMath>{"W_{hy}"}</InlineMath> is the hidden-to-output weight matrix
            </List.Item>
            <List.Item>
              <InlineMath>b_h</InlineMath> and <InlineMath>b_y</InlineMath> are bias vectors
            </List.Item>
            <List.Item>
              <InlineMath>\sigma</InlineMath> is a non-linear activation function, typically tanh or ReLU
            </List.Item>
          </List>
          
          <Title order={3} mt="md" mb="sm">
            Backpropagation Through Time (BPTT)
          </Title>
          
          <Text>
            RNNs are trained using Backpropagation Through Time, which unrolls the RNN
            and treats each time step as a layer in a deep network.
          </Text>
          
          <BlockMath>
            {`\\frac{\\partial L}{\\partial W} = \\sum_{t=1}^{T} \\frac{\\partial L_t}{\\partial W}`}
          </BlockMath>
          
          <Text>
            For each parameter, gradients are accumulated across all time steps:
          </Text>
          
          <BlockMath>
            {`\\frac{\\partial L_t}{\\partial W_{hh}} = \\sum_{k=1}^{t} \\frac{\\partial L_t}{\\partial y_t} \\frac{\\partial y_t}{\\partial h_t} \\frac{\\partial h_t}{\\partial h_k} \\frac{\\partial h_k}{\\partial W_{hh}}`}
          </BlockMath>
          
          <Text>
            The chain rule term <InlineMath>{"\\frac{\\partial h_t}{\\partial h_k}"}</InlineMath> involves multiplying
            many Jacobian matrices, which can lead to vanishing or exploding gradients.
          </Text>
        </Grid.Col>
        
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Title order={3} mt="md" mb="sm">
            PyTorch Implementation
          </Title>
          
          <CodeBlock
            language="python"
            code={`import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        # Model hyperparameters
        self.hidden_size = hidden_size
        
        # RNN cell: handles input-to-hidden and hidden-to-hidden transformations
        self.rnn_cell = nn.RNNCell(
            input_size=input_size,    # Dimension of input features
            hidden_size=hidden_size,  # Dimension of hidden state
            nonlinearity='tanh'       # Activation: 'tanh' or 'relu'
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length, input_size)
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, 
                                 device=x.device)
        
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_length):
            # Update hidden state with current input
            hidden = self.rnn_cell(x[:, t, :], hidden)
            # Compute output for current step
            output = self.output_layer(hidden)
            outputs.append(output)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden`}
          />
          
          <Title order={4} mt="md" mb="sm">
            Key Hyperparameters
          </Title>
          
          <Table striped withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Parameter</Table.Th>
                <Table.Th>Description</Table.Th>
                <Table.Th>Typical Values</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              <Table.Tr>
                <Table.Td>input_size</Table.Td>
                <Table.Td>Dimension of input features</Table.Td>
                <Table.Td>Embedding dim (e.g., 300)</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>hidden_size</Table.Td>
                <Table.Td>Dimension of hidden state</Table.Td>
                <Table.Td>128-512 for NLP tasks</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>nonlinearity</Table.Td>
                <Table.Td>Activation function</Table.Td>
                <Table.Td>'tanh' (default) or 'relu'</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>batch_first</Table.Td>
                <Table.Td>Input/output tensor format</Table.Td>
                <Table.Td>True (batch, seq, features)</Table.Td>
              </Table.Tr>
            </Table.Tbody>
          </Table>
        </Grid.Col>
      </Grid>
      
      <Title order={3} mt="md" mb="sm">
        Pros and Cons of Basic RNNs
      </Title>
      
      <Grid>
        <Grid.Col span={6}>
          <Card withBorder p="xs">
            <Title order={4} mb="xs">Advantages</Title>
            <List withPadding>
              <List.Item>Can process variable-length sequences</List.Item>
              <List.Item>Parameter sharing across time steps</List.Item>
              <List.Item>Maintains sequential information</List.Item>
              <List.Item>Relatively simple architecture</List.Item>
            </List>
          </Card>
        </Grid.Col>
        <Grid.Col span={6}>
          <Card withBorder p="xs">
            <Title order={4} mb="xs">Limitations</Title>
            <List withPadding>
              <List.Item>Suffers from vanishing/exploding gradients</List.Item>
              <List.Item>Difficulty modeling long-range dependencies</List.Item>
              <List.Item>Limited memory capacity</List.Item>
              <List.Item>Slow sequential processing (not parallelizable)</List.Item>
            </List>
          </Card>
        </Grid.Col>
      </Grid>

      {/* LSTM and GRU Architectures */}
      <Title order={2} id="lstm-gru" mt="lg" mb="sm">
        LSTM and GRU Architectures
      </Title>
      
      <Text>
        Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks were
        designed to address the limitations of vanilla RNNs, particularly for modeling
        long-range dependencies.
      </Text>

      <Tabs defaultValue="lstm">
        <Tabs.List>
          <Tabs.Tab value="lstm">LSTM</Tabs.Tab>
          <Tabs.Tab value="gru">GRU</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="lstm" pt="xs">
          <Grid>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Title order={3} mt="md" mb="sm">
                LSTM Mathematical Formulation
              </Title>
              
              <Text>
                LSTMs introduce a cell state <InlineMath>C_t</InlineMath> and three gates:
                forget gate, input gate, and output gate.
              </Text>
              
              <BlockMath>{`
                f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)
              `}</BlockMath>
              <Text size="sm">Forget gate: decides what information to discard from the cell state</Text>
              
              <BlockMath>{`
                i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)
              `}</BlockMath>
              <Text size="sm">Input gate: decides which values to update</Text>
              
              <BlockMath>{`
                \\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)
              `}</BlockMath>
              <Text size="sm">Candidate cell state: new candidate values</Text>
              
              <BlockMath>{`
                C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t
              `}</BlockMath>
              <Text size="sm">Cell state update: combines old state and new candidates</Text>
              
              <BlockMath>{`
                o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)
              `}</BlockMath>
              <Text size="sm">Output gate: decides what to output based on cell state</Text>
              
              <BlockMath>{`
                h_t = o_t \\odot \\tanh(C_t)
              `}</BlockMath>
              <Text size="sm">Hidden state: filtered version of cell state</Text>
              
              <Text mt="md">
                Where <InlineMath>\sigma</InlineMath> is the sigmoid function,
                <InlineMath>\odot</InlineMath> is element-wise multiplication,
                and <InlineMath>{"[h_{t-1}, x_t]"}</InlineMath> is the concatenation
                of the previous hidden state and current input.
              </Text>
              
              <Title order={4} mt="md" mb="sm">
                LSTM Backpropagation
              </Title>
              
              <Text>
                The gradient flow in LSTMs is improved through the cell state, which
                acts as a "highway" for gradient propagation:
              </Text>
              
              <BlockMath>{`
                \\frac{\\partial C_t}{\\partial C_{t-1}} = f_t
              `}</BlockMath>
              
              <Text>
                When the forget gate <InlineMath>f_t</InlineMath> is close to 1,
                gradients can flow backwards without significant scaling, mitigating
                the vanishing gradient problem.
              </Text>
            </Grid.Col>
            
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Title order={3} mt="md" mb="sm">
                LSTM PyTorch Implementation
              </Title>
              
              <CodeBlock
                language="python"
                code={`import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, 
                 dropout=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        
        # Model hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,        # Input feature dimension
            hidden_size=hidden_size,      # Hidden state dimension
            num_layers=num_layers,        # Number of stacked LSTM layers
            batch_first=True,             # Input/output: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0, # Dropout between layers
            bidirectional=bidirectional   # Whether bidirectional
        )
        
        # Output layer with adjustment for bidirectional case
        self.output_layer = nn.Linear(
            hidden_size * self.num_directions,
            output_size
        )
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden and cell states if not provided
        if hidden is None:
            # (num_layers * num_directions, batch_size, hidden_size)
            h0 = torch.zeros(
                self.num_layers * self.num_directions,
                x.size(0),
                self.hidden_size,
                device=x.device
            )
            
            c0 = torch.zeros(
                self.num_layers * self.num_directions,
                x.size(0),
                self.hidden_size,
                device=x.device
            )
            
            hidden = (h0, c0)
        
        # Forward pass through LSTM
        # outputs: (batch_size, seq_length, hidden_size * num_directions)
        # h_n, c_n: (num_layers * num_directions, batch_size, hidden_size)
        outputs, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply output layer to all timestamps
        predictions = self.output_layer(outputs)
        
        return predictions, (h_n, c_n)`}
              />
              
              <Title order={4} mt="md" mb="sm">
                Key Hyperparameters
              </Title>
              
              <Table striped withColumnBorders>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Parameter</Table.Th>
                    <Table.Th>Description</Table.Th>
                    <Table.Th>Typical Values</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  <Table.Tr>
                    <Table.Td>hidden_size</Table.Td>
                    <Table.Td>Dimension of hidden/cell states</Table.Td>
                    <Table.Td>256-1024 for NLP tasks</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>num_layers</Table.Td>
                    <Table.Td>Number of stacked LSTM layers</Table.Td>
                    <Table.Td>1-3 typically</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>dropout</Table.Td>
                    <Table.Td>Dropout probability between layers</Table.Td>
                    <Table.Td>0.1-0.5</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>bidirectional</Table.Td>
                    <Table.Td>Process sequence in both directions</Table.Td>
                    <Table.Td>True/False</Table.Td>
                  </Table.Tr>
                </Table.Tbody>
              </Table>
            </Grid.Col>
          </Grid>
          
          <Title order={3} mt="md" mb="sm">
            Pros and Cons of LSTMs
          </Title>
          
          <Grid>
            <Grid.Col span={6}>
              <Card withBorder p="xs">
                <Title order={4} mb="xs">Advantages</Title>
                <List withPadding>
                  <List.Item>Effective at capturing long-range dependencies</List.Item>
                  <List.Item>Mitigates vanishing gradient problem through gating mechanisms</List.Item>
                  <List.Item>Separate cell state allows for better information flow</List.Item>
                  <List.Item>More stable training compared to vanilla RNNs</List.Item>
                </List>
              </Card>
            </Grid.Col>
            <Grid.Col span={6}>
              <Card withBorder p="xs">
                <Title order={4} mb="xs">Limitations</Title>
                <List withPadding>
                  <List.Item>More complex architecture with more parameters</List.Item>
                  <List.Item>Computationally expensive</List.Item>
                  <List.Item>Still processes sequences sequentially</List.Item>
                  <List.Item>Can still struggle with very long sequences (>100 tokens)</List.Item>
                </List>
              </Card>
            </Grid.Col>
          </Grid>
        </Tabs.Panel>

        <Tabs.Panel value="gru" pt="xs">
          <Grid>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Title order={3} mt="md" mb="sm">
                GRU Mathematical Formulation
              </Title>
              
              <Text>
                GRUs simplify the LSTM architecture by combining the forget gate and
                input gate into a single "update gate", and merging the cell state and
                hidden state.
              </Text>
              
              <BlockMath>{`
                z_t = \\sigma(W_z \\cdot [h_{t-1}, x_t] + b_z)
              `}</BlockMath>
              <Text size="sm">Update gate: determines how much of the previous state to keep</Text>
              
              <BlockMath>{`
                r_t = \\sigma(W_r \\cdot [h_{t-1}, x_t] + b_r)
              `}</BlockMath>
              <Text size="sm">Reset gate: determines how much of the previous state to forget</Text>
              
              <BlockMath>{`
                \\tilde{h}_t = \\tanh(W \\cdot [r_t \\odot h_{t-1}, x_t] + b)
              `}</BlockMath>
              <Text size="sm">Candidate hidden state: potential new values</Text>
              
              <BlockMath>{`
                h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t
              `}</BlockMath>
              <Text size="sm">Hidden state update: interpolation between previous state and candidate</Text>
              
              <Text mt="md">
                Where <InlineMath>\sigma</InlineMath> is the sigmoid function,
                <InlineMath>\odot</InlineMath> is element-wise multiplication,
                and <InlineMath>{"[h_{t-1}, x_t]"}</InlineMath> is the concatenation
                of the previous hidden state and current input.
              </Text>
              
              <Title order={4} mt="md" mb="sm">
                GRU Backpropagation
              </Title>
              
              <Text>
                The gradient flow in GRUs is controlled by the update gate:
              </Text>
              
              <BlockMath>{`
                \\frac{\\partial h_t}{\\partial h_{t-1}} = (1 - z_t) + z_t \\cdot \\frac{\\partial \\tilde{h}_t}{\\partial h_{t-1}}
              `}</BlockMath>
              
              <Text>
                When the update gate <InlineMath>z_t</InlineMath> is close to 0,
                the gradient flows directly through <InlineMath>(1 - z_t)</InlineMath>,
                creating a path for gradient propagation similar to LSTM's cell state.
              </Text>
            </Grid.Col>
            
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Title order={3} mt="md" mb="sm">
                GRU PyTorch Implementation
              </Title>
              
              <CodeBlock
                language="python"
                code={`import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 dropout=0.0, bidirectional=False):
        super(GRUModel, self).__init__()
        
        # Model hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,        # Input feature dimension
            hidden_size=hidden_size,      # Hidden state dimension
            num_layers=num_layers,        # Number of stacked GRU layers
            batch_first=True,             # Input/output: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0, # Dropout between layers
            bidirectional=bidirectional   # Whether bidirectional
        )
        
        # Output layer with adjustment for bidirectional case
        self.output_layer = nn.Linear(
            hidden_size * self.num_directions,
            output_size
        )
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden state if not provided
        if hidden is None:
            # (num_layers * num_directions, batch_size, hidden_size)
            hidden = torch.zeros(
                self.num_layers * self.num_directions,
                x.size(0),
                self.hidden_size,
                device=x.device
            )
        
        # Forward pass through GRU
        # outputs: (batch_size, seq_length, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        outputs, h_n = self.gru(x, hidden)
        
        # Apply output layer to all timestamps
        predictions = self.output_layer(outputs)
        
        return predictions, h_n`}
              />
              
              <Title order={4} mt="md" mb="sm">
                Key Hyperparameters
              </Title>
              
              <Table striped withColumnBorders>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Parameter</Table.Th>
                    <Table.Th>Description</Table.Th>
                    <Table.Th>Typical Values</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  <Table.Tr>
                    <Table.Td>hidden_size</Table.Td>
                    <Table.Td>Dimension of hidden state</Table.Td>
                    <Table.Td>256-1024 for NLP tasks</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>num_layers</Table.Td>
                    <Table.Td>Number of stacked GRU layers</Table.Td>
                    <Table.Td>1-3 typically</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>dropout</Table.Td>
                    <Table.Td>Dropout probability between layers</Table.Td>
                    <Table.Td>0.1-0.5</Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>bidirectional</Table.Td>
                    <Table.Td>Process sequence in both directions</Table.Td>
                    <Table.Td>True/False</Table.Td>
                  </Table.Tr>
                </Table.Tbody>
              </Table>
            </Grid.Col>
          </Grid>
          
          <Title order={3} mt="md" mb="sm">
            Pros and Cons of GRUs
          </Title>
          
          <Grid>
            <Grid.Col span={6}>
              <Card withBorder p="xs">
                <Title order={4} mb="xs">Advantages</Title>
                <List withPadding>
                  <List.Item>Simpler architecture than LSTM (fewer parameters)</List.Item>
                  <List.Item>Computationally more efficient than LSTM</List.Item>
                  <List.Item>Similar performance to LSTM on many tasks</List.Item>
                  <List.Item>Effective at capturing medium-range dependencies</List.Item>
                </List>
              </Card>
            </Grid.Col>
            <Grid.Col span={6}>
              <Card withBorder p="xs">
                <Title order={4} mb="xs">Limitations</Title>
                <List withPadding>
                  <List.Item>May perform worse than LSTM on certain tasks</List.Item>
                  <List.Item>Still sequential (not parallelizable)</List.Item>
                  <List.Item>Less expressive than LSTM in some contexts</List.Item>
                  <List.Item>Can struggle with very long sequences</List.Item>
                </List>
              </Card>
            </Grid.Col>
          </Grid>
        </Tabs.Panel>
      </Tabs>

      {/* Vanishing/Exploding Gradient Problem */}
      <Title order={2} id="vanishing-gradient" mt="lg" mb="sm">
        Vanishing/Exploding Gradient Problem
      </Title>
      
      <Text>
        One of the main challenges with training RNNs is the vanishing or exploding
        gradient problem, which is particularly severe in recurrent architectures due
        to the repeated application of the same weights.
      </Text>
      
      <Grid>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Title order={3} mt="md" mb="sm">
            Mathematical Explanation
          </Title>
          
          <Text>
            During backpropagation through time, the gradient for a particular weight
            depends on the product of gradients across multiple time steps:
          </Text>
          
          <BlockMath>{`
            \\frac{\\partial L}{\\partial W} = \\sum_{t=1}^{T} \\frac{\\partial L_t}{\\partial W}
          `}</BlockMath>
          
          <Text>
            The gradient at any time step <InlineMath>t</InlineMath> with respect to an
            earlier time step <InlineMath>k</InlineMath> involves a product of Jacobians:
          </Text>
          
          <BlockMath>{`
            \\frac{\\partial h_t}{\\partial h_k} = \\prod_{i=k+1}^{t} \\frac{\\partial h_i}{\\partial h_{i-1}}
          `}</BlockMath>
          
          <Text>
            For a vanilla RNN with <InlineMath>{"h_t = \\tanh(W h_{t-1} + U x_t)"}</InlineMath>:
          </Text>
          
          <BlockMath>{`
            \\frac{\\partial h_t}{\\partial h_{t-1}} = \\text{diag}(1 - \\tanh^2(Wh_{t-1} + Ux_t)) \\cdot W
          `}</BlockMath>
          
          <Text>
            If the eigenvalues of this Jacobian matrix are less than 1, gradients
            will vanish exponentially with sequence length. If greater than 1, they
            will explode.
          </Text>
          
          <Alert icon={<IconAlertCircle size="1rem" />} title="Key Insight" color="blue" mt="md">
            For vanilla RNNs, the maximum eigenvalue of <InlineMath>W</InlineMath> is crucial:
            <List withPadding>
              <List.Item>If <InlineMath>{"|\\lambda_{\\text{max}}(W)| < 1"}</InlineMath>: vanishing gradients</List.Item>
              <List.Item>If <InlineMath>{"|\\lambda_{\\text{max}}(W)| > 1"}</InlineMath>: exploding gradients</List.Item>
            </List>
          </Alert>
        </Grid.Col>
        
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Title order={3} mt="md" mb="sm">
            Mitigation Strategies
          </Title>
          
          <Table striped withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Problem</Table.Th>
                <Table.Th>Solution</Table.Th>
                <Table.Th>Implementation</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              <Table.Tr>
                <Table.Td>Vanishing Gradients</Table.Td>
                <Table.Td>LSTM/GRU architectures</Table.Td>
                <Table.Td>Use gating mechanisms to control gradient flow</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Vanishing Gradients</Table.Td>
                <Table.Td>Identity initialization</Table.Td>
                <Table.Td>Initialize recurrent weights close to identity matrix</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Vanishing Gradients</Table.Td>
                <Table.Td>Skip connections</Table.Td>
                <Table.Td>Add residual connections between layers</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Exploding Gradients</Table.Td>
                <Table.Td>Gradient clipping</Table.Td>
                <Table.Td>Scale gradients when norm exceeds threshold</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Both</Table.Td>
                <Table.Td>Proper initialization</Table.Td>
                <Table.Td>Use orthogonal initialization for weights</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Both</Table.Td>
                <Table.Td>Truncated BPTT</Table.Td>
                <Table.Td>Limit the number of timesteps for backpropagation</Table.Td>
              </Table.Tr>
            </Table.Tbody>
          </Table>
          
          <CodeBlock
            language="python"
            code={`# Gradient clipping example in PyTorch
import torch.nn as nn
import torch.optim as optim

# Define model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in data_loader:
        # Forward pass
        outputs, _ = model(batch_x)
        loss = criterion(outputs.view(-1, output_size), batch_y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update weights
        optimizer.step()`}
          />
        </Grid.Col>
      </Grid>

      {/* Bidirectional RNNs */}
      <Title order={2} id="bidirectional" mt="lg" mb="sm">
        Bidirectional RNNs
      </Title>
      
      <Text>
        Bidirectional RNNs process sequences from both directions, enabling the model
        to capture context from both past and future tokens, which is especially
        useful for many NLP tasks.
      </Text>
      
      <Grid>
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Title order={3} mt="md" mb="sm">
            Architecture and Mathematical Formulation
          </Title>
          
          <Text>
            A bidirectional RNN maintains two separate hidden states:
          </Text>
          
          <BlockMath>{`
            \\overrightarrow{h}_t = f(\\overrightarrow{h}_{t-1}, x_t)
          `}</BlockMath>
          <Text size="sm">Forward hidden state (processing left-to-right)</Text>
          
          <BlockMath>{`
            \\overleftarrow{h}_t = f(\\overleftarrow{h}_{t+1}, x_t)
          `}</BlockMath>
          <Text size="sm">Backward hidden state (processing right-to-left)</Text>
          
          <Text>
            The final output combines information from both directions:
          </Text>
          
          <BlockMath>{`
            y_t = g(\\overrightarrow{h}_t, \\overleftarrow{h}_t)
          `}</BlockMath>
          
          <Text>
            Where <InlineMath>g</InlineMath> is typically a concatenation followed by
            a linear transformation:
          </Text>
          
          <BlockMath>{`
            y_t = W_y [\\overrightarrow{h}_t; \\overleftarrow{h}_t] + b_y
          `}</BlockMath>
          
          <Alert icon={<IconAlertCircle size="1rem" />} title="Important Consideration" color="orange" mt="md">
            Bidirectional RNNs cannot be used for autoregressive generation tasks
            because they require access to future tokens, which are unavailable
            during generation.
          </Alert>
        </Grid.Col>
        
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Title order={3} mt="md" mb="sm">
            PyTorch Implementation
          </Title>
          
          <CodeBlock
            language="python"
            code={`import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(BidirectionalLSTM, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Enable bidirectional processing
        )
        
        # Output layer (note the *2 for bidirectional)
        self.output_layer = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Process in both directions
        # outputs: (batch_size, seq_length, hidden_size * 2)
        outputs, _ = self.lstm(x)
        
        # Apply output layer
        predictions = self.output_layer(outputs)
        
        return predictions

# Example usage for NER task
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_tags, 
                 num_layers=1, dropout=0.5):
        super(BiLSTM_NER, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = BidirectionalLSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            output_size=num_tags,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x):
        # x: (batch_size, seq_length) integer tensor
        
        # Get embeddings
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through BiLSTM
        tag_scores = self.bilstm(embedded)  # (batch_size, seq_length, num_tags)
        
        return tag_scores`}
          />
          
          <Title order={3} mt="md" mb="sm">
            Applications in NLP
          </Title>
          
          <Table striped withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Task</Table.Th>
                <Table.Th>Suitability</Table.Th>
                <Table.Th>Explanation</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              <Table.Tr>
                <Table.Td>Named Entity Recognition</Table.Td>
                <Table.Td>Excellent</Table.Td>
                <Table.Td>Entity classification benefits from both left and right context</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Part-of-Speech Tagging</Table.Td>
                <Table.Td>Excellent</Table.Td>
                <Table.Td>Word role often depends on surrounding context</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Sentiment Analysis</Table.Td>
                <Table.Td>Good</Table.Td>
                <Table.Td>Sentence-level classification benefits from full context</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Machine Translation</Table.Td>
                <Table.Td>Limited</Table.Td>
                <Table.Td>Encoder can be bidirectional, but decoder usually cannot</Table.Td>
              </Table.Tr>
              <Table.Tr>
                <Table.Td>Text Generation</Table.Td>
                <Table.Td>Poor</Table.Td>
                <Table.Td>Future tokens unavailable during autoregressive generation</Table.Td>
              </Table.Tr>
            </Table.Tbody>
          </Table>
        </Grid.Col>
      </Grid>

      {/* PyTorch Implementations */}
      <Title order={2} id="implementations" mt="lg" mb="sm">
        PyTorch Implementations
      </Title>
      
      <Text>
        Here we provide a systematic comparison of the different RNN variants in
        PyTorch with a focus on practical implementation details.
      </Text>
      
      <Title order={3} mt="md" mb="sm">
        Comprehensive Example: Sequence Classification
      </Title>
      
      <CodeBlock
        language="python"
        code={`import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define a generic RNN-based sequence classifier
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, 
                 rnn_type='lstm', num_layers=1, bidirectional=False, 
                 dropout=0.5):
        super(RNNClassifier, self).__init__()
        
        # Configuration
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("rnn_type must be one of ['rnn', 'lstm', 'gru']")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_length) integer tensor of token indices
        # lengths: (batch_size,) tensor of sequence lengths
        
        # Embed tokens
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # If we have variable length sequences, use pack_padded_sequence
        if lengths is not None:
            # Sort by length for packing
            lengths, sort_idx = lengths.sort(descending=True)
            embedded = embedded[sort_idx]
            
            # Pack sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True
            )
            
            # Process with RNN
            if self.rnn_type == 'lstm':
                packed_output, (hidden, _) = self.rnn(packed_embedded)
            else:
                packed_output, hidden = self.rnn(packed_embedded)
            
            # Unpack output
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            
            # Unsort
            _, unsort_idx = sort_idx.sort()
            output = output[unsort_idx]
            
            # Get the relevant hidden state
            if self.bidirectional:
                # For bidirectional, concatenate the last hidden state from both directions
                if self.rnn_type == 'lstm':
                    # Take the last layer's hidden state
                    hidden = hidden.view(self.num_layers, self.num_directions, 
                                        hidden.size(1), self.hidden_size)
                    last_layer_hidden = hidden[-1]
                    
                    # Concatenate forward and backward
                    h_forward = last_layer_hidden[0]
                    h_backward = last_layer_hidden[1]
                    final_hidden = torch.cat([h_forward, h_backward], dim=1)
                else:
                    # For GRU/RNN, similar procedure
                    hidden = hidden.view(self.num_layers, self.num_directions, 
                                        hidden.size(1), self.hidden_size)
                    last_layer_hidden = hidden[-1]
                    final_hidden = torch.cat([last_layer_hidden[0], last_layer_hidden[1]], dim=1)
            else:
                # For unidirectional, just take the last hidden state
                if self.rnn_type == 'lstm':
                    final_hidden = hidden[-1]
                else:
                    final_hidden = hidden[-1]
        
        else:
            # For fixed-length sequences, simpler processing
            if self.rnn_type == 'lstm':
                _, (hidden, _) = self.rnn(embedded)
                final_hidden = hidden[-1]
            else:
                _, hidden = self.rnn(embedded)
                final_hidden = hidden[-1]
            
            # For bidirectional, reshape and concatenate
            if self.bidirectional:
                final_hidden = final_hidden.view(self.num_layers, self.num_directions, 
                                               x.size(0), self.hidden_size)[-1]
                final_hidden = torch.cat([final_hidden[0], final_hidden[1]], dim=1)
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        # Apply classification layer
        output = self.fc(final_hidden)
        
        return output

# Example usage
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_lengths, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x, batch_lengths)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_lengths, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_lengths = batch_lengths.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x, batch_lengths)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100*correct/total:.2f}%')
    
    return model`}
      />
      
      <Title order={3} mt="md" mb="sm">
        Key Configuration Options
      </Title>
      
      <Table striped withColumnBorders>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Setting</Table.Th>
            <Table.Th>Options</Table.Th>
            <Table.Th>Impact</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          <Table.Tr>
            <Table.Td>batch_first</Table.Td>
            <Table.Td>True/False</Table.Td>
            <Table.Td>Input/output tensor format: (batch, seq, feature) vs (seq, batch, feature)</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>bidirectional</Table.Td>
            <Table.Td>True/False</Table.Td>
            <Table.Td>Process sequence in both directions, doubles output dimension</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>num_layers</Table.Td>
            <Table.Td>Integer ≥ 1</Table.Td>
            <Table.Td>Number of stacked RNN layers, increases capacity but risk of overfitting</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>dropout</Table.Td>
            <Table.Td>Float [0, 1]</Table.Td>
            <Table.Td>Dropout between layers (only applied if num_layers > 1)</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>proj_size</Table.Td>
            <Table.Td>Integer ≥ 0</Table.Td>
            <Table.Td>Dimension of projected hidden state (LSTM only, PyTorch 1.8+)</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>nonlinearity (RNN)</Table.Td>
            <Table.Td>'tanh'/'relu'</Table.Td>
            <Table.Td>Activation function for vanilla RNN</Table.Td>
          </Table.Tr>
        </Table.Tbody>
      </Table>

      <Title order={3} mt="md" mb="sm">
        Handling Variable-Length Sequences
      </Title>
      
      <CodeBlock
        language="python"
        code={`# Example of using PackedSequence for efficient processing
def prepare_sequence_batch(sequences, tokenizer):
    """
    Prepare a batch of variable-length sequences for RNN processing.
    Returns padded sequences and their lengths.
    """
    # Tokenize and convert to indices
    indexed_seqs = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq)) for seq in sequences]
    
    # Get sequence lengths
    seq_lengths = torch.LongTensor([len(s) for s in indexed_seqs])
    
    # Pad sequences to max length in batch
    max_len = max(seq_lengths)
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
    
    # Fill padded tensor
    for i, seq in enumerate(indexed_seqs):
        end = seq_lengths[i]
        padded_seqs[i, :end] = torch.LongTensor(seq[:end])
    
    return padded_seqs, seq_lengths

# In DataLoader:
for batch_texts, batch_labels in dataloader:
    # Prepare variable-length sequences
    padded_seqs, seq_lengths = prepare_sequence_batch(batch_texts, tokenizer)
    
    # Forward pass with lengths for proper packing
    outputs = model(padded_seqs, seq_lengths)
`}
      />

      {/* Comparison with Transformers */}
      <Title order={2} id="comparison" mt="lg" mb="sm">
        Comparison with Transformers
      </Title>
      
      <Text>
        RNNs dominated sequence modeling in NLP before the introduction of Transformers.
        Here's a comparison of their relative strengths and weaknesses.
      </Text>
      
      <Table striped withColumnBorders>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Aspect</Table.Th>
            <Table.Th>RNNs (LSTM/GRU)</Table.Th>
            <Table.Th>Transformers</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          <Table.Tr>
            <Table.Td>Parallelization</Table.Td>
            <Table.Td>Sequential processing, cannot parallelize</Table.Td>
            <Table.Td>Fully parallelizable across sequence length</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Long-range Dependencies</Table.Td>
            <Table.Td>Difficulty with very long sequences</Table.Td>
            <Table.Td>Direct connections between any positions</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Computational Complexity</Table.Td>
            <Table.Td>O(n) with sequence length</Table.Td>
            <Table.Td>O(n²) with sequence length (self-attention)</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Memory Usage</Table.Td>
            <Table.Td>Lower, state size independent of sequence length</Table.Td>
            <Table.Td>Higher, attention maps grow quadratically</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Positional Information</Table.Td>
            <Table.Td>Inherent in sequential processing</Table.Td>
            <Table.Td>Requires explicit positional encodings</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Variable Length Handling</Table.Td>
            <Table.Td>Natural, maintains state regardless of length</Table.Td>
            <Table.Td>Requires padding or truncation to fixed length</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Performance on NLP Tasks</Table.Td>
            <Table.Td>Good, but generally inferior to Transformers</Table.Td>
            <Table.Td>State-of-the-art on most NLP tasks</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Training Stability</Table.Td>
            <Table.Td>Can be unstable due to vanishing/exploding gradients</Table.Td>
            <Table.Td>Generally more stable with layer normalization</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Streaming/Online Processing</Table.Td>
            <Table.Td>Well-suited for incremental processing</Table.Td>
            <Table.Td>Challenging due to need for full context</Table.Td>
          </Table.Tr>
        </Table.Tbody>
      </Table>
      
      <Title order={3} mt="lg" mb="sm">
        When to Use RNNs vs. Transformers
      </Title>
      
      <Grid>
        <Grid.Col span={6}>
          <Card withBorder p="xs">
            <Title order={4} mb="xs">Consider RNNs When:</Title>
            <List withPadding>
              <List.Item>Working with limited computational resources</List.Item>
              <List.Item>Processing very long sequences (> 1000 tokens)</List.Item>
              <List.Item>Implementing streaming/online processing</List.Item>
              <List.Item>Model size and inference speed are critical</List.Item>
              <List.Item>The task involves explicit sequence modeling where order is critical</List.Item>
            </List>
          </Card>
        </Grid.Col>
        <Grid.Col span={6}>
          <Card withBorder p="xs">
            <Title order={4} mb="xs">Consider Transformers When:</Title>
            <List withPadding>
              <List.Item>Maximum accuracy is the primary goal</List.Item>
              <List.Item>Parallel processing is available (GPUs)</List.Item>
              <List.Item>Tasks involve complex linguistic understanding</List.Item>
              <List.Item>Training data is abundant</List.Item>
              <List.Item>Long-range dependencies are critical</List.Item>
            </List>
          </Card>
        </Grid.Col>
      </Grid>
      
      <Alert icon={<IconAlertCircle size="1rem" />} title="Hybrid Approaches" color="blue" mt="md">
        Some recent work explores combining RNNs and Transformers to get the best of both worlds:
        <List withPadding>
          <List.Item>RNN-enhanced Transformers for streaming applications</List.Item>
          <List.Item>Transformer-enhanced RNNs for improved expressivity</List.Item>
          <List.Item>Linear attention mechanisms to reduce the quadratic complexity of Transformers</List.Item>
        </List>
      </Alert>
    </div>
  );
};

export default RNN;