import React from 'react';
import { Container, Title, Text, Stack, Grid, Paper, Code, List } from '@mantine/core';

const RecurrentNetworks = () => {
  return (
    <Container size="xl" className="py-6">
      <Stack spacing="xl">
        
        {/* Slide 1: Title and Introduction */}
        <div data-slide className="min-h-[500px] flex flex-col justify-center">
          <Title order={1} className="text-center mb-8">
            Recurrent Neural Networks
          </Title>
          <Text size="xl" className="text-center mb-6">
            Sequential Data Processing and Memory
          </Text>
          <div className="max-w-3xl mx-auto">
            <Paper className="p-6 bg-blue-50">
              <Text size="lg" className="mb-4">
                Recurrent Neural Networks (RNNs) are designed to handle sequential data by maintaining 
                hidden states that capture information from previous time steps. They excel at tasks 
                involving temporal dependencies and variable-length sequences.
              </Text>
              <List>
                <List.Item>Basic RNN architecture and hidden states</List.Item>
                <List.Item>LSTM and GRU for long-term dependencies</List.Item>
                <List.Item>Bidirectional and multi-layer RNNs</List.Item>
                <List.Item>Applications in NLP and time series</List.Item>
              </List>
            </Paper>
          </div>
        </div>

        {/* Slide 2: Basic RNN */}
        <div data-slide className="min-h-[500px]" id="basic-rnn">
          <Title order={2} className="mb-6">Basic RNN Architecture</Title>
          
          <Paper className="p-6 bg-gray-50 mb-6">
            <Text size="lg">
              A basic RNN processes sequences by maintaining a hidden state that is updated at each time step.
              The hidden state serves as memory, allowing the network to use information from previous inputs.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Vanilla RNN Implementation</Title>
                <Code block language="python">{`import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input to hidden weights
        self.W_ih = nn.Linear(input_size, hidden_size)
        # Hidden to hidden weights
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        # Hidden to output weights
        self.W_ho = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        hidden_states = []
        
        for t in range(seq_len):
            # h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
            h_t = torch.tanh(
                self.W_ih(x[:, t, :]) + self.W_hh(h_prev)
            )
            # Output at time t
            output_t = self.W_ho(h_t)
            
            outputs.append(output_t)
            hidden_states.append(h_t)
            h_prev = h_t
        
        return torch.stack(outputs, dim=1), torch.stack(hidden_states, dim=1)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">PyTorch RNN Layers</Title>
                <Code block language="python">{`# Using PyTorch's built-in RNN layers
rnn_layer = nn.RNN(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    dropout=0.2,
    bidirectional=False
)

# LSTM layer
lstm_layer = nn.LSTM(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    dropout=0.2,
    bidirectional=True
)

# GRU layer
gru_layer = nn.GRU(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

# Example usage
batch_size, seq_len, input_size = 32, 50, 100
x = torch.randn(batch_size, seq_len, input_size)

# RNN forward pass
output, hidden = rnn_layer(x)
print(f"RNN Output shape: {output.shape}")  # (32, 50, 128)

# LSTM forward pass
output, (hidden, cell) = lstm_layer(x)
print(f"LSTM Output shape: {output.shape}")  # (32, 50, 256) bidirectional`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 3: LSTM Networks */}
        <div data-slide className="min-h-[500px]" id="lstm-networks">
          <Title order={2} className="mb-6">LSTM (Long Short-Term Memory)</Title>
          
          <Paper className="p-6 bg-purple-50 mb-6">
            <Text size="lg">
              LSTMs solve the vanishing gradient problem of vanilla RNNs through gating mechanisms 
              that control information flow. They maintain both hidden states and cell states for better long-term memory.
            </Text>
          </Paper>
          
          <Grid gutter="lg">
            <Grid.Col span={12}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">LSTM Cell Implementation</Title>
                <Code block language="python">{`class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined linear transformation for all gates
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, input_t, hidden_t):
        h_prev, c_prev = hidden_t
        
        # Combined computation for efficiency
        gi = self.weight_ih(input_t)
        gh = self.weight_hh(h_prev)
        i_t, f_t, g_t, o_t = (gi + gh).chunk(4, 1)
        
        # Apply activation functions
        i_t = torch.sigmoid(i_t)  # Input gate
        f_t = torch.sigmoid(f_t)  # Forget gate
        g_t = torch.tanh(g_t)     # New candidate values
        o_t = torch.sigmoid(o_t)  # Output gate
        
        # Update cell state
        c_t = f_t * c_prev + i_t * g_t
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden and cell states
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        
        outputs = []
        
        for t in range(seq_len):
            input_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](
                    input_t if layer == 0 else h[layer-1], 
                    (h[layer], c[layer])
                )
            
            outputs.append(h[-1])  # Output from last layer
        
        return torch.stack(outputs, dim=1), (torch.stack(h), torch.stack(c))`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 4: GRU Networks */}
        <div data-slide className="min-h-[500px]" id="gru-networks">
          <Title order={2} className="mb-6">GRU (Gated Recurrent Unit)</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} className="mb-3">GRU Cell Implementation</Title>
                <Code block language="python">{`class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset and update gates
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size)
        
    def forward(self, input_t, hidden_t):
        gi = self.weight_ih(input_t)
        gh = self.weight_hh(hidden_t)
        
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        
        # Reset and update gates
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        
        # New gate with reset applied
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        
        # Update hidden state
        h_t = (1 - update_gate) * new_gate + update_gate * hidden_t
        
        return h_t`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-orange-50">
                <Title order={4} className="mb-3">Comparison: LSTM vs GRU</Title>
                <Code block language="python">{`# LSTM vs GRU comparison
class RNNComparison:
    def __init__(self, input_size, hidden_size):
        # LSTM: 4 gates (input, forget, output, candidate)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # GRU: 3 gates (reset, update, new)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
    
    def parameter_count(self):
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        gru_params = sum(p.numel() for p in self.gru.parameters())
        
        return {
            'LSTM': lstm_params,
            'GRU': gru_params,
            'GRU/LSTM ratio': gru_params / lstm_params
        }

# Example usage
comp = RNNComparison(100, 128)
print(comp.parameter_count())
# GRU typically has ~75% of LSTM parameters`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-indigo-50 mt-4">
            <Title order={4} className="mb-3">When to Use LSTM vs GRU</Title>
            <Grid gutter="lg">
              <Grid.Col span={6}>
                <Text className="font-semibold mb-2">Use LSTM when:</Text>
                <List>
                  <List.Item>You need strong long-term memory</List.Item>
                  <List.Item>Working with very long sequences</List.Item>
                  <List.Item>Complex temporal patterns are present</List.Item>
                  <List.Item>You have sufficient computational resources</List.Item>
                </List>
              </Grid.Col>
              <Grid.Col span={6}>
                <Text className="font-semibold mb-2">Use GRU when:</Text>
                <List>
                  <List.Item>You need faster training and inference</List.Item>
                  <List.Item>Working with shorter sequences</List.Item>
                  <List.Item>Computational resources are limited</List.Item>
                  <List.Item>Simpler patterns in the data</List.Item>
                </List>
              </Grid.Col>
            </Grid>
          </Paper>
        </div>

        {/* Slide 5: Advanced RNN Techniques */}
        <div data-slide className="min-h-[500px]" id="advanced-techniques">
          <Title order={2} className="mb-6">Advanced RNN Techniques</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Bidirectional RNNs</Title>
                <Code block language="python">{`class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, 
            bidirectional=True, batch_first=True
        )
        # Output layer (hidden_size * 2 due to bidirectionality)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * 2)
        
        # Can use last output or all outputs
        output = self.fc(lstm_out[:, -1, :])  # Last time step
        return output

# Multi-layer bidirectional
multi_bilstm = nn.LSTM(
    input_size=100,
    hidden_size=128,
    num_layers=3,
    bidirectional=True,
    dropout=0.3,
    batch_first=True
)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-blue-50">
                <Title order={4} className="mb-3">Attention Mechanism for RNNs</Title>
                <Code block language="python">{`class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Get all hidden states
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Calculate attention weights
        attention_weights = F.softmax(
            self.attention(lstm_out), dim=1
        )  # (batch, seq_len, 1)
        
        # Apply attention to get context vector
        context = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # (batch, hidden_size)
        
        # Final output
        output = self.fc(context)
        return output, attention_weights

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        # query: (batch, hidden_size)
        # keys: (batch, seq_len, hidden_size)
        
        query_expanded = query.unsqueeze(1)  # (batch, 1, hidden_size)
        scores = self.v(torch.tanh(
            self.W(keys) + self.U(query_expanded)
        ))  # (batch, seq_len, 1)
        
        attention_weights = F.softmax(scores, dim=1)
        context = torch.sum(attention_weights * keys, dim=1)
        
        return context, attention_weights`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

        {/* Slide 6: RNN Applications */}
        <div data-slide className="min-h-[500px]" id="applications">
          <Title order={2} className="mb-6">RNN Applications and Examples</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-purple-50">
                <Title order={4} className="mb-3">Sequence Classification</Title>
                <Code block language="python">{`class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len) token indices
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state
        final_hidden = hidden[-1]  # (batch, hidden_dim)
        output = self.dropout(final_hidden)
        return self.classifier(output)

# Example: Sentiment Analysis
sentiment_model = SequenceClassifier(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    num_classes=2  # Positive/Negative
)`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-yellow-50">
                <Title order={4} className="mb-3">Sequence-to-Sequence</Title>
                <Code block language="python">{`class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, embed_dim)
        
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        self.output_proj = nn.Linear(hidden_dim, tgt_vocab_size)
    
    def encode(self, src):
        embedded = self.encoder_embed(src)
        _, (hidden, cell) = self.encoder(embedded)
        return hidden, cell
    
    def decode(self, tgt, encoder_states):
        hidden, cell = encoder_states
        embedded = self.decoder_embed(tgt)
        
        output, _ = self.decoder(embedded, (hidden, cell))
        return self.output_proj(output)
    
    def forward(self, src, tgt):
        encoder_states = self.encode(src)
        decoder_output = self.decode(tgt, encoder_states)
        return decoder_output`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
          
          <Paper className="p-4 bg-orange-50 mt-4">
            <Title order={4} className="mb-3">Time Series Prediction</Title>
            <Code block language="python">{`class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last time step for prediction
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction

# Multi-step ahead prediction
class MultiStepPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_steps):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.predictors = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_steps)
        ])
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = []
        for i, predictor in enumerate(self.predictors):
            pred = predictor(lstm_out[:, -1, :])  # Last hidden state
            predictions.append(pred)
        return torch.cat(predictions, dim=1)`}</Code>
          </Paper>
        </div>

        {/* Slide 7: Training and Optimization */}
        <div data-slide className="min-h-[500px]" id="training-optimization">
          <Title order={2} className="mb-6">Training RNNs: Best Practices</Title>
          
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Paper className="p-4 bg-green-50">
                <Title order={4} className="mb-3">Gradient Clipping</Title>
                <Code block language="python">{`import torch.nn.utils as utils

def train_step(model, data_loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        inputs, targets = batch
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        utils.clip_grad_norm_(model.parameters(), clip_value)
        # Or clip by value
        # utils.clip_grad_value_(model.parameters(), clip_value)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# Gradient monitoring
def monitor_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm`}</Code>
              </Paper>
            </Grid.Col>
            
            <Grid.Col span={6}>
              <Paper className="p-4 bg-red-50">
                <Title order={4} className="mb-3">Handling Variable Sequences</Title>
                <Code block language="python">{`from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class VariableLengthRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, sequences, lengths):
        # sequences: padded tensor (batch, max_len, input_size)
        # lengths: actual lengths of each sequence
        
        # Pack sequences for efficient processing
        packed = pack_padded_sequence(
            sequences, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Process with LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack the sequences
        output, lengths = pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # Use last actual output for each sequence
        batch_size = output.size(0)
        last_outputs = output[range(batch_size), lengths-1]
        
        return self.fc(last_outputs)

# Custom collate function for DataLoader
def collate_variable_length(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    
    # Pad sequences
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    
    return padded_sequences, torch.tensor(lengths), torch.tensor(labels)`}</Code>
              </Paper>
            </Grid.Col>
          </Grid>
        </div>

      </Stack>
    </Container>
  );
};

export default RecurrentNetworks;