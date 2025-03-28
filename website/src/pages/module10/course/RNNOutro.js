import React from 'react';
import { Box, Title, Stack, Text } from "@mantine/core";
import CodeBlock from "components/CodeBlock";

const RNNOutro = () => {
  return (
    <Stack spacing="xl" className="w-full">
      {/* Code snippet for time series prediction */}
      <Title order={2} id="torch-example-prediction">Torch Example Sequence Prediction</Title>
      <Text>
        Below is a simple example of how to prepare a 2D time series for an RNN and predict the next 2 time steps:
      </Text>
      
      <CodeBlock
        language="python"
        code={`import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 2D time series (e.g., temperature and humidity over time)
# Shape: [time_steps, features]
time_steps = 100
features = 2  # 2D time series

# Generate synthetic data
np.random.seed(42)
time_series = np.zeros((time_steps, features))
time_series[:, 0] = np.sin(np.linspace(0, 10, time_steps)) + np.random.normal(0, 0.1, time_steps)  # Feature 1
time_series[:, 1] = np.cos(np.linspace(0, 10, time_steps)) + np.random.normal(0, 0.1, time_steps)  # Feature 2

# Function to create sequences from time series data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 2):  # -2 for predicting 2 steps ahead
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+2]  # Next 2 time steps
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences for training
seq_length = 10  # Use 10 time steps to predict the next 2
X, y = create_sequences(time_series, seq_length)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

print(f"Input sequences shape: {X_tensor.shape}")   # [90, 10, 2]
print(f"Target sequences shape: {y_tensor.shape}")  # [90, 2, 2]

# Define a simple RNN model
class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_steps, num_layers=1):
        super(TimeSeriesRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # RNN layer
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size * output_steps)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_size]
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, (h0, c0))
        
        # Extract the outputs from the last time step
        out = self.fc(out[:, -1, :])
        
        # Reshape to [batch_size, output_steps, output_size]
        out = out.view(batch_size, self.output_steps, -1)
        
        return out

# Instantiate the model
input_size = features  # Number of features in input
hidden_size = 20       # Number of features in hidden state
output_size = features  # Number of features in output
output_steps = 2       # Predict 2 time steps ahead
model = TimeSeriesRNN(input_size, hidden_size, output_size, output_steps)

# Make a prediction with the model
with torch.no_grad():
    test_input = X_tensor[:1]  # Take first sequence as test
    print(f"Test input shape: {test_input.shape}")  # [1, 10, 2]
    
    prediction = model(test_input)
    print(f"Prediction shape: {prediction.shape}")  # [1, 2, 2]
    
    # Compare with actual values
    print("Predicted next 2 time steps:")
    print(prediction[0].numpy())
    print("Actual next 2 time steps:")
    print(y_tensor[0].numpy())`}
      />
    </Stack>
  );
};

export default RNNOutro;